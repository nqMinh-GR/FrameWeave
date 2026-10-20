"""CPU frame sampling for RelJoin training — no temp MP4 round-trip.

LanguageBind Video-LLaVA's processor.__call__ only accepts a video path and
re-decodes + linspace-subsamples. Training instead samples K frames once,
pads to native length, and applies processor.transform on a C,T,H,W tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data_perception import QASample


def sample_rgb_frames(video_path: str, K: int) -> tuple[list[np.ndarray], list[float]]:
    """Uniformly sample K RGB frames. Same time/index formula as the old MP4 path."""
    import cv2

    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) or 1
    duration = total / fps
    times = [duration * i / max(K - 1, 1) for i in range(K)]
    indices = [min(total - 1, max(0, int(round(t * fps)))) for t in times]
    frames: list[np.ndarray] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed frame {idx} from {video_path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(np.ascontiguousarray(rgb))
    finally:
        cap.release()
    return frames, times


def pad_frames_to_native(
    frames: list[np.ndarray],
    times: list[float],
    native_frames: int,
) -> tuple[list[np.ndarray], list[float], torch.Tensor]:
    """Pad/truncate to native_frames; mask marks real (non-duplicated) frames."""
    if native_frames < 1:
        raise ValueError(f"native_frames must be >= 1, got {native_frames}")
    if not frames:
        raise RuntimeError("empty frame list")
    K = len(times)
    if K != len(frames):
        raise ValueError(f"frames ({len(frames)}) != times ({K})")
    mask = torch.zeros(native_frames, dtype=torch.bool)
    mask[: min(K, native_frames)] = True
    if K == native_frames:
        return frames, list(times), mask
    if K > native_frames:
        return frames[:native_frames], list(times[:native_frames]), mask
    padded = list(frames)
    padded_times = list(times)
    last_t = padded_times[-1] if padded_times else 0.0
    while len(padded) < native_frames:
        padded.append(padded[-1].copy())
        padded_times.append(last_t)
    return padded, padded_times, mask


def frames_to_pil(frames: Sequence[np.ndarray]):
    """RGB HxWxC uint8 frames → list of PIL Images."""
    from PIL import Image

    out = []
    for frame in frames:
        arr = np.ascontiguousarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got {arr.shape}")
        out.append(Image.fromarray(arr))
    return out


def pil_frames_to_rgb(frames: Sequence) -> list[np.ndarray]:
    """PIL or ndarray frames → contiguous RGB uint8 arrays."""
    from PIL import Image

    out: list[np.ndarray] = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            arr = np.asarray(frame.convert("RGB"))
        else:
            arr = np.ascontiguousarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got {arr.shape}")
        out.append(arr)
    return out


def frames_to_cthw(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """RGB HxWxC uint8 frames → C,T,H,W (LanguageBind opencv layout before transform)."""
    if not frames:
        raise RuntimeError("empty frame list")
    cols = []
    for frame in frames:
        arr = np.ascontiguousarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 RGB, got {arr.shape}")
        cols.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(cols, dim=1)


def resolve_video_transform(transform_or_processor) -> Callable:
    t = getattr(transform_or_processor, "transform", None)
    if callable(t):
        return t
    if callable(transform_or_processor):
        return transform_or_processor
    raise TypeError(
        "Need a callable transform or a processor with .transform "
        f"(got {type(transform_or_processor)!r})"
    )


def apply_video_transform(transform_or_processor, cthw: torch.Tensor) -> torch.Tensor:
    """Apply LanguageBind-style video transform. Returns [1, C, T, H, W]."""
    if cthw.dim() != 4:
        raise ValueError(f"expected C,T,H,W got {tuple(cthw.shape)}")
    transform = resolve_video_transform(transform_or_processor)
    out = transform(cthw)
    if isinstance(out, dict):
        out = out.get("video", out.get("pixel_values"))
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"transform returned {type(out)!r}, expected Tensor")
    if out.dim() == 4:
        out = out.unsqueeze(0)
    if out.dim() != 5:
        raise ValueError(f"expected [1,C,T,H,W] after unsqueeze, got {tuple(out.shape)}")
    return out


def load_pixel_values(
    video_path: str,
    K: int,
    native_frames: int,
    transform_or_processor,
) -> dict:
    """Sample, pad, transform. CPU tensors only."""
    frames, times = sample_rgb_frames(video_path, K)
    frames, times, frame_mask = pad_frames_to_native(frames, times, native_frames)
    pixel_values = apply_video_transform(transform_or_processor, frames_to_cthw(frames))
    timestamps = torch.tensor(times, dtype=torch.float32)
    return {
        "pixel_values": pixel_values.contiguous(),
        "timestamps": timestamps,
        "frame_mask": frame_mask,
        "K_real": int(frame_mask.sum().item()),
        "native_frames": int(native_frames),
    }


@dataclass
class FrameJob:
    sample: QASample
    K: int
    choices: list[str]
    answer_index: int


class PerceptionFrameDataset(Dataset):
    """CPU-only: decode + pad + transform. Tokenize / embed_tokens stay on the main process."""

    def __init__(
        self,
        jobs: list[FrameJob],
        native_frames: int,
        transform_or_processor,
    ) -> None:
        self.jobs = jobs
        self.native_frames = int(native_frames)
        self.transform_or_processor = transform_or_processor

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int) -> dict:
        job = self.jobs[idx]
        packed = load_pixel_values(
            job.sample.video_path,
            job.K,
            self.native_frames,
            self.transform_or_processor,
        )
        packed.update(
            {
                "question": job.sample.question,
                "choices": job.choices,
                "answer_index": int(job.answer_index),
                "sample_id": job.sample.sample_id,
                "original_choices": list(job.sample.choices),
                "original_answer_index": int(job.sample.answer_index),
            }
        )
        return packed


def _worker_init(_worker_id: int) -> None:
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass


def make_frame_loader(
    jobs: list[FrameJob],
    native_frames: int,
    transform_or_processor,
    num_workers: int,
    *,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """batch_size=None so variable metadata is not collated. Falls back to 0 workers."""
    ds = PerceptionFrameDataset(jobs, native_frames, transform_or_processor)
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())
    n = max(0, int(num_workers))
    kwargs: dict = {
        "dataset": ds,
        "batch_size": None,
        "num_workers": n,
        "pin_memory": pin_memory,
        "shuffle": False,
    }
    if n > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = False
        kwargs["worker_init_fn"] = _worker_init
    try:
        return DataLoader(**kwargs)
    except Exception as exc:
        print(f"DataLoader num_workers={n} failed ({exc}); falling back to 0")
        return DataLoader(
            ds,
            batch_size=None,
            num_workers=0,
            pin_memory=pin_memory,
            shuffle=False,
        )


def iter_frame_items(
    jobs: list[FrameJob],
    native_frames: int,
    transform_or_processor,
    num_workers: int,
    *,
    pin_memory: Optional[bool] = None,
):
    """Yield dataset items; if worker prefetch dies after CUDA init, retry with 0 workers."""
    n = max(0, int(num_workers))
    loader = make_frame_loader(
        jobs,
        native_frames,
        transform_or_processor,
        n,
        pin_memory=pin_memory,
    )
    if n <= 0:
        yield from loader
        return
    try:
        it = iter(loader)
        first = next(it)
    except StopIteration:
        return
    except Exception as exc:
        print(f"Frame prefetch failed ({exc}); retrying with num_workers=0")
        yield from make_frame_loader(
            jobs,
            native_frames,
            transform_or_processor,
            0,
            pin_memory=pin_memory,
        )
        return
    yield first
    yield from it
