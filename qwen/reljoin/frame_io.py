"""CPU frame sampling for RelJoin training on Qwen2.5-VL multi-image inputs.

Each evidence frame stays a PIL image at native aspect. Do not pad to a
fixed native_frames count and do not pack frames as one native video tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data_perception import QASample


def sample_rgb_frames(video_path: str, K: int) -> tuple[list[np.ndarray], list[float]]:
    """Uniformly sample K RGB frames."""
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


def load_pil_frames(video_path: str, K: int) -> dict:
    """Sample K real frames. No pad-to-native-8."""
    frames, times = sample_rgb_frames(video_path, K)
    k_real = len(frames)
    frame_mask = torch.ones(k_real, dtype=torch.bool)
    timestamps = torch.tensor(times, dtype=torch.float32)
    return {
        "frames": frames_to_pil(frames),
        "timestamps": timestamps,
        "frame_mask": frame_mask,
        "K_real": int(k_real),
    }


@dataclass
class FrameJob:
    sample: QASample
    K: int
    choices: list[str]
    answer_index: int


class PerceptionFrameDataset(Dataset):
    """CPU-only: decode K PIL frames. Tokenize / LLM embeds stay on the main process."""

    def __init__(self, jobs: list[FrameJob]) -> None:
        self.jobs = jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int) -> dict:
        job = self.jobs[idx]
        packed = load_pil_frames(job.sample.video_path, job.K)
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
    num_workers: int,
    *,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """batch_size=None so variable metadata is not collated."""
    ds = PerceptionFrameDataset(jobs)
    if pin_memory is None:
        pin_memory = False  # PIL objects cannot be pinned
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
            pin_memory=False,
            shuffle=False,
        )


def iter_frame_items(
    jobs: list[FrameJob],
    num_workers: int,
    *,
    pin_memory: Optional[bool] = None,
):
    """Yield dataset items; if worker prefetch dies, retry with 0 workers."""
    n = max(0, int(num_workers))
    loader = make_frame_loader(jobs, n, pin_memory=pin_memory)
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
        yield from make_frame_loader(jobs, 0, pin_memory=False)
        return
    yield first
    yield from it
