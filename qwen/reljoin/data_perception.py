"""PerceptionTest subset from LLaVA-Video-178K — external warm-start only.

HERBench must never enter the training dataloader.

Actual annotation schema (pinned revision):
  {
    "id": "perceptiontest_video_2779",
    "video": "perception_test/videos/video_2779.mp4",
    "conversations": [
      {"from": "human", "value": "<image>\\nQ?\\nA. ..\\nB. ..\\nC. ..\\nPlease provide..."},
      {"from": "gpt", "value": "A. ..."},
      ... more Q/A turns ...
    ]
  }
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import tarfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .config import (
    HERBENCH_ID,
    LITE_TRAIN,
    LITE_VAL,
    OVERFIT_N,
    SEED,
    SMOKE_TRAIN,
    SMOKE_VAL,
    TRAIN_ANNOTATION,
    TRAIN_DATASET_ID,
    TRAIN_DATASET_REVISION,
    TRAIN_VIDEO_ARCHIVE,
)

# Bump when schema adapter changes — written into train_manifest for Colab sanity checks
DATA_PARSER_VERSION = "perception_conversations_mc_v3"

_OPT_LINE = re.compile(r"^([A-H])\.\s*(.*)\s*$")
_ANS_LETTER = re.compile(r"^([A-H])(?:\.|\s|$)", re.IGNORECASE)


@dataclass
class QASample:
    sample_id: str
    video_id: str
    video_path: str
    question: str
    choices: list[str]
    answer: str  # letter A/B/C/...
    answer_index: int
    meta: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _pick(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def parse_mc_human(text: str) -> tuple[str, list[str]]:
    """Parse question + choices from LLaVA-Video MC human turn."""
    text = re.sub(r"</?image>", "", str(text), flags=re.IGNORECASE).strip()
    q_lines: list[str] = []
    choices: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("please provide your answer"):
            continue
        m = _OPT_LINE.match(line)
        if m:
            choices.append(m.group(2).strip())
            continue
        if not choices:
            q_lines.append(line)
    question = "\n".join(q_lines).strip()
    return question, choices


def parse_answer_letter(text: str, n_choices: int) -> Optional[tuple[str, int]]:
    s = str(text).strip()
    m = _ANS_LETTER.match(s)
    if m:
        letter = m.group(1).upper()
        idx = ord(letter) - ord("A")
        if 0 <= idx < n_choices:
            return letter, idx
    # bare letter
    if len(s) == 1 and s.upper() in "ABCDEFGH":
        idx = ord(s.upper()) - ord("A")
        if 0 <= idx < n_choices:
            return s.upper(), idx
    return None


def expand_annotation_item(raw: dict, idx: int) -> tuple[list[dict], str]:
    """Expand one JSON item into one-or-more MC samples. Returns (samples, fail_reason)."""
    video = _pick(raw, "video", "video_path", "video_name", "path")
    if video is None:
        return [], "missing_video_field"

    base_id = str(_pick(raw, "id", "sample_id", "question_id", default=f"pt_{idx}"))
    video_str = str(video)
    video_id = Path(video_str).stem
    out: list[dict] = []

    # Preferred: multi-turn conversations with inlined A/B/C options
    convs = raw.get("conversations")
    if isinstance(convs, list) and convs:
        turn = 0
        i = 0
        while i < len(convs) - 1:
            a, b = convs[i], convs[i + 1]
            if a.get("from") in ("human", "user") and b.get("from") in ("gpt", "assistant"):
                question, choices = parse_mc_human(a.get("value") or a.get("text") or "")
                ans = parse_answer_letter(b.get("value") or b.get("text") or "", len(choices))
                if question and len(choices) >= 2 and ans is not None:
                    letter, answer_index = ans
                    out.append(
                        {
                            "sample_id": f"{base_id}__item{idx}__q{turn}",
                            "video_id": video_id,
                            "video_rel": video_str,
                            "question": question,
                            "choices": choices,
                            "answer": letter,
                            "answer_index": answer_index,
                        }
                    )
                    turn += 1
                i += 2
                continue
            i += 1
        if out:
            return out, ""
        return [], "conversation_parse_failed"

    # Fallback: flat fields
    question = _pick(raw, "question", "query", "prompt", "text")
    choices = _pick(raw, "choices", "options", "candidates")
    answer = _pick(raw, "answer", "label", "gt", "ground_truth", "correct")
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except json.JSONDecodeError:
            choices = [c.strip() for c in choices.split("\n") if c.strip()]
    if question and choices and len(choices) >= 2 and answer is not None:
        ans = parse_answer_letter(str(answer), len(choices))
        if ans is None:
            # match by text
            for j, c in enumerate(choices):
                if str(c).strip().lower() == str(answer).strip().lower():
                    ans = ("ABCDEFGH"[j], j)
                    break
        if ans is not None:
            letter, answer_index = ans
            return (
                [
                    {
                        "sample_id": f"{base_id}__item{idx}",
                        "video_id": video_id,
                        "video_rel": video_str,
                        "question": str(question).strip(),
                        "choices": [str(c).strip() for c in choices],
                        "answer": letter,
                        "answer_index": answer_index,
                    }
                ],
                "",
            )
    return [], "flat_schema_failed"


def download_perception_lite(root: Path) -> tuple[Path, Path]:
    """Download PerceptionTest train annotation + videos_1.tar.gz (not HERBench)."""
    from huggingface_hub import hf_hub_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ann_path = Path(
        hf_hub_download(
            repo_id=TRAIN_DATASET_ID,
            repo_type="dataset",
            revision=TRAIN_DATASET_REVISION,
            filename=TRAIN_ANNOTATION,
            local_dir=str(root / "hf_cache"),
        )
    )
    archive_path = Path(
        hf_hub_download(
            repo_id=TRAIN_DATASET_ID,
            repo_type="dataset",
            revision=TRAIN_DATASET_REVISION,
            filename=TRAIN_VIDEO_ARCHIVE,
            local_dir=str(root / "hf_cache"),
        )
    )
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation missing after download: {ann_path}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Video archive missing after download: {archive_path}")
    print(
        "PerceptionTest TRAIN shard (LLaVA-Video-178K). "
        "This is not HERBench; HERBench is eval-only."
    )
    print(f"Annotation: {ann_path} ({ann_path.stat().st_size} bytes)")
    print(f"Archive: {archive_path} ({archive_path.stat().st_size / 2**30:.2f} GiB)")

    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    existing = list(videos_dir.rglob("*.mp4"))
    marker = videos_dir / ".videos_1_extracted"
    need_extract = len(existing) < 50 or not marker.exists()
    if need_extract:
        print(f"Extracting {archive_path.name} → {videos_dir} (this can take several minutes)...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=videos_dir)
        marker.write_text(archive_path.name, encoding="utf-8")
        existing = list(videos_dir.rglob("*.mp4"))
    print(f"PerceptionTest videos available: {len(existing)} under {videos_dir}")
    if existing:
        print("  example:", existing[0])
    else:
        raise RuntimeError(
            f"No mp4 after extract under {videos_dir}. "
            f"Archive exists={archive_path.is_file()} size={archive_path.stat().st_size}."
        )
    # Quick path-check against annotation style
    probe = videos_dir / "perception_test" / "videos"
    if not probe.is_dir():
        print(
            "WARNING: expected folder missing:",
            probe,
            "— resolve_video_path will fall back to rglob by filename.",
        )
    return ann_path, videos_dir


def resolve_video_path(video_rel: str, videos_dir: Path) -> Optional[Path]:
    rel = Path(str(video_rel))
    candidates = [
        videos_dir / rel,
        videos_dir / rel.name,
        videos_dir / "videos" / rel.name,
        videos_dir / "perception_test" / "videos" / rel.name,
        videos_dir / "0_30_s_perceptiontest" / rel.name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    # Indexed stem lookup (built once per process)
    index = getattr(resolve_video_path, "_stem_index", None)
    if index is None or getattr(resolve_video_path, "_index_root", None) != str(videos_dir):
        index = {}
        for m in videos_dir.rglob("*"):
            if m.suffix.lower() in {".mp4", ".webm", ".mkv", ".avi"}:
                index.setdefault(m.stem, []).append(m)
        resolve_video_path._stem_index = index  # type: ignore[attr-defined]
        resolve_video_path._index_root = str(videos_dir)  # type: ignore[attr-defined]
    hits = index.get(rel.stem) or index.get(rel.name)
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0].resolve()
    pref = [m for m in hits if "perception" in str(m).lower()]
    return (pref[0] if pref else hits[0]).resolve()


def diagnose_perception_data(root: Path) -> None:
    """Print data readiness for Colab debugging."""
    root = Path(root)
    print("=== RelJoin PerceptionTest diagnose ===")
    print("DATA_PARSER_VERSION:", DATA_PARSER_VERSION)
    ann, videos_dir = download_perception_lite(root / "perceptiontest_lite")
    samples, excluded = load_and_filter_samples(ann, videos_dir)
    print("usable_samples:", len(samples))
    print("unique_videos:", len({s.video_id for s in samples}))
    print("exclusion_reasons:", dict(Counter(e["reason"] for e in excluded)))
    if samples:
        print("example_sample:", samples[0].to_dict())
    train, val = select_for_mode(samples, "overfit")
    print("overfit_train_n:", len(train))
    print("=== diagnose OK ===")


def load_and_filter_samples(ann_path: Path, videos_dir: Path) -> tuple[list[QASample], list[dict]]:
    raw = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("data", "annotations", "samples", "questions"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError(f"Unrecognized annotation schema keys: {list(raw)[:20]}")
    assert isinstance(raw, list), type(raw)
    print("Annotation schema sample keys:", list(raw[0].keys()) if raw else [])
    print("Num raw video items:", len(raw))
    if raw:
        # Show one parse preview
        preview, reason = expand_annotation_item(raw[0], 0)
        print(f"First-item expand: n={len(preview)} reason={reason or 'ok'}")
        if preview:
            print("  sample0 question:", preview[0]["question"][:120])
            print("  sample0 choices:", preview[0]["choices"])
            print("  sample0 answer:", preview[0]["answer"])

    kept: list[QASample] = []
    excluded: list[dict] = []
    n_mp4 = len(list(videos_dir.rglob("*.mp4")))
    for i, item in enumerate(raw):
        adapted_list, fail = expand_annotation_item(item, i)
        if not adapted_list:
            excluded.append({"idx": i, "reason": fail or "schema_adapt_failed", "id": item.get("id")})
            continue
        for adapted in adapted_list:
            vpath = resolve_video_path(adapted["video_rel"], videos_dir)
            if vpath is None:
                excluded.append(
                    {
                        "idx": i,
                        "sample_id": adapted["sample_id"],
                        "reason": "missing_video",
                        "video_rel": adapted["video_rel"],
                        "n_mp4_available": n_mp4,
                    }
                )
                continue
            kept.append(
                QASample(
                    sample_id=adapted["sample_id"],
                    video_id=adapted["video_id"],
                    video_path=str(vpath),
                    question=adapted["question"],
                    choices=adapted["choices"],
                    answer=adapted["answer"],
                    answer_index=adapted["answer_index"],
                    meta={"video_rel": adapted["video_rel"]},
                )
            )

    reasons = Counter(e["reason"] for e in excluded)
    ids = [s.sample_id for s in kept]
    if len(ids) != len(set(ids)):
        dup = [sid for sid, n in Counter(ids).items() if n > 1]
        raise RuntimeError(f"Duplicate sample_id after parse: {dup[:8]}")
    print(f"Kept {len(kept)} QA / excluded entries {len(excluded)} reasons={dict(reasons)}")
    print(f"DATA_PARSER_VERSION={DATA_PARSER_VERSION}")
    if len(kept) == 0:
        raise RuntimeError(
            "No PerceptionTest training samples retained. "
            f"mp4_under_videos_dir={n_mp4}, exclusion_reasons={dict(reasons)}. "
            "If missing_video dominates, videos_1.tar.gz may not be extracted or paths mismatch. "
            "If conversation_parse_failed dominates, annotation schema changed."
        )
    return kept, excluded


def split_by_video(
    samples: list[QASample],
    n_train: int,
    n_val: int,
    seed: int = SEED,
) -> tuple[list[QASample], list[QASample]]:
    """Video-disjoint split. Closest sizes if exact would break disjointness."""
    rng = random.Random(seed)
    by_video: dict[str, list[QASample]] = defaultdict(list)
    for s in samples:
        by_video[s.video_id].append(s)
    video_ids = list(by_video.keys())
    rng.shuffle(video_ids)

    train, val = [], []
    for vid in video_ids:
        group = by_video[vid]
        if len(train) < n_train:
            if len(train) + len(group) <= n_train + max(8, n_train // 20):
                train.extend(group)
                continue
        if len(val) < n_val:
            val.extend(group)
            continue
        if len(train) < n_train:
            train.extend(group)

    train_vids = {s.video_id for s in train}
    val_vids = {s.video_id for s in val}
    assert train_vids.isdisjoint(val_vids), "Train/val video intersection must be empty"
    return train, val


def select_for_mode(
    samples: list[QASample],
    mode: str,
    seed: int = SEED,
) -> tuple[list[QASample], list[QASample]]:
    if not samples:
        raise RuntimeError(f"select_for_mode({mode}): empty sample list")
    if mode == "overfit":
        rng = random.Random(seed)
        by_video: dict[str, list[QASample]] = defaultdict(list)
        for s in samples:
            by_video[s.video_id].append(s)
        vids = list(by_video.keys())
        rng.shuffle(vids)
        train = []
        for v in vids:
            train.extend(by_video[v])
            if len(train) >= OVERFIT_N:
                break
        train = train[:OVERFIT_N]
        if not train:
            raise RuntimeError("overfit split produced 0 samples")
        return train, []
    if mode == "smoke":
        train, val = split_by_video(samples, SMOKE_TRAIN, SMOKE_VAL, seed=seed)
        return train[:SMOKE_TRAIN], val[:SMOKE_VAL]
    if mode in {"lite", "full"}:
        return split_by_video(samples, LITE_TRAIN, LITE_VAL, seed=seed)
    raise ValueError(f"No PerceptionTest split for mode={mode}")


def save_manifests(
    train: list[QASample],
    val: list[QASample],
    excluded: list[dict],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(train) == 0:
        raise RuntimeError("Refusing to write empty train_manifest.json")
    (out_dir / "train_manifest.json").write_text(
        json.dumps(
            {
                "parser_version": DATA_PARSER_VERSION,
                "n": len(train),
                "video_ids": sorted({s.video_id for s in train}),
                "sample_ids": [s.sample_id for s in train],
                "samples": [s.to_dict() for s in train],
                "note": (
                    "External uniform-frame warm-start and engineering validation, "
                    "not definitive multi-evidence supervision."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "val_manifest.json").write_text(
        json.dumps(
            {
                "n": len(val),
                "video_ids": sorted({s.video_id for s in val}),
                "sample_ids": [s.sample_id for s in val],
                "samples": [s.to_dict() for s in val],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Compact excluded summary (full list can be huge)
    reason_counts = Counter(e.get("reason", "?") for e in excluded)
    (out_dir / "excluded_samples.json").write_text(
        json.dumps(
            {
                "n_excluded": len(excluded),
                "reason_counts": dict(reason_counts),
                "examples": excluded[:50],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    train_vids = {s.video_id for s in train}
    val_vids = {s.video_id for s in val}
    assert train_vids.isdisjoint(val_vids)


def assert_no_herbench_in_training(samples: list[QASample]) -> None:
    for s in samples:
        path_l = s.video_path.lower()
        if "herbench" in path_l or HERBENCH_ID.lower().replace("/", "_") in path_l:
            raise AssertionError(f"HERBench path leaked into training: {s.video_path}")
        if "herbench" in s.sample_id.lower():
            raise AssertionError(f"HERBench sample_id leaked: {s.sample_id}")


def stable_int_seed(key: str) -> int:
    """Process-stable seed. Do not use Python hash() — it is salted per process."""
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def permute_choices(
    choices: list[str],
    answer_index: int,
    rng: random.Random,
) -> tuple[list[str], int, list[int]]:
    n = len(choices)
    perm = list(range(n))
    rng.shuffle(perm)
    new_choices = [choices[i] for i in perm]
    new_answer_index = perm.index(answer_index)
    return new_choices, new_answer_index, perm


def deterministic_permute(
    choices: list[str],
    answer_index: int,
    seed_key: str,
) -> tuple[list[str], int, list[int]]:
    rng = random.Random(seed_key)
    return permute_choices(choices, answer_index, rng)


def unit_test_option_permutation() -> None:
    choices = ["red", "blue", "green"]
    answer_index = 1  # blue
    rng = random.Random(0)
    for _ in range(50):
        new_c, new_i, perm = permute_choices(choices, answer_index, rng)
        assert new_c[new_i] == "blue"
        assert [choices[p] for p in perm] == new_c
    print("option_permutation unit test OK")


def unit_test_perception_schema_parse() -> None:
    raw = {
        "id": "perceptiontest_video_2779",
        "video": "perception_test/videos/video_2779.mp4",
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\nWhat is the person preparing?\n"
                    "A. a tea\nB. a sandwich\nC. a salad\n"
                    "Please provide your answer by stating the letter followed by the full option."
                ),
            },
            {"from": "gpt", "value": "A. a tea"},
            {
                "from": "human",
                "value": (
                    "What action not related to making tea did the person do?\n"
                    "A. hitting the spoon with a fork\n"
                    "B. showing an egg to the camera\n"
                    "C. putting eggs in a bowl\n"
                    "Please provide your answer by stating the letter followed by the full option."
                ),
            },
            {"from": "gpt", "value": "A. hitting the spoon with a fork"},
        ],
        "data_source": "0_30_s_perceptiontest",
    }
    samples, reason = expand_annotation_item(raw, 0)
    assert reason == "", reason
    assert len(samples) == 2, samples
    assert samples[0]["answer"] == "A"
    assert samples[0]["choices"][0] == "a tea"
    assert samples[1]["answer"] == "A"
    assert samples[0]["sample_id"] == "perceptiontest_video_2779__item0__q0"
    assert samples[1]["sample_id"] == "perceptiontest_video_2779__item0__q1"
    other, _ = expand_annotation_item(raw, 7)
    assert other[0]["sample_id"] == "perceptiontest_video_2779__item7__q0"
    assert other[0]["sample_id"] != samples[0]["sample_id"]
    print("perception_schema_parse unit test OK")
