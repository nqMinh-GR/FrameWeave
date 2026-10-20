# Published checkpoints

Eval notebooks use these files first. A notebook falls back to Google Drive only when the file is absent from this clone. Train notebooks do not load this directory.

The Ovis ablation seed-42 suite (`token_mlp`, `global_only`, `local_only`, `hybrid`, `hybrid_no_time`) was a retrain. Those adapters are not in this repository, and `FrameWeave_Ovis_Ablation_HERBench_Eval.ipynb` does not fetch them from GitHub or Drive.

`epoch_*` and `resume.pt` are not published. `best` is the selected checkpoint. `last` is the fallback the backbone notebooks already accept.

## Ovis2.5-9B, default seed 42

- `checkpoints/ovis/best_external_val/Hybrid.pt`
- `checkpoints/ovis/last/Hybrid.pt`

Used by the Ovis HERBench, HERBench full, NExT-GQA, residual-direction, TSO abstention, and frame-visualization notebooks.

## Qwen2.5-VL-7B, default seed 42

`SEED = 42` in `qwen/reljoin/config.py`. This Hybrid run is the default Qwen checkpoint.

- `checkpoints/qwen/best_external_val/Hybrid.pt`
- `checkpoints/qwen/last/Hybrid.pt`

The same two files are also stored at `checkpoints/qwen_multiseed/seed_42/` so that seed has all three variants. The multiseed eval notebook still runs seeds 1, 100, and 2027.

## Qwen multiseed

`checkpoints/qwen_multiseed/seed_{1,100,2027}/{best_external_val,last}/{GlobalTimeFix,Hybrid,HybridNoGlobal}.pt`

Seed 42 adds `GlobalTimeFix` and `HybridNoGlobal` from the extra run, plus `Hybrid` from the default run above.

## Video-LLaVA-7B

- `checkpoints/videollava/best_external_val/Hybrid.pt`
- `checkpoints/videollava/last/Hybrid.pt`

## Ovis multiseed

`checkpoints/ovis_multiseed/seed_{1,100,2027}/{global_only,local_only,hybrid}/{best,last}/adapter.pt`

## Ovis frame-only, seed 42

- `checkpoints/ovis_frame_only/seed_42/frame_only/best/adapter.pt`
- `checkpoints/ovis_frame_only/seed_42/frame_only/last/adapter.pt`
