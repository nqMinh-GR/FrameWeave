# FrameWeave

Public source for FrameWeave Hybrid v6: a gated residual on frozen visual tokens, trained on PerceptionTest and evaluated on HERBench and NExT-GQA.

Three backbones each keep their own `reljoin` package. They are not importable from one Python path at the same time.

| Backbone | Code | Environment |
| --- | --- | --- |
| Ovis2.5-9B | [`ovis/`](ovis) | Python 3.11, torch 2.4.0, transformers 4.51.3 |
| Qwen2.5-VL-7B | [`qwen/`](qwen) | Python 3.11, torch 2.4.0, transformers 4.51.3 |
| Video-LLaVA-7B | [`videollava/`](videollava) | Python 3.11, torch 2.0.1+cu118, transformers 4.31.0 |
| Ovis ablations | [`ovis_ablation/`](ovis_ablation) | same pins as Ovis; imports `ovis/reljoin` |

Pins are listed in [`requirements/`](requirements). Colab notebooks still install those packages themselves.

## Colab

Open a notebook from this repo, then Run All. The notebook clones this repository into `/content/FrameWeave` and symlinks the code into `/content`. Do not upload `.py` files.

On the next Run All in the same runtime, the notebook runs `git pull --ff-only`. If `/content/reljoin` is a real uploaded folder, restart the runtime before Run All.

Checkpoints, datasets, and model weights are not in git. Eval notebooks still read them from `/content` or Google Drive. The TSO evidence-flow notebook uses the published `checkpoints/ovis/best_external_val/Hybrid.pt` and the logs in `ovis/assets/evidence_flow/`. It downloads Ovis weights and HERBench videos from Hugging Face.

### Qwen

- [Train multi-seed](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/qwen/notebooks/FrameWeave_Qwen_Multiseed_Train.ipynb)
- [HERBench multi-seed](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/qwen/notebooks/FrameWeave_Qwen_Multiseed_HERBench_Eval.ipynb)
- [Train](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/qwen/notebooks/RelJoin_Residual_Qwen25VL_Experiment.ipynb)
- [HERBench](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/qwen/notebooks/RelJoin_Residual_Qwen25VL_HERBench_Eval.ipynb)
- [HERBench full](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/qwen/notebooks/RelJoin_Residual_Qwen25VL_HERBench_Full_Eval.ipynb)
- [NExT-GQA](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/qwen/notebooks/RelJoin_Residual_Qwen25VL_NExTGQA_Eval.ipynb)

### Ovis

- [Train](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_Experiment.ipynb)
- [HERBench](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_HERBench_Eval.ipynb)
- [HERBench full](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_HERBench_Full_Eval.ipynb)
- [HERBench setup on Drive](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/HERBench_Full_Drive_Setup.ipynb)
- [NExT-GQA](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_NExTGQA_Eval.ipynb)
- [Residual direction](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_HERBench_Residual_Direction.ipynb)
- [TSO abstention](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_TSO_Abstention.ipynb)
- [Frame visualizations](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_Frame_Visualizations.ipynb)
- [TSO evidence flow](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis/notebooks/RelJoin_Residual_Ovis25_TSO_Evidence_Flow.ipynb)

### Ovis ablations

Ablation notebooks link `ovis/reljoin` together with `ablation/`, `configs/`, and `run_ablation.py`.

- [Ablation train](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis_ablation/notebooks/FrameWeave_Ovis_Ablation_Experiment.ipynb)
- [Ablation HERBench](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis_ablation/notebooks/FrameWeave_Ovis_Ablation_HERBench_Eval.ipynb)
- [Multi-seed train](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis_ablation/notebooks/FrameWeave_Ovis_Multiseed_Train.ipynb)
- [Multi-seed HERBench](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis_ablation/notebooks/FrameWeave_Ovis_Multiseed_HERBench_Eval.ipynb)
- [Frame-only train](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis_ablation/notebooks/FrameWeave_Ovis_FrameOnly_Train.ipynb)
- [Frame-only HERBench](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/ovis_ablation/notebooks/FrameWeave_Ovis_FrameOnly_HERBench_Eval.ipynb)

### Video-LLaVA

- [Train](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/videollava/notebooks/RelJoin_Residual_VideoLLaVA_Experiment.ipynb)
- [HERBench](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/videollava/notebooks/RelJoin_Residual_HERBench_Eval.ipynb)
- [HERBench full](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/videollava/notebooks/RelJoin_Residual_HERBench_Full_Eval.ipynb)
- [HERBench setup on Drive](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/videollava/notebooks/HERBench_Full_Drive_Setup.ipynb)
- [NExT-GQA](https://colab.research.google.com/github/nqMinh-GR/FrameWeave/blob/main/videollava/notebooks/RelJoin_Residual_VideoLLaVA_NExTGQA_Eval.ipynb)

## Local

```bash
git clone https://github.com/nqMinh-GR/FrameWeave.git
cd FrameWeave/qwen
python run_reljoin.py --help
```

Run a backbone from its own directory so `import reljoin` resolves to that backbone. For ablations:

```bash
cd FrameWeave
PYTHONPATH=ovis:ovis_ablation python ovis_ablation/run_ablation.py --help
```

## Updating code

Edit files in this repository and push. The next Colab Run All pulls that commit. Uploading `reljoin/` to `/content` is no longer part of the workflow.
