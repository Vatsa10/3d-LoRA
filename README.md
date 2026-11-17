# TRELLIS.2 geometry LoRA fine-tuning on Colab

`trellis2_lora_finetune_colab.ipynb` runs the pipeline from the Hugging Face blog post
[The Definitive Fine-Tuning Guide for 3D Generation Models](https://huggingface.co/blog/3d-lora-finetuning)
end to end in Google Colab, without Docker or HF Jobs.

It LoRA-adapts the two geometry flows of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
(sparse-structure flow and shape flow) on footwear meshes from Google Scanned Objects, evaluates
base vs LoRA on held-out objects, and exports textured GLBs. Training, preprocessing and evaluation
use the blog's companion code ([suvadityamuk/trellis2-finetuning](https://github.com/suvadityamuk/trellis2-finetuning))
unchanged; the notebook rebuilds the blog's Docker environment natively.

> Status: the notebook has been syntax-checked and its helper logic tested with mock data, but it
> has not been executed on a GPU. See [Troubleshooting](#troubleshooting) for the likely failure points.

## Requirements

| | |
|---|---|
| Runtime | Colab **L4** or **A100**, High-RAM. T4 is rejected (16 GB VRAM) |
| VRAM | 24 GB minimum |
| HF account | License accepted for [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (gated) |
| HF token | Colab secret `HF_TOKEN`. Read access is enough; write access only if `PUSH_TO` is set |
| Disk | Roughly 25 GB: TRELLIS.2-4B weights, DINOv3, the ~1.9 GB dataset, build artifacts |
| Google Drive | Optional, recommended: stores adapters and compiled wheels for resume |

## Quick start

1. Open the notebook in Colab and select an L4 or A100 runtime.
2. Accept the DINOv3 license and add `HF_TOKEN` under the key icon (Secrets).
3. Adjust the config cell if needed.
4. Runtime -> Run all.

To validate the environment before committing GPU hours, set `USE_PUBLISHED_ADAPTERS = True`.
That skips training and runs evaluation and inference with the blog's released adapters.

## What each section does

| # | Section | What happens |
|---|---|---|
| 0 | Config | All knobs in one cell |
| 1 | Runtime checks | GPU and VRAM check, batch size selection, Drive mount, token |
| 2 | Environment | Clones TRELLIS.2 at commit `75fbf01`, applies the release patch, installs torch 2.6.0/cu124, transformers 5.18.0, flash-attn 2.7.3, compiles nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel |
| 3 | Hub check | Confirms the token and DINOv3 access before any long step |
| 4 | Data | Downloads the preprocessed latents, or preprocesses your own meshes |
| 5 | Training | `finetune.py --flow ss`, then `--flow shape`; loss curves parsed from the logs |
| 6 | Adapters | Picks the newest local checkpoints, or downloads the published ones |
| 7 | Evaluation | `eval_geometry.py`: Chamfer distance, F-score@0.02, normal consistency, base vs LoRA |
| 8 | Inference | Image -> textured GLB for base and LoRA, with an inline 3D viewer |
| 9 | Download | Zips adapters, GLBs and metrics |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TRAIN_SS`, `TRAIN_SHAPE` | `True` | Train each flow's adapter |
| `USE_PUBLISHED_ADAPTERS` | `False` | Skip training; use `suvadityamuk/trellis-2-finetune-gso` |
| `RUN_EVAL`, `RUN_INFERENCE` | `True` | Run sections 7 and 8 |
| `LATENTS` | `suvadityamuk/gso-footwear-trellis2-latents` | Preprocessed dataset repo |
| `USE_OWN_MESHES`, `MESHES` | `False`, `/content/my_meshes` | Preprocess a local folder or Hub dataset of meshes instead |
| `HOLDOUT_FRAC` | `0.05` | Fraction held out when preprocessing your own meshes |
| `MAX_STEPS` | `1000` | Optimizer steps per flow. The blog uses 3000 |
| `I_SAVE` | `250` | Checkpoint interval |
| `BATCH_SIZE`, `GRAD_ACCUM` | auto | 4 x 4 at 70 GB+ VRAM, otherwise 2 x 8. Effective batch stays 16 |
| `LR` | `1e-4` | AdamW learning rate |
| `USE_DRIVE` | `True` | Persist results and wheels to `MyDrive/trellis2_lora` |
| `PUSH_TO` | empty | Model repo that receives every checkpoint |
| `TRACKIO_SPACE` | empty | Space for a hosted Trackio dashboard; empty prints metrics only |
| `PIPELINE_TYPE` | `512` | Generation resolution. `1024_cascade` needs more VRAM |
| `EVAL_MAX_OBJECTS` | `0` | Cap on evaluated objects; 0 means all |
| `INFER_IMAGE` | empty | Your own input image; empty uses the first held-out render |
| `SEED` | `42` | Shared by the base and LoRA generations |

## Training setup

| | Sparse-structure flow | Shape flow |
|---|---|---|
| Checkpoint | `ss_flow_img_dit_1_3B_64_bf16` | `slat_flow_img2shape_dit_1_3B_512_bf16` |
| LoRA rank / alpha | 8 / 16 | 16 / 32 |
| Target modules | `to_qkv`, `to_out`, `to_q`, `to_kv`, `mlp.mlp.0`, `mlp.mlp.2` | `to_qkv`, `to_out`, `to_q`, `to_kv` |
| Timestep sampling | logit-normal (mean 1, std 1) | uniform |

Shared settings:

- Objective: rectified-flow velocity MSE, `x_t = (1-t) x0 + (sigma_min + (1-sigma_min) t) eps`,
  target `(1-sigma_min) eps - x0`, `sigma_min = 1e-5`, timestep scaled by 1000.
- Optimizer: AdamW, lr 1e-4, betas (0.9, 0.95), weight decay 0.01, grad clip 1.0.
- Precision: LoRA weights in fp32, forward pass under bf16 autocast, gradient checkpointing on the DiT blocks.
- Conditioning: frozen DINOv3-L on one random render per sample, 10% classifier-free-guidance dropout.
- Adapters are injected with `peft.inject_adapter_in_model`, because TRELLIS.2 uses a custom forward pass.
- The shape flow's MLPs are `SparseLinear` layers that PEFT cannot wrap, hence attention-only at a higher rank.
- The texture flow is not trained; textures come from the base model.

## Time budget

| Step | Estimate |
|---|---|
| Environment, first run | 20-40 min (compiles CUDA extensions; my estimate) |
| Environment, later runs | A few minutes (cached wheels on Drive) |
| Training, blog reference | 3000 steps on an A100 80 GB: 7.2 h sparse-structure, 3.6 h shape |
| Training on an L4 | Slower than the reference; 3000 steps per flow will not fit one session |

Every training cell resumes from the newest `adapter-<step>.safetensors`, so after a disconnect
run all cells again with the same config. Optimizer state restarts; the LoRA weights do not.

## Outputs

With `USE_DRIVE = True` everything below lives under `MyDrive/trellis2_lora/`; otherwise under `/content/work/`.

| Path | Content |
|---|---|
| `results/ss_lora/adapter-<step>.safetensors` | Sparse-structure LoRA weights |
| `results/shape_lora/adapter-<step>.safetensors` | Shape LoRA weights |
| `results/<run>/config.json`, `train.log` | Hyperparameters and training log |
| `eval/metrics.csv` | Per-object and mean metrics, base and LoRA |
| `generated/base.glb`, `generated/lora.glb` | Textured meshes from the same image and seed |
| `wheels/` | Compiled extension wheels, keyed by Python version and GPU architecture |

## Reference results

Blog numbers at 3000 steps on 9 held-out shoes:

| Metric | Base | LoRA |
|---|---|---|
| F-score@0.02 | 0.276 | 0.803 |
| Chamfer distance | 0.174 | 0.033 |
| Normal consistency | 0.530 | 0.804 |

Expect smaller gains at the notebook's default of 1000 steps.

## Using your own meshes

1. Put meshes in a folder or a Hub dataset repo. Any format Blender imports works
   (`.glb`, `.gltf`, `.obj` with its `.mtl` and textures, `.fbx`, `.ply`, `.stl`, `.usd`).
   Each file is one object.
2. Set `USE_OWN_MESHES = True` and `MESHES` to the folder or repo id.
3. Run all. Section 4 installs Blender 3.0.1 and 4.5.1 and runs the eight preprocessing stages
   (index, mesh dump, O-Voxel dual grids, shape latents, sparse-structure latents, 16 Cycles renders,
   held-out split, GLB export). It resumes if re-run.

Training needs ground-truth 3D geometry; images alone are not enough.

## Using the adapters elsewhere

```python
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file
from trellis2.pipelines import Trellis2ImageTo3DPipeline

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

def load_lora(model, path, rank, alpha, targets):
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=targets)
    inject_adapter_in_model(cfg, model)
    set_peft_model_state_dict(model, load_file(path))

attn = ["to_qkv", "to_out", "to_q", "to_kv"]
load_lora(pipeline.models["sparse_structure_flow_model"], "ss_lora/adapter-3000.safetensors",
          8, 16, attn + ["mlp.mlp.0", "mlp.mlp.2"])
load_lora(pipeline.models["shape_slat_flow_model_512"], "shape_lora/adapter-3000.safetensors",
          16, 32, attn)

mesh = pipeline.run(image, seed=42, pipeline_type="512")[0]
```

For `1024_cascade`, also inject the shape adapter into `shape_slat_flow_model_1024`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvcc not found` | The runtime has no CUDA toolkit under `/usr/local/cuda`. Switch to a standard Colab GPU runtime |
| torch 2.6.0 fails to install | Colab's Python version has no torch 2.6.0 wheel. Check `python --version`; 3.9 to 3.13 are supported |
| flash-attn builds from source for an hour | No prebuilt wheel matched the Python version; the script falls back to compiling. Let it finish once |
| Extension build fails after a partial run | Delete the matching folder under `wheels/` and `/tmp/extensions`, then re-run section 2 |
| Wheels stop loading after switching GPU type | Wheels are compiled per GPU architecture; a new folder is built automatically for the new GPU |
| `DINOv3 access DENIED` | Accept the license with the same account that owns the token |
| CUDA out of memory in training | Set `BATCH_SIZE = 1`, `GRAD_ACCUM = 16` |
| CUDA out of memory in eval or inference | Keep `PIPELINE_TYPE = "512"`; restart the runtime to clear leftover processes |
| `could not find both adapters` | The published adapter repo layout differs from the filename pattern; download the two files manually and set `SS_ADAPTER` / `SHAPE_ADAPTER` in section 6 |
| Blender download fails | The mirror is unreachable; retry, or place both Blender tarballs' contents under `/tmp/` |
| GLB preview is blank | The file is over the inline limit or the viewer script was blocked; download the GLB instead |

## Limits

- Geometry only. Textures are produced by the unmodified base model.
- The held-out split groups near-duplicates using Google Scanned Objects naming rules; with other
  naming it is a plain seeded random split.
- Resuming restores LoRA weights but not optimizer state.

## Credits and licenses

- TRELLIS.2: Xiang et al., Microsoft, MIT. [Paper](https://arxiv.org/abs/2512.14692)
- Fine-tuning code, containers, dataset and adapters: the blog's companion repo and Hub repos by `suvadityamuk`
- Google Scanned Objects: Google Research, CC-BY-4.0
- DINOv3: Meta, under its own license; downloaded at run time
- Blender: GPL
