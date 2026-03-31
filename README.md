# Personal Likeness LoRA Training

Train a personal likeness LoRA on AWS EC2. Go from a folder of selfies to a `.safetensors` file that generates you in any style or scene.

## Supported Models

| Model | Config | VRAM | Best For |
|-------|--------|------|----------|
| **Flux 2 Klein 9B** | `klein9b_likeness.yaml` | 24 GB (A10G) | Recommended — best quality/cost ratio |
| Flux 1 [dev] | `flux_dev_standard.yaml` | 24 GB (A10G) | Legacy fallback |
| Flux Kontext Dev | `kontext_personal_likeness.yaml` | 24 GB (A10G) | Edit-based (requires input image) |

## Prerequisites

- **AWS account** with a g5.xlarge instance (24 GB A10G GPU)
- **HuggingFace account** with access to gated model (accept license)
- **15-25 personal photos** — varied angles, lighting, expressions. Clean-shaven if that's the look you want. The model learns whatever is consistent across your photos.
- **SSH key pair** configured in AWS

## Quick Start

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for the full step-by-step guide.

```bash
# 1. Launch EC2 and SSH in (see RUNBOOK for instance launch command)
ssh -i ~/.ssh/your-key.pem ubuntu@<ec2-ip>

# 2. Bootstrap
git clone https://github.com/The-CarL/likeness-lab.git ~/likeness-lab
cd ~/likeness-lab && bash scripts/bootstrap_ec2.sh

# 3. Upload photos from local
scp -r datasets/raw_photos/* ubuntu@<ec2-ip>:~/likeness-lab/datasets/standard/

# 4. Caption and train
python scripts/caption_dataset.py --mode standard
bash scripts/launch_training.sh --config klein9b_likeness.yaml

# 5. Generate images
python scripts/inference.py --prompt "photo of ohwx_person at the beach, sunset"

# 6. Download weights to local
bash scripts/download_weights.sh <ec2-ip>
```

## Key Lessons Learned

- **Captions should describe the scene, NOT the person.** The LoRA learns appearance as the unspoken constant across images.
- **Use a non-dictionary trigger word** like `ohwx_person`, not a real name.
- **Flux needs higher LR than SDXL** — 1e-4 minimum, not 1e-5.
- **24 GB GPUs need quantization** — the `patch_aitoolkit.sh` script handles this for Flux 1 models. Klein 9B handles it natively.

## Cost Estimate

A full training run on g5.xlarge ($1.006/hr): **$3-5** including setup and inference.

## Repo Structure

```
likeness-lab/
├── configs/
│   ├── klein9b_likeness.yaml              # Flux 2 Klein 9B (recommended)
│   ├── flux_dev_standard.yaml             # Flux 1 Dev standard LoRA
│   └── kontext_personal_likeness.yaml     # Flux Kontext (edit-based)
├── scripts/
│   ├── bootstrap_ec2.sh                   # EC2 setup from fresh DLAMI
│   ├── patch_aitoolkit.sh                 # Patches for 24 GB GPU compat (Flux 1)
│   ├── prepare_kontext_pairs.py           # Kontext paired dataset prep
│   ├── caption_dataset.py                 # Auto-caption with trigger word
│   ├── launch_training.sh                 # Start training
│   ├── inference.py                       # Generate images with trained LoRA
│   ├── setup_comfyui.sh                   # ComfyUI on EC2 for validation
│   └── download_weights.sh               # Pull weights to local machine
├── docs/
│   └── RUNBOOK.md                         # Step-by-step operational guide
├── datasets/                              # (gitignored) Training data
└── outputs/                               # (gitignored) Weights & generated images
```

## License

MIT
