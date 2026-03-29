# Personal Likeness LoRA Training (Flux Kontext Dev)

Train a personal likeness LoRA using [Flux Kontext Dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) on AWS EC2. Go from a folder of selfies to a `.safetensors` file that can insert you into any scene from a reference photo, with local ComfyUI validation via SSH tunnel.

This repo uses **Kontext-style training** — paired before/after images with instructional captions — so the resulting LoRA works with Flux Kontext's edit-by-instruction paradigm. A standard Flux Dev LoRA config is included as a fallback.

## Prerequisites

- **AWS account** with permissions to launch EC2 instances (g5.xlarge)
- **HuggingFace account** with access to [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (accept the gated model license)
- **20-30 personal photos** — varied angles, lighting, expressions, backgrounds
- **SSH key pair** configured in AWS
- **Local machine** with ComfyUI installed (for final validation)

## Quick Start

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for the complete step-by-step guide.

```bash
# 1. Launch a g5.xlarge EC2 instance (see RUNBOOK for the full command)
# 2. SSH in and bootstrap
ssh ubuntu@<ec2-ip>
bash bootstrap_ec2.sh

# 3. Upload your photos from local
scp -r datasets/raw_photos/* ubuntu@<ec2-ip>:~/likeness-lab/datasets/raw_photos/

# 4. Prepare dataset, caption, and train
python scripts/prepare_kontext_pairs.py
python scripts/caption_dataset.py --mode kontext
bash scripts/launch_training.sh

# 5. Download the trained LoRA to your Mac
bash scripts/download_weights.sh <ec2-ip>
```

## Cost Estimate

A full training run (1500 steps) on a g5.xlarge typically takes 1-3 hours, costing **$1-3** at on-demand pricing ($1.006/hr). Add ~$1 for setup/validation time. Total: **$2-5 per run**.

## Repo Structure

```
likeness-lab/
├── README.md                              # This file
├── configs/
│   ├── kontext_personal_likeness.yaml     # Kontext LoRA training config (primary)
│   └── flux_dev_standard.yaml             # Standard Flux Dev LoRA config (fallback)
├── scripts/
│   ├── bootstrap_ec2.sh                   # Full EC2 setup from fresh DLAMI
│   ├── prepare_kontext_pairs.py           # Generate before/after pairs for Kontext
│   ├── caption_dataset.py                 # Auto-caption dataset with trigger word
│   ├── launch_training.sh                 # Start training with the right config
│   ├── setup_comfyui.sh                   # Install ComfyUI on EC2 for validation
│   └── download_weights.sh               # Pull trained weights to local machine
├── workflows/                             # ComfyUI workflow JSONs (add manually)
├── docs/
│   └── RUNBOOK.md                         # Step-by-step operational guide
├── datasets/                              # (gitignored) Training data
│   ├── raw_photos/                        # Your original photos
│   ├── kontext_pairs/{target,control}/    # Prepared paired dataset
│   └── standard/                          # Standard LoRA dataset
└── outputs/                               # (gitignored) Trained weights & samples
```

## Training Tool

This repo wraps [Ostris AI-Toolkit](https://github.com/ostris/ai-toolkit), the standard open-source tool for training Flux LoRAs. The configs and scripts here handle the setup, dataset prep, and operational workflow around it.

## License

MIT
