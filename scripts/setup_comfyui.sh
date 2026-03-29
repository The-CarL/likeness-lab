#!/usr/bin/env bash
# ============================================================================
# setup_comfyui.sh — Install and start ComfyUI on EC2 for validation
# ============================================================================
# Usage:
#   bash scripts/setup_comfyui.sh              # Full install + start
#   bash scripts/setup_comfyui.sh --start-only # Just start if already installed
#
# Access from your local machine:
#   ssh -L 8188:localhost:8188 ubuntu@<ec2-ip>
#   Then open http://localhost:8188 in your browser
# ============================================================================
set -euo pipefail

COMFYUI_DIR="$HOME/ComfyUI"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_ONLY=false

# ── Colors ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --start-only|-s)
            START_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: bash scripts/setup_comfyui.sh [--start-only]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"; exit 1
            ;;
    esac
done

# ── Activate conda ──────────────────────────────────────────────────
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate pytorch 2>/dev/null || conda activate pytorch_p310 2>/dev/null || conda activate base

# ── Start-only mode ─────────────────────────────────────────────────
if [ "$START_ONLY" = true ]; then
    if [ ! -d "$COMFYUI_DIR" ]; then
        echo "ERROR: ComfyUI not found at $COMFYUI_DIR. Run without --start-only first."
        exit 1
    fi
    info "Starting ComfyUI..."
    cd "$COMFYUI_DIR"
    python main.py --listen 127.0.0.1 --port 8188 &
    sleep 3
    info "ComfyUI running at http://127.0.0.1:8188"
    info "SSH tunnel from local: ssh -L 8188:localhost:8188 ubuntu@<ec2-ip>"
    exit 0
fi

# ── Full installation ────────────────────────────────────────────────
info "Installing ComfyUI..."

if [ -d "$COMFYUI_DIR" ]; then
    info "ComfyUI already exists, pulling latest..."
    cd "$COMFYUI_DIR"
    git pull
else
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    cd "$COMFYUI_DIR"
fi

info "Installing ComfyUI dependencies..."
pip install -r requirements.txt

# ── Install ComfyUI-Manager ─────────────────────────────────────────
info "Installing ComfyUI-Manager..."
MANAGER_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-Manager"
if [ -d "$MANAGER_DIR" ]; then
    cd "$MANAGER_DIR"
    git pull
else
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$MANAGER_DIR"
fi

# ── Set up model directories ────────────────────────────────────────
info "Setting up model directories..."
mkdir -p "$COMFYUI_DIR/models/"{unet,clip,vae,loras}

# ── Symlink/download Flux Kontext Dev model ──────────────────────────
# Check if the model was already downloaded by ai-toolkit via HuggingFace cache
HF_CACHE_DIR="$HOME/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-Kontext-dev"

if [ -d "$HF_CACHE_DIR" ]; then
    info "Found cached Flux Kontext Dev model, creating symlinks..."

    # Find the latest snapshot
    SNAPSHOT_DIR=$(ls -td "$HF_CACHE_DIR/snapshots/"*/ 2>/dev/null | head -1)
    if [ -n "$SNAPSHOT_DIR" ]; then
        # Symlink the transformer (unet) weights
        if [ -d "${SNAPSHOT_DIR}transformer" ]; then
            ln -sf "${SNAPSHOT_DIR}transformer" "$COMFYUI_DIR/models/unet/flux-kontext-dev"
            info "Symlinked transformer to models/unet/flux-kontext-dev"
        fi

        # Symlink text encoders
        if [ -d "${SNAPSHOT_DIR}text_encoder" ]; then
            ln -sf "${SNAPSHOT_DIR}text_encoder" "$COMFYUI_DIR/models/clip/flux-kontext-clip-l"
            info "Symlinked text_encoder to models/clip/"
        fi
        if [ -d "${SNAPSHOT_DIR}text_encoder_2" ]; then
            ln -sf "${SNAPSHOT_DIR}text_encoder_2" "$COMFYUI_DIR/models/clip/flux-kontext-t5"
            info "Symlinked text_encoder_2 to models/clip/"
        fi

        # Symlink VAE
        if [ -d "${SNAPSHOT_DIR}vae" ]; then
            ln -sf "${SNAPSHOT_DIR}vae" "$COMFYUI_DIR/models/vae/flux-kontext-vae"
            info "Symlinked vae to models/vae/"
        fi
    fi
else
    warn "Flux Kontext Dev model not found in HuggingFace cache."
    warn "You may need to download it manually or it will be downloaded on first use."
    warn "Run: huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev"
fi

# ── Symlink trained LoRA weights ─────────────────────────────────────
# Find the latest .safetensors in outputs/
LORA_FILES=$(find "$PROJECT_DIR/outputs" -name "*.safetensors" -type f 2>/dev/null | sort -t/ -k1 | tail -5)
if [ -n "$LORA_FILES" ]; then
    info "Found trained LoRA weights, symlinking to ComfyUI..."
    while IFS= read -r lora_file; do
        lora_name=$(basename "$lora_file")
        ln -sf "$lora_file" "$COMFYUI_DIR/models/loras/$lora_name"
        info "  Linked: $lora_name"
    done <<< "$LORA_FILES"
else
    warn "No trained LoRA weights found yet. They'll appear after training."
    info "You can manually symlink later: ln -s ~/likeness-lab/outputs/<name>.safetensors ~/ComfyUI/models/loras/"
fi

# ── Start ComfyUI ────────────────────────────────────────────────────
info "Starting ComfyUI on port 8188..."
cd "$COMFYUI_DIR"
python main.py --listen 127.0.0.1 --port 8188 &
COMFYUI_PID=$!
sleep 5

# Check if it started successfully
if kill -0 $COMFYUI_PID 2>/dev/null; then
    info "============================================"
    info "  ComfyUI is running! (PID: $COMFYUI_PID)"
    info "============================================"
    echo ""
    info "Access from your local machine:"
    info "  1. Open a NEW terminal on your Mac"
    info "  2. Run: ssh -L 8188:localhost:8188 ubuntu@<ec2-public-ip>"
    info "  3. Open http://localhost:8188 in your browser"
    echo ""
    info "To stop ComfyUI: kill $COMFYUI_PID"
    info "To restart later: bash scripts/setup_comfyui.sh --start-only"
else
    echo "ERROR: ComfyUI failed to start. Check the output above for errors."
    exit 1
fi
