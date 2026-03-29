#!/usr/bin/env bash
# ============================================================================
# launch_training.sh — Kick off LoRA training with ai-toolkit
# ============================================================================
# Usage:
#   bash scripts/launch_training.sh                           # Uses default Kontext config
#   bash scripts/launch_training.sh --config flux_dev_standard.yaml  # Use standard config
# ============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
CONFIG_NAME="kontext_personal_likeness.yaml"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT_DIR="$HOME/ai-toolkit"
COST_PER_HOUR=1.006  # g5.xlarge on-demand price in us-east-1

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash scripts/launch_training.sh [--config CONFIG_FILE]"
            echo ""
            echo "Options:"
            echo "  --config, -c   Config filename in configs/ (default: kontext_personal_likeness.yaml)"
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            ;;
    esac
done

CONFIG_PATH="${PROJECT_DIR}/configs/${CONFIG_NAME}"

# ── Pre-flight checks ───────────────────────────────────────────────
info "Running pre-flight checks..."

# Check GPU
if ! nvidia-smi &>/dev/null; then
    error "No GPU detected. Are you running this on a GPU instance?"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)
info "GPU: ${GPU_NAME} (${GPU_MEM} free)"

# Check config exists
if [ ! -f "$CONFIG_PATH" ]; then
    error "Config not found: $CONFIG_PATH"
fi
info "Config: $CONFIG_PATH"

# Check ai-toolkit
if [ ! -f "$TOOLKIT_DIR/run.py" ]; then
    error "ai-toolkit not found at $TOOLKIT_DIR. Run bootstrap_ec2.sh first."
fi

# Check dataset
if echo "$CONFIG_NAME" | grep -q "kontext"; then
    DATASET_DIR="${PROJECT_DIR}/datasets/kontext_pairs/target"
    CONTROL_DIR="${PROJECT_DIR}/datasets/kontext_pairs/control"
    if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
        error "Kontext dataset not found at $DATASET_DIR. Run prepare_kontext_pairs.py first."
    fi
    TARGET_COUNT=$(find "$DATASET_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) | wc -l)
    CONTROL_COUNT=$(find "$CONTROL_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) | wc -l)
    CAPTION_COUNT=$(find "$DATASET_DIR" -type f -name "*.txt" | wc -l)
    info "Dataset: ${TARGET_COUNT} target images, ${CONTROL_COUNT} control images, ${CAPTION_COUNT} captions"
    if [ "$TARGET_COUNT" -ne "$CONTROL_COUNT" ]; then
        warn "Target and control image counts don't match! Check your dataset."
    fi
    if [ "$CAPTION_COUNT" -lt "$TARGET_COUNT" ]; then
        warn "Some images are missing captions. Run caption_dataset.py first."
    fi
else
    DATASET_DIR="${PROJECT_DIR}/datasets/standard"
    if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
        error "Standard dataset not found at $DATASET_DIR."
    fi
    IMG_COUNT=$(find "$DATASET_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) | wc -l)
    info "Dataset: ${IMG_COUNT} images"
fi

# ── Activate environment ─────────────────────────────────────────────
info "Activating PyTorch environment..."
if [ -f "/opt/pytorch/bin/activate" ]; then
    source /opt/pytorch/bin/activate
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate pytorch 2>/dev/null || conda activate base
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    conda activate pytorch 2>/dev/null || conda activate base
fi

# ── Launch training ──────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${PROJECT_DIR}/logs/training_${TIMESTAMP}.log"
mkdir -p "${PROJECT_DIR}/logs"

info "============================================"
info "  Starting LoRA training"
info "  Config:  ${CONFIG_NAME}"
info "  Log:     ${LOG_FILE}"
info "  Started: $(date)"
info "============================================"
echo ""

SECONDS=0

# Run training from the ai-toolkit directory with our config
cd "$TOOLKIT_DIR"
python run.py "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"

ELAPSED=$SECONDS
HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)
COST=$(echo "scale=2; $HOURS * $COST_PER_HOUR" | bc)

echo ""
info "============================================"
info "  Training complete!"
info "  Duration: $(printf '%dh %dm %ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))"
info "  Est cost: \$${COST} (${HOURS} hours @ \$${COST_PER_HOUR}/hr)"
info "  Log:      ${LOG_FILE}"
info "============================================"
info ""
info "Next steps:"
info "  1. Check samples in outputs/ directory"
info "  2. Set up ComfyUI:  bash scripts/setup_comfyui.sh"
info "  3. Or download:     bash scripts/download_weights.sh <ec2-ip>"
