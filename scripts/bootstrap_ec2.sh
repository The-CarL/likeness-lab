#!/usr/bin/env bash
# ============================================================================
# bootstrap_ec2.sh — Set up a fresh g5.xlarge DLAMI for LoRA training
# ============================================================================
# Run this immediately after SSHing into a new EC2 instance:
#   bash bootstrap_ec2.sh
#
# What it does:
#   1. Verifies GPU availability
#   2. Updates system packages
#   3. Activates the PyTorch conda environment
#   4. Clones and installs ai-toolkit
#   5. Creates project directory structure
#   6. Logs into HuggingFace and pre-downloads model weights
#   7. Clones this repo (lora-training) if not already present
# ============================================================================
set -euo pipefail

SECONDS=0
REPO_URL="https://github.com/The-CarL/likeness-lab.git"
PROJECT_DIR="$HOME/likeness-lab"
TOOLKIT_DIR="$HOME/ai-toolkit"

# ── Colors for output ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Verify GPU ───────────────────────────────────────────────────
info "Checking GPU availability..."
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Is this a GPU instance with NVIDIA drivers?"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
info "GPU detected: ${GPU_NAME} (${GPU_MEM})"

if ! echo "$GPU_NAME" | grep -qi "A10G\|A10\|A100\|H100\|L4\|T4"; then
    warn "Expected an A10G GPU but found: ${GPU_NAME}. Training may need config adjustments."
fi

# ── 2. Update system packages ───────────────────────────────────────
info "Updating system packages..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq

# ── 3. Activate PyTorch environment ──────────────────────────────────
info "Activating PyTorch environment..."
# Newer DLAMIs use a virtualenv at /opt/pytorch; older ones use conda.
if [ -f "/opt/pytorch/bin/activate" ]; then
    source /opt/pytorch/bin/activate
    info "Activated /opt/pytorch virtualenv."
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate pytorch 2>/dev/null || conda activate pytorch_p310 2>/dev/null || conda activate base
    info "Activated conda environment."
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    conda activate pytorch 2>/dev/null || conda activate base
    info "Activated conda environment."
else
    error "Could not find PyTorch environment. Is this a PyTorch DLAMI?"
fi

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── 4. Clone and install ai-toolkit ─────────────────────────────────
info "Setting up ai-toolkit..."
if [ -d "$TOOLKIT_DIR" ]; then
    info "ai-toolkit already exists at $TOOLKIT_DIR, pulling latest..."
    cd "$TOOLKIT_DIR"
    git pull
    git submodule update --init --recursive
else
    git clone --recurse-submodules https://github.com/ostris/ai-toolkit.git "$TOOLKIT_DIR"
    cd "$TOOLKIT_DIR"
fi

info "Installing ai-toolkit dependencies..."
pip install -r requirements.txt

# ── 5. Create project directory structure ────────────────────────────
info "Creating project directories..."
mkdir -p "$PROJECT_DIR"/{datasets/{raw_photos,kontext_pairs/{target,control},standard},outputs,logs}

# ── 6. HuggingFace login and model download ──────────────────────────
info "Setting up HuggingFace access..."
pip install -q huggingface_hub[cli]

if huggingface-cli whoami &>/dev/null; then
    HF_USER=$(huggingface-cli whoami | head -1)
    info "Already logged in as: $HF_USER"
else
    warn "You need to log in to HuggingFace to download gated models."
    warn "Make sure you've accepted the license at:"
    warn "  https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev"
    echo ""
    huggingface-cli login
fi

info "Pre-downloading Flux Kontext Dev model weights..."
info "This may take 15-30 minutes on first run (~24 GB download)."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'black-forest-labs/FLUX.1-Kontext-dev',
    ignore_patterns=['*.md', '*.txt'],
)
print('Model download complete.')
"

# ── 7. Clone this repo ──────────────────────────────────────────────
info "Setting up training repo..."
if [ -d "$PROJECT_DIR/.git" ]; then
    info "Repo already cloned at $PROJECT_DIR, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    # If we created the directory structure above, move it aside briefly
    if [ -d "$PROJECT_DIR/datasets" ]; then
        TEMP_DATASETS=$(mktemp -d)
        mv "$PROJECT_DIR/datasets" "$TEMP_DATASETS/"
        mv "$PROJECT_DIR/outputs" "$TEMP_DATASETS/" 2>/dev/null || true
        mv "$PROJECT_DIR/logs" "$TEMP_DATASETS/" 2>/dev/null || true
        rm -rf "$PROJECT_DIR"
        git clone "$REPO_URL" "$PROJECT_DIR"
        # Restore data directories
        mv "$TEMP_DATASETS/datasets" "$PROJECT_DIR/"
        mv "$TEMP_DATASETS/outputs" "$PROJECT_DIR/" 2>/dev/null || true
        mv "$TEMP_DATASETS/logs" "$PROJECT_DIR/" 2>/dev/null || true
        rm -rf "$TEMP_DATASETS"
    else
        git clone "$REPO_URL" "$PROJECT_DIR"
        mkdir -p "$PROJECT_DIR"/{datasets/{raw_photos,kontext_pairs/{target,control},standard},outputs,logs}
    fi
fi

# Symlink ai-toolkit's run.py for convenience
if [ ! -L "$PROJECT_DIR/run.py" ]; then
    ln -sf "$TOOLKIT_DIR/run.py" "$PROJECT_DIR/run.py"
    info "Symlinked ai-toolkit run.py into project directory."
fi

# ── Summary ──────────────────────────────────────────────────────────
ELAPSED=$SECONDS
info "============================================"
info "  Bootstrap complete! (${ELAPSED}s elapsed)"
info "============================================"
echo ""
info "Project directory:  $PROJECT_DIR"
info "AI-Toolkit:         $TOOLKIT_DIR"
info "GPU:                $GPU_NAME ($GPU_MEM)"
echo ""
info "Next steps:"
info "  1. Upload your photos:  scp -r photos/ ec2-user@<this-ip>:~/likeness-lab/datasets/raw_photos/"
info "  2. Prepare pairs:       cd ~/likeness-lab && python scripts/prepare_kontext_pairs.py"
info "  3. Generate captions:   python scripts/caption_dataset.py --mode kontext"
info "  4. Start training:      bash scripts/launch_training.sh"
echo ""
