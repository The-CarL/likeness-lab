#!/usr/bin/env bash
# ============================================================================
# download_weights.sh — Pull trained LoRA weights from EC2 to your local Mac
# ============================================================================
# Run this FROM YOUR LOCAL MAC (not on EC2).
#
# Usage:
#   bash scripts/download_weights.sh <ec2-ip>
#   bash scripts/download_weights.sh <ec2-ip> --key ~/.ssh/my-key.pem
#   bash scripts/download_weights.sh <ec2-ip> --user ubuntu
#   bash scripts/download_weights.sh <ec2-ip> --latest   # Only download the latest file
# ============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
EC2_USER="ubuntu"
SSH_KEY=""
REMOTE_DIR="~/likeness-lab/outputs"
LOCAL_COMFYUI_DIR="$HOME/ComfyUI/models/loras"
LOCAL_BACKUP_DIR="$HOME/lora-backups"
LATEST_ONLY=false

# ── Colors ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Usage ────────────────────────────────────────────────────────────
usage() {
    echo "Usage: bash scripts/download_weights.sh <ec2-ip> [options]"
    echo ""
    echo "Options:"
    echo "  --key, -k PATH      SSH key file (default: uses ssh-agent or default key)"
    echo "  --user, -u USER     SSH user (default: ubuntu)"
    echo "  --latest             Only download the most recent .safetensors file"
    echo "  --help, -h           Show this help"
    exit 0
}

# ── Parse arguments ──────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    usage
fi

EC2_HOST="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --key|-k)    SSH_KEY="$2"; shift 2 ;;
        --user|-u)   EC2_USER="$2"; shift 2 ;;
        --latest)    LATEST_ONLY=true; shift ;;
        --help|-h)   usage ;;
        *)           error "Unknown argument: $1" ;;
    esac
done

# Build SSH/SCP options
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

# ── Test connectivity ────────────────────────────────────────────────
info "Testing SSH connection to ${EC2_USER}@${EC2_HOST}..."
if ! ssh $SSH_OPTS "${EC2_USER}@${EC2_HOST}" "echo 'Connected'" 2>/dev/null; then
    error "Cannot connect to ${EC2_USER}@${EC2_HOST}. Check the IP and your SSH key."
fi

# ── List available weights ───────────────────────────────────────────
info "Finding .safetensors files on EC2..."
REMOTE_FILES=$(ssh $SSH_OPTS "${EC2_USER}@${EC2_HOST}" \
    "find ${REMOTE_DIR} -name '*.safetensors' -type f 2>/dev/null | sort")

if [ -z "$REMOTE_FILES" ]; then
    error "No .safetensors files found in ${REMOTE_DIR} on EC2."
fi

echo ""
echo "Available weights:"
echo "$REMOTE_FILES" | while IFS= read -r f; do
    SIZE=$(ssh $SSH_OPTS "${EC2_USER}@${EC2_HOST}" "du -h '$f' | cut -f1" 2>/dev/null)
    echo "  $f ($SIZE)"
done
echo ""

# ── Select files to download ────────────────────────────────────────
if [ "$LATEST_ONLY" = true ]; then
    DOWNLOAD_FILES=$(echo "$REMOTE_FILES" | tail -1)
    info "Downloading latest file only..."
else
    DOWNLOAD_FILES="$REMOTE_FILES"
    FILE_COUNT=$(echo "$REMOTE_FILES" | wc -l | tr -d ' ')
    info "Downloading all ${FILE_COUNT} files..."
fi

# ── Create local directories ────────────────────────────────────────
mkdir -p "$LOCAL_COMFYUI_DIR" 2>/dev/null || warn "Could not create $LOCAL_COMFYUI_DIR (ComfyUI may not be installed)"
mkdir -p "$LOCAL_BACKUP_DIR"

# ── Download ─────────────────────────────────────────────────────────
echo "$DOWNLOAD_FILES" | while IFS= read -r remote_path; do
    filename=$(basename "$remote_path")
    info "Downloading ${filename}..."

    # Download to backup directory
    scp $SSH_OPTS "${EC2_USER}@${EC2_HOST}:${remote_path}" "${LOCAL_BACKUP_DIR}/${filename}"
    info "  Saved to: ${LOCAL_BACKUP_DIR}/${filename}"

    # Copy to ComfyUI loras directory
    if [ -d "$LOCAL_COMFYUI_DIR" ]; then
        cp "${LOCAL_BACKUP_DIR}/${filename}" "${LOCAL_COMFYUI_DIR}/${filename}"
        info "  Copied to: ${LOCAL_COMFYUI_DIR}/${filename}"
    fi
done

echo ""
info "============================================"
info "  Download complete!"
info "============================================"
info "Backup:  ${LOCAL_BACKUP_DIR}/"
if [ -d "$LOCAL_COMFYUI_DIR" ]; then
    info "ComfyUI: ${LOCAL_COMFYUI_DIR}/"
fi
echo ""
info "Next: Open ComfyUI locally and load the LoRA for testing."
