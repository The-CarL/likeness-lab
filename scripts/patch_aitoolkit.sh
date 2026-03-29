#!/usr/bin/env bash
# ============================================================================
# patch_aitoolkit.sh — Apply patches to ai-toolkit for 24 GB GPU compatibility
# ============================================================================
# The stock ai-toolkit OOMs on 24 GB GPUs (A10G, RTX 3090/4090) when loading
# the Flux Kontext Dev model, because it loads the full BF16 transformer to
# GPU before quantizing (~24 GB in BF16 > 24 GB VRAM). These patches:
#
#   1. Add a low_vram code path that loads to CPU, quantizes with quanto,
#      then moves the quantized (~12 GB) model to GPU.
#   2. Create a 32 GB swap file so the CPU can hold the model during
#      quantization (the g5.xlarge only has 16 GB RAM).
#   3. Pin numpy<2 and mediapipe==0.10.14 for compatibility with ai-toolkit's
#      controlnet_aux dependency (requires mp.solutions API).
#
# This script is idempotent — safe to run multiple times.
#
# Usage:  bash scripts/patch_aitoolkit.sh
# ============================================================================
set -euo pipefail

TOOLKIT_DIR="${HOME}/ai-toolkit"
KONTEXT_FILE="$TOOLKIT_DIR/extensions_built_in/diffusion_models/flux_kontext/flux_kontext.py"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Patch flux_kontext.py for CPU-based quantization ──────────────
if [ ! -f "$KONTEXT_FILE" ]; then
    error "Cannot find $KONTEXT_FILE. Make sure ai-toolkit is cloned at $TOOLKIT_DIR"
fi

if grep -q "Loading transformer to CPU (low_vram mode)" "$KONTEXT_FILE"; then
    info "flux_kontext.py already patched, skipping."
else
    info "Patching flux_kontext.py for low_vram CPU quantization..."
    cp "$KONTEXT_FILE" "${KONTEXT_FILE}.bak"

    python3 -c "
with open('$KONTEXT_FILE', 'r') as f:
    content = f.read()

old = '''        self.print_and_status_update(\"Loading transformer\")
        transformer = FluxTransformer2DModel.from_pretrained(
            transformer_path,
            subfolder=transformer_subfolder,
            torch_dtype=dtype
        )
        transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update(\"Quantizing transformer\")
            quantize(transformer, weights=quantization_type,
                     **self.model_config.quantize_kwargs)
            freeze(transformer)
            transformer.to(self.device_torch)
        else:
            transformer.to(self.device_torch, dtype=dtype)'''

new = '''        self.print_and_status_update(\"Loading transformer\")
        if self.model_config.quantize and self.model_config.low_vram:
            # Low VRAM: load to CPU, quantize on CPU, then move to GPU.
            # Requires sufficient system RAM or swap (~24 GB).
            self.print_and_status_update(\"Loading transformer to CPU (low_vram mode)\")
            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                subfolder=transformer_subfolder,
                torch_dtype=dtype,
                device_map=\"cpu\",
            )
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update(\"Quantizing transformer on CPU\")
            quantize(transformer, weights=quantization_type,
                     **self.model_config.quantize_kwargs)
            freeze(transformer)
            self.print_and_status_update(\"Moving quantized transformer to GPU\")
            transformer.to(self.device_torch)
        elif self.model_config.quantize:
            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                subfolder=transformer_subfolder,
                torch_dtype=dtype
            )
            transformer.to(self.quantize_device, dtype=dtype)
            patch_dequantization_on_save(transformer)
            quantization_type = get_qtype(self.model_config.qtype)
            self.print_and_status_update(\"Quantizing transformer\")
            quantize(transformer, weights=quantization_type,
                     **self.model_config.quantize_kwargs)
            freeze(transformer)
            transformer.to(self.device_torch)
        else:
            transformer = FluxTransformer2DModel.from_pretrained(
                transformer_path,
                subfolder=transformer_subfolder,
                torch_dtype=dtype
            )
            transformer.to(self.device_torch, dtype=dtype)'''

if old in content:
    content = content.replace(old, new)
    with open('$KONTEXT_FILE', 'w') as f:
        f.write(content)
    print('OK')
else:
    print('FAIL')
    exit(1)
" || error "Could not patch flux_kontext.py — ai-toolkit version may have changed."
    info "flux_kontext.py patched."
fi

# ── 2. Create swap file (needed for CPU quantization on <=16 GB RAM) ─
if swapon --show | grep -q '/swapfile'; then
    info "Swap already enabled, skipping."
else
    TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM_GB" -lt 24 ]; then
        info "System has ${TOTAL_RAM_GB} GB RAM — creating 32 GB swap file..."
        sudo fallocate -l 32G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        info "Swap enabled (32 GB)."
    else
        info "System has ${TOTAL_RAM_GB} GB RAM — swap not needed."
    fi
fi

# ── 3. Pin dependency versions ───────────────────────────────────────
info "Pinning dependency versions..."

# Activate the PyTorch environment
if [ -f "/opt/pytorch/bin/activate" ]; then
    source /opt/pytorch/bin/activate
fi

# mediapipe 0.10.14 is the last version with mp.solutions (needed by controlnet_aux)
# numpy<2 is needed for scipy/diffusers compatibility on the DLAMI
pip install 'numpy<2' 'mediapipe==0.10.14' --quiet 2>&1 | tail -1
info "Dependencies pinned (numpy<2, mediapipe==0.10.14)."

echo ""
info "All patches applied. Ready for 24 GB GPU training with low_vram: true."
