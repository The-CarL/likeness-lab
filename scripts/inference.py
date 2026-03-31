#!/usr/bin/env python3
"""
inference.py — Generate images using a trained LoRA.

Supports both Flux 1 [dev] and Flux 2 Klein 9B models.
Loads the model in 8-bit quantization for 24 GB GPUs.

Usage:
    # Single prompt
    python scripts/inference.py --prompt "photo of ohwx_person at the beach, sunset"

    # Batch from file (one prompt per line)
    python scripts/inference.py --prompts-file prompts.txt

    # With options
    python scripts/inference.py \
        --prompt "photo of ohwx_person, studio portrait" \
        --lora-path outputs/carlop_klein9b_v1/carlop_klein9b_v1.safetensors \
        --lora-scale 0.9 \
        --model black-forest-labs/FLUX.2-klein-base-9B \
        --steps 50 \
        --output-dir outputs/my_images \
        --seed 42
"""

import argparse
import os
import sys
from pathlib import Path


def find_latest_lora(outputs_dir: str = "outputs") -> str | None:
    """Find the most recent .safetensors file in the outputs directory."""
    outputs = Path(outputs_dir)
    if not outputs.exists():
        return None
    safetensors = sorted(outputs.rglob("*.safetensors"), key=os.path.getmtime)
    # Prefer the final checkpoint (no step number in name)
    finals = [s for s in safetensors if not any(c.isdigit() for c in s.stem.split("_")[-1])]
    if finals:
        return str(finals[-1])
    return str(safetensors[-1]) if safetensors else None


def main():
    parser = argparse.ArgumentParser(description="Generate images with a trained LoRA.")
    parser.add_argument("--prompt", "-p", type=str, help="Single prompt to generate")
    parser.add_argument("--prompts-file", "-f", type=str, help="File with one prompt per line")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-9B",
        help="Base model (default: Klein 9B)",
    )
    parser.add_argument(
        "--lora-path", "-l",
        type=str,
        default=None,
        help="Path to .safetensors LoRA file (default: auto-detect latest)",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA strength (default: 1.0)")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/inference", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps (50 for Klein Base, 20 for Flux 1)")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", "-n", type=int, default=1, help="Number of variations per prompt (different seeds)")
    args = parser.parse_args()

    if not args.prompt and not args.prompts_file:
        parser.error("Provide either --prompt or --prompts-file")

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.append(("prompt", args.prompt))
    if args.prompts_file:
        with open(args.prompts_file) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append((f"batch_{i:03d}", line))

    # Find LoRA
    lora_path = args.lora_path or find_latest_lora()
    if not lora_path:
        print("ERROR: No LoRA found. Specify --lora-path or ensure outputs/ has .safetensors files.")
        sys.exit(1)

    print(f"Model:      {args.model}")
    print(f"LoRA:       {lora_path}")
    print(f"LoRA scale: {args.lora_scale}")
    print(f"Steps:      {args.steps}")
    print(f"Prompts:    {len(prompts)}")
    print(f"Variations: {args.count} per prompt")
    print(f"Output:     {args.output_dir}")
    print()

    # Load model
    import torch
    from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig

    print("Loading transformer in 8-bit...", flush=True)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        quantization_config=quantization_config,
    )

    print("Loading pipeline...", flush=True)
    pipe = FluxPipeline.from_pretrained(
        args.model,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(lora_path)
    if args.lora_scale != 1.0:
        pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.enable_model_cpu_offload()
    print("Pipeline ready.", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    total = len(prompts) * args.count
    idx = 0
    for name, prompt in prompts:
        for v in range(args.count):
            idx += 1
            seed = args.seed + v
            out_name = f"{name}_s{seed}.jpg" if args.count > 1 else f"{name}.jpg"
            print(f"[{idx}/{total}] {out_name}: {prompt[:70]}...", flush=True)

            result = pipe(
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                width=args.width,
                height=args.height,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            result.save(os.path.join(args.output_dir, out_name))

    print(f"\nDone! {total} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
