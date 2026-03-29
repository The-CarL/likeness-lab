#!/usr/bin/env python3
"""
caption_dataset.py — Generate or validate captions for LoRA training datasets.

Modes:
  kontext  — Validate that every target image has an instruction caption
             containing the trigger word. Fix any missing or bad captions.
  standard — Auto-caption images using Florence-2, prepending the trigger word.

Usage:
    python scripts/caption_dataset.py --mode kontext
    python scripts/caption_dataset.py --mode standard
    python scripts/caption_dataset.py --mode standard --dataset datasets/standard
    python scripts/caption_dataset.py --mode kontext --dry-run
"""

import argparse
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Instruction templates for Kontext captions (used to fill in missing captions)
KONTEXT_TEMPLATES = [
    "Replace this person with {trigger}",
    "Make this person look like {trigger}",
    "Transform this into a photo of {trigger}",
    "Change the person in this image to {trigger}",
    "Replace the face with {trigger}",
    "Turn this person into {trigger}",
    "Make the person in this photo become {trigger}",
    "Swap this person for {trigger}",
]


def find_images(directory: Path) -> list[Path]:
    """Find all image files in a directory."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_kontext(dataset_dir: Path, trigger_word: str, dry_run: bool) -> None:
    """Validate and fix captions for a Kontext paired dataset."""
    target_dir = dataset_dir / "target"
    control_dir = dataset_dir / "control"

    if not target_dir.exists():
        print(f"ERROR: Target directory not found: {target_dir}")
        print("Run prepare_kontext_pairs.py first.")
        sys.exit(1)

    images = find_images(target_dir)
    if not images:
        print(f"ERROR: No images found in {target_dir}")
        sys.exit(1)

    print(f"Validating {len(images)} images in {target_dir}")
    print(f"Trigger word: {trigger_word}\n")

    issues = {"missing_caption": [], "missing_trigger": [], "missing_control": [], "ok": []}

    for i, img_path in enumerate(images):
        caption_path = img_path.with_suffix(".txt")
        stem = img_path.stem
        # Check for matching control image (any extension)
        control_exists = any(
            (control_dir / f"{stem}{ext}").exists()
            for ext in IMAGE_EXTENSIONS
        ) if control_dir.exists() else False

        if not control_exists:
            issues["missing_control"].append(img_path.name)

        if not caption_path.exists():
            issues["missing_caption"].append(img_path)
        else:
            caption = caption_path.read_text().strip()
            if trigger_word not in caption:
                issues["missing_trigger"].append((img_path, caption_path, caption))
            else:
                issues["ok"].append(img_path.name)

    # Report
    print(f"  OK:               {len(issues['ok'])}")
    print(f"  Missing caption:  {len(issues['missing_caption'])}")
    print(f"  Missing trigger:  {len(issues['missing_trigger'])}")
    print(f"  Missing control:  {len(issues['missing_control'])}")
    print()

    if issues["missing_control"]:
        print("WARNING: These target images have no matching control image:")
        for name in issues["missing_control"]:
            print(f"  - {name}")
        print()

    # Fix missing captions
    if issues["missing_caption"]:
        print("Fixing missing captions...")
        for i, img_path in enumerate(issues["missing_caption"]):
            caption_path = img_path.with_suffix(".txt")
            template = KONTEXT_TEMPLATES[i % len(KONTEXT_TEMPLATES)]
            caption = template.format(trigger=trigger_word)
            if dry_run:
                print(f"  [DRY RUN] Would write: {caption_path.name} → '{caption}'")
            else:
                caption_path.write_text(caption)
                print(f"  Created: {caption_path.name} → '{caption}'")

    # Fix captions missing the trigger word
    if issues["missing_trigger"]:
        print("Fixing captions missing trigger word...")
        for img_path, caption_path, old_caption in issues["missing_trigger"]:
            new_caption = f"{old_caption}. This person is {trigger_word}"
            if dry_run:
                print(f"  [DRY RUN] Would update: {caption_path.name}")
                print(f"    Old: '{old_caption}'")
                print(f"    New: '{new_caption}'")
            else:
                caption_path.write_text(new_caption)
                print(f"  Updated: {caption_path.name}")

    if not issues["missing_caption"] and not issues["missing_trigger"] and not issues["missing_control"]:
        print("All captions are valid!")


def caption_standard(dataset_dir: Path, trigger_word: str, dry_run: bool) -> None:
    """Auto-caption images for standard (non-Kontext) LoRA training using Florence-2."""
    images = find_images(dataset_dir)
    if not images:
        print(f"ERROR: No images found in {dataset_dir}")
        sys.exit(1)

    # Check which images already have captions
    needs_caption = []
    already_captioned = 0
    for img_path in images:
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            already_captioned += 1
        else:
            needs_caption.append(img_path)

    print(f"Found {len(images)} images in {dataset_dir}")
    print(f"  Already captioned: {already_captioned}")
    print(f"  Need captioning:   {len(needs_caption)}")
    print()

    if not needs_caption:
        print("All images already have captions!")
        _validate_trigger_in_standard(dataset_dir, trigger_word, dry_run)
        return

    if dry_run:
        print("DRY RUN — would caption these images:")
        for img_path in needs_caption:
            print(f"  {img_path.name} → {img_path.stem}.txt")
        return

    # Load Florence-2 for captioning
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError:
        print("ERROR: Auto-captioning requires torch, transformers, and Pillow.")
        print("Install with: pip install torch transformers Pillow")
        sys.exit(1)

    print("Loading Florence-2 captioning model...")
    model_id = "microsoft/Florence-2-base"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    device = next(model.parameters()).device
    print(f"Model loaded on {device}\n")

    for i, img_path in enumerate(needs_caption, 1):
        print(f"[{i}/{len(needs_caption)}] Captioning {img_path.name}...")

        image = Image.open(img_path).convert("RGB")
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
            )

        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Clean up the caption
        caption = caption.replace("<MORE_DETAILED_CAPTION>", "").strip()
        # Prepend trigger word
        full_caption = f"{trigger_word}, {caption}"

        caption_path = img_path.with_suffix(".txt")
        caption_path.write_text(full_caption)
        print(f"  → '{full_caption}'")

    print(f"\nDone! Captioned {len(needs_caption)} images.")

    # Clean up
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_trigger_in_standard(dataset_dir: Path, trigger_word: str, dry_run: bool) -> None:
    """Ensure all standard captions start with the trigger word."""
    images = find_images(dataset_dir)
    fixed = 0
    for img_path in images:
        caption_path = img_path.with_suffix(".txt")
        if not caption_path.exists():
            continue
        caption = caption_path.read_text().strip()
        if not caption.startswith(trigger_word):
            new_caption = f"{trigger_word}, {caption}"
            if dry_run:
                print(f"  [DRY RUN] Would prepend trigger to {caption_path.name}")
            else:
                caption_path.write_text(new_caption)
                fixed += 1
    if fixed:
        print(f"Prepended trigger word to {fixed} captions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate or validate captions for LoRA training datasets."
    )
    parser.add_argument(
        "--mode",
        choices=["kontext", "standard"],
        required=True,
        help="kontext: validate paired dataset captions. standard: auto-caption with Florence-2.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset directory (default: datasets/kontext_pairs for kontext, datasets/standard for standard)",
    )
    parser.add_argument(
        "--trigger-word",
        default="CARLOP",
        help="Trigger word to include in captions (default: CARLOP)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying any files",
    )
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = (
            Path("datasets/kontext_pairs") if args.mode == "kontext"
            else Path("datasets/standard")
        )

    if args.mode == "kontext":
        validate_kontext(args.dataset, args.trigger_word, args.dry_run)
    else:
        caption_standard(args.dataset, args.trigger_word, args.dry_run)
