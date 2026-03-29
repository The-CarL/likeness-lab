#!/usr/bin/env python3
"""
prepare_kontext_pairs.py — Generate Kontext-compatible paired datasets.

Takes a folder of personal photos and creates:
  - target/ folder: your original photos (the "after" / desired output)
  - control/ folder: versions with your face replaced/blurred (the "before" / input)
  - .txt caption files with transformation instructions

For best results, use a real inpainting pipeline to replace faces with generic
people. This script provides two approaches:
  1. Face blur (default) — fast, no extra models needed, decent results
  2. Diffusion inpainting — higher quality, requires extra VRAM

Usage:
    python scripts/prepare_kontext_pairs.py
    python scripts/prepare_kontext_pairs.py --input datasets/raw_photos --output datasets/kontext_pairs
    python scripts/prepare_kontext_pairs.py --dry-run
    python scripts/prepare_kontext_pairs.py --method inpaint  # uses diffusion inpainting
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Extensions the training pipeline can read
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Extensions that need conversion before use
HEIC_EXTENSIONS = {".heic", ".heif"}

# Caption templates — the trainer will insert the trigger word automatically,
# but we include it explicitly for clarity. These describe the transformation
# from control (generic/blurred) to target (your real photo).
CAPTION_TEMPLATES = [
    "Replace this person with CARLOP",
    "Make this person look like CARLOP",
    "Transform this into a photo of CARLOP",
    "Change the person in this image to CARLOP",
    "Replace the face with CARLOP",
]


def convert_heic_images(directory: Path) -> int:
    """
    Convert any HEIC/HEIF files in the directory to JPG in-place.

    Uses macOS `sips` (built-in, no install needed) with a fallback to
    `pillow-heif` for Linux/EC2.

    Returns the number of files converted.
    """
    heic_files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in HEIC_EXTENSIONS
    )
    if not heic_files:
        return 0

    print(f"Found {len(heic_files)} HEIC/HEIF files — converting to JPG...")

    converted = 0
    for heic_path in heic_files:
        jpg_path = heic_path.with_suffix(".jpg")
        if jpg_path.exists():
            print(f"  SKIP {heic_path.name} — {jpg_path.name} already exists")
            continue

        # Try macOS sips first (built-in, fast)
        if shutil.which("sips"):
            try:
                subprocess.run(
                    ["sips", "-s", "format", "jpeg", str(heic_path), "--out", str(jpg_path)],
                    check=True, capture_output=True,
                )
                print(f"  Converted {heic_path.name} → {jpg_path.name} (sips)")
                converted += 1
                continue
            except subprocess.CalledProcessError:
                pass  # Fall through to pillow-heif

        # Fallback: pillow-heif (works on Linux/EC2)
        try:
            from PIL import Image
            import pillow_heif
            pillow_heif.register_heif_opener()
            img = Image.open(heic_path)
            img.save(jpg_path, "JPEG", quality=95)
            print(f"  Converted {heic_path.name} → {jpg_path.name} (pillow-heif)")
            converted += 1
        except ImportError:
            print(f"  ERROR: Cannot convert {heic_path.name}.")
            print("    On macOS this should work automatically via sips.")
            print("    On Linux, install: pip install pillow-heif")
            sys.exit(1)
        except Exception as e:
            print(f"  ERROR converting {heic_path.name}: {e}")

    print(f"Converted {converted}/{len(heic_files)} HEIC files.\n")
    return converted


def find_images(directory: Path) -> list[Path]:
    """Find all image files in a directory (non-recursive)."""
    images = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


def blur_face(image_path: Path, output_path: Path) -> bool:
    """
    Detect and blur the face in an image to create a "before" control image.

    Uses MediaPipe for face detection and applies a heavy Gaussian blur
    to the face region. This is a simple but effective approach.

    Returns True if a face was found and blurred, False otherwise.
    """
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError:
        print("ERROR: Face blur requires opencv-python and mediapipe.")
        print("Install with: pip install opencv-python mediapipe")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  WARNING: Could not read {image_path.name}, skipping.")
        return False

    h, w = img.shape[:2]

    # Detect face using MediaPipe
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if not results.detections:
            print(f"  WARNING: No face detected in {image_path.name}.")
            # Still create the control image — just copy the original with slight blur
            blurred = cv2.GaussianBlur(img, (25, 25), 15)
            cv2.imwrite(str(output_path), blurred)
            return True

        # Use the first (most confident) detection
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Expand bounding box by 50% to cover the full head area
        cx = bbox.xmin + bbox.width / 2
        cy = bbox.ymin + bbox.height / 2
        bw = bbox.width * 1.5
        bh = bbox.height * 1.5

        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))

        # Apply heavy Gaussian blur to the face region
        face_region = img[y1:y2, x1:x2]
        kernel_size = max(99, (min(x2 - x1, y2 - y1) // 2) * 2 + 1)
        blurred_face = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 50)

        # Create an elliptical mask for smooth blending
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        center = ((x2 - x1) // 2, (y2 - y1) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        mask = mask[:, :, np.newaxis]

        # Blend blurred face with original
        blended = (blurred_face * mask + face_region * (1 - mask)).astype(np.uint8)
        result = img.copy()
        result[y1:y2, x1:x2] = blended

        cv2.imwrite(str(output_path), result)
        return True


def inpaint_face(image_path: Path, output_path: Path) -> bool:
    """
    Replace the face using a diffusion inpainting pipeline.

    This produces higher-quality "before" images by replacing your face
    with a generic person, but requires more VRAM and time.

    Returns True on success, False otherwise.
    """
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        from PIL import Image
    except ImportError:
        print("ERROR: Inpainting requires opencv-python, mediapipe, diffusers, and torch.")
        print("Install with: pip install opencv-python mediapipe diffusers torch accelerate")
        sys.exit(1)

    img_cv = cv2.imread(str(image_path))
    if img_cv is None:
        print(f"  WARNING: Could not read {image_path.name}, skipping.")
        return False

    h, w = img_cv.shape[:2]

    # Detect face
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if not results.detections:
            print(f"  WARNING: No face detected in {image_path.name}, falling back to blur.")
            return blur_face(image_path, output_path)

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Create mask for the face region (expanded)
        cx = bbox.xmin + bbox.width / 2
        cy = bbox.ymin + bbox.height / 2
        bw = bbox.width * 1.6
        bh = bbox.height * 1.6

        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))

        mask = np.zeros((h, w), dtype=np.uint8)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 10)

    # Run inpainting
    pil_image = Image.open(image_path).convert("RGB")
    pil_mask = Image.fromarray(mask).convert("L")

    # Resize to 512x512 for inpainting, then resize back
    orig_size = pil_image.size
    pil_image_resized = pil_image.resize((512, 512))
    pil_mask_resized = pil_mask.resize((512, 512))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    ).to("cuda")

    result = pipe(
        prompt="a person, neutral expression, natural lighting",
        image=pil_image_resized,
        mask_image=pil_mask_resized,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    # Resize back and save
    result = result.resize(orig_size)
    result.save(str(output_path))

    # Clean up GPU memory
    del pipe
    torch.cuda.empty_cache()

    return True


def prepare_pairs(
    input_dir: Path,
    output_dir: Path,
    method: str = "blur",
    trigger_word: str = "CARLOP",
    dry_run: bool = False,
) -> None:
    """Main function to prepare Kontext training pairs."""

    target_dir = output_dir / "target"
    control_dir = output_dir / "control"

    # Convert HEIC/HEIF files to JPG before processing
    convert_heic_images(input_dir)

    images = find_images(input_dir)
    if not images:
        print(f"ERROR: No images found in {input_dir}")
        print(f"  Place your photos in {input_dir} and run again.")
        sys.exit(1)

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")
    print()

    if dry_run:
        heic_count = sum(
            1 for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in HEIC_EXTENSIONS
        )
        if heic_count:
            print(f"Note: {heic_count} HEIC/HEIF files will be converted to JPG first.\n")
        print("DRY RUN — showing what would be created:\n")
        for i, img in enumerate(images, 1):
            idx = f"{i:03d}"
            ext = img.suffix.lower()
            print(f"  {idx}{ext}  →  target/{idx}{ext} + control/{idx}{ext} + target/{idx}.txt")
        print(f"\nTotal: {len(images)} pairs would be created.")
        return

    # Create output directories
    target_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    process_fn = inpaint_face if method == "inpaint" else blur_face

    success_count = 0
    for i, img_path in enumerate(images, 1):
        idx = f"{i:03d}"
        ext = img_path.suffix.lower()

        target_path = target_dir / f"{idx}{ext}"
        control_path = control_dir / f"{idx}{ext}"
        caption_path = target_dir / f"{idx}.txt"

        print(f"[{i}/{len(images)}] Processing {img_path.name}...")

        # Copy original as the target (the "after" image — your real photo)
        shutil.copy2(img_path, target_path)

        # Generate control image (the "before" image — face replaced/blurred)
        if process_fn(img_path, control_path):
            # Write caption — cycle through templates for variety
            template = CAPTION_TEMPLATES[(i - 1) % len(CAPTION_TEMPLATES)]
            caption_path.write_text(template)
            success_count += 1
            print(f"  ✓ Created pair {idx}")
        else:
            # Clean up failed pair
            target_path.unlink(missing_ok=True)
            print(f"  ✗ Failed to create pair for {img_path.name}")

    print(f"\nDone! Created {success_count}/{len(images)} pairs.")
    print(f"  Target images: {target_dir}")
    print(f"  Control images: {control_dir}")
    print(f"\nNext step: review the control images, then run caption_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Kontext-compatible paired datasets from personal photos."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("datasets/raw_photos"),
        help="Directory containing your personal photos (default: datasets/raw_photos)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("datasets/kontext_pairs"),
        help="Output directory for paired dataset (default: datasets/kontext_pairs)",
    )
    parser.add_argument(
        "--method", "-m",
        choices=["blur", "inpaint"],
        default="blur",
        help="Method to generate 'before' images: blur (fast, default) or inpaint (higher quality, needs GPU)",
    )
    parser.add_argument(
        "--trigger-word",
        default="CARLOP",
        help="Trigger word for captions (default: CARLOP)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating any files",
    )
    args = parser.parse_args()
    prepare_pairs(args.input, args.output, args.method, args.trigger_word, args.dry_run)
