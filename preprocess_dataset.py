"""
Preprocess strabismus dataset:
1. Resize all images to 256x256 with black padding (maintaining aspect ratio)
2. Reorganize into two folders: STRABISMUS and NORMAL
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

def resize_with_padding(img, target_size=(256, 256)):
    """
    Resize image to target size with black padding to maintain aspect ratio.
    """
    # Get original size
    original_width, original_height = img.size
    target_width, target_height = target_size

    # Calculate scaling factor (maintain aspect ratio)
    scale = min(target_width / original_width, target_height / original_height)

    # Calculate new size
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with black background
    new_img = Image.new('RGB', target_size, (0, 0, 0))

    # Calculate position to paste (center the image)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste resized image onto black background
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


def preprocess_dataset(
    source_dir='STRABISMUS',
    output_dir='data',
    target_size=(256, 256)
):
    """
    Preprocess the entire dataset.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Define categories
    strabismus_types = ['ESOTROPIA', 'EXOTROPIA', 'HYPERTROPIA', 'HYPOTROPIA']
    normal_type = 'NORMAL'

    # Create output directories
    strabismus_out = output_dir / 'STRABISMUS'
    normal_out = output_dir / 'NORMAL'
    strabismus_out.mkdir(parents=True, exist_ok=True)
    normal_out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dataset Preprocessing")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size: {target_size}")
    print()

    # Process strabismus images
    print("Processing STRABISMUS images...")
    strab_count = 0

    for strab_type in strabismus_types:
        strab_dir = source_dir / strab_type
        if not strab_dir.exists():
            print(f"  Warning: {strab_type} directory not found, skipping...")
            continue

        image_files = list(strab_dir.glob('*.jpg')) + list(strab_dir.glob('*.png'))
        print(f"  Processing {strab_type}: {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"  {strab_type}"):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')

                # Resize with padding
                img_processed = resize_with_padding(img, target_size)

                # Save with new naming: strabismus_TYPE_ORIGINAL_NAME.jpg
                new_name = f"strabismus_{strab_type.lower()}_{img_path.stem}.jpg"
                output_path = strabismus_out / new_name
                img_processed.save(output_path, 'JPEG', quality=95)

                strab_count += 1
            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")

    print(f"  Total STRABISMUS images processed: {strab_count}")
    print()

    # Process normal images
    print("Processing NORMAL images...")
    normal_count = 0

    normal_dir = source_dir / normal_type
    if normal_dir.exists():
        image_files = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))
        print(f"  Found {len(image_files)} images")

        for img_path in tqdm(image_files, desc="  NORMAL"):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')

                # Resize with padding
                img_processed = resize_with_padding(img, target_size)

                # Save with new naming: normal_ORIGINAL_NAME.jpg
                new_name = f"normal_{img_path.stem}.jpg"
                output_path = normal_out / new_name
                img_processed.save(output_path, 'JPEG', quality=95)

                normal_count += 1
            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")
    else:
        print(f"  Warning: NORMAL directory not found!")

    print(f"  Total NORMAL images processed: {normal_count}")
    print()

    # Summary
    print("=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  STRABISMUS: {strab_count} images")
    print(f"  NORMAL: {normal_count} images")
    print(f"  Total: {strab_count + normal_count} images")
    print()
    print("New dataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── STRABISMUS/  ({strab_count} images)")
    print(f"    └── NORMAL/      ({normal_count} images)")
    print()
    print("All images are now 256x256 with black padding!")
    print("=" * 60)

    return strab_count, normal_count


def verify_preprocessing(output_dir='data'):
    """
    Verify that all images are 256x256.
    """
    output_dir = Path(output_dir)

    print("\nVerifying preprocessed images...")

    issues = []
    for folder in ['STRABISMUS', 'NORMAL']:
        folder_path = output_dir / folder
        if folder_path.exists():
            for img_path in folder_path.glob('*.jpg'):
                try:
                    img = Image.open(img_path)
                    if img.size != (256, 256):
                        issues.append(f"{img_path}: {img.size}")
                except Exception as e:
                    issues.append(f"{img_path}: Error - {e}")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
    else:
        print("✓ All images are 256x256!")

    return len(issues) == 0


if __name__ == "__main__":
    # Run preprocessing
    strab_count, normal_count = preprocess_dataset(
        source_dir='STRABISMUS',
        output_dir='data',
        target_size=(256, 256)
    )

    # Verify
    verify_preprocessing('data')

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Check the output folder: data/")
    print("2. Verify a few images to ensure quality")
    print("3. Update your notebook to use 'data' as DATA_DIR")
    print("=" * 60)
