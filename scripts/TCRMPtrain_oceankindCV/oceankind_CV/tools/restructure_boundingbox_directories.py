#!/usr/bin/env python3

"""
Author: LO
Created: 1 March 2025

Restructures CVAT bounding box annotations into standardized YOLO format:
1. Moves all label files from nested directories into a single all_labels directory
2. Updates data.yaml with correct paths and validation set
3. Creates test.yaml for testing configuration
"""

import os
import shutil
import argparse
import re
from pathlib import Path
from coco_to_yolo_format import COCO2YOLOBB
from PIL import Image

def clean_directory(directory):
    """Remove directory and its contents if it exists"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def create_directory_structure(base_path):
    """Create the required directory structure"""
    # Clean and create directories
    clean_directory(base_path)
    all_images_dir = os.path.join(base_path, "all_images")
    all_labels_dir = os.path.join(base_path, "all_labels")
    os.makedirs(all_images_dir)
    os.makedirs(all_labels_dir)
    return all_images_dir, all_labels_dir

def sanitize_filename(filename):
    """Convert filename to a consistent format for both images and labels"""
    # Get the file extension
    path = Path(filename)
    stem = path.stem  # Gets filename without extension
    ext = path.suffix.lower()  # Gets extension with dot, convert to lowercase
    
    # First, decode any URL-encoded characters
    stem = stem.replace('%20', ' ').replace('%C2%A9', '©')
    
    # Replace copyright symbol with text
    stem = stem.replace('©', 'copyright')
    
    # Replace specific characters with underscores
    stem = re.sub(r'[-\s]', '_', stem)  # Replace hyphens and spaces with underscores
    
    # Remove any other non-alphanumeric characters except underscores
    stem = re.sub(r'[^a-zA-Z0-9_]', '', stem)
    
    # Remove consecutive underscores
    stem = re.sub(r'_+', '_', stem)
    
    # Remove leading/trailing underscores
    stem = stem.strip('_')
    
    # Ensure the filename is not empty
    if not stem:
        stem = 'image'
    
    return f"{stem}{ext}"

def copy_images(src_image_dir, dest_image_dir):
    """Copy all images from source to destination with sanitized names"""
    image_mapping = {}  # Keep track of original to new names
    
    # Expanded list of supported image formats
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.webp', '.jfif')
    
    # Walk through the source directory
    for root, _, files in os.walk(src_image_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                try:
                    src_path = os.path.join(root, file)
                    
                    # Handle special characters in original filename
                    decoded_file = file.encode('utf-8').decode('utf-8', errors='replace')
                    new_name = sanitize_filename(decoded_file)
                    
                    # Convert all images to .jpg format for consistency
                    base = os.path.splitext(new_name)[0]
                    new_name = f"{base}.jpg"
                    
                    # Handle potential filename collisions
                    counter = 1
                    while os.path.exists(os.path.join(dest_image_dir, new_name)):
                        new_name = f"{base}_{counter}.jpg"
                        counter += 1
                    
                    dest_path = os.path.join(dest_image_dir, new_name)
                    
                    # Store the mapping using the original filename from COCO JSON
                    image_mapping[file] = new_name
                    
                    # Copy and convert the file using PIL to ensure format conversion
                    try:
                        with Image.open(src_path) as img:
                            # Convert RGBA to RGB if necessary
                            if img.mode in ('RGBA', 'LA'):
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                background.paste(img, mask=img.split()[-1])
                                img = background
                            elif img.mode not in ('RGB', 'L'):
                                img = img.convert('RGB')
                            
                            # Save as JPEG
                            img.save(dest_path, 'JPEG', quality=95)
                            print(f"Converted and copied {file} to {new_name}")
                    except Exception as e:
                        print(f"Warning: Failed to process image {file}: {str(e)}")
                        continue
                    
                except Exception as e:
                    print(f"Warning: Failed to process {file}: {str(e)}")
                    continue
    
    return image_mapping

def validate_dataset(images_dir, labels_dir):
    """Cross-reference images and labels, return mismatches"""
    image_files = {Path(f).stem for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.jfif'))}
    label_files = {Path(f).stem for f in os.listdir(labels_dir) if f.endswith('.txt')}
    
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    
    return images_without_labels, labels_without_images

def handle_mismatches(images_dir, labels_dir, images_without_labels, labels_without_images):
    """Handle mismatched files with user interaction"""
    if not images_without_labels and not labels_without_images:
        print("\nValidation successful! All images have corresponding labels and vice versa.")
        return

    print("\nValidation Results:")
    
    if images_without_labels:
        print(f"\nFound {len(images_without_labels)} images without labels:")
        for img in sorted(images_without_labels):
            print(f"  - {img}")
        
        response = input("\nWould you like to remove these images without labels? (y/n): ").lower()
        if response == 'y':
            for img in images_without_labels:
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.jfif']:
                    img_path = os.path.join(images_dir, f"{img}{ext}")
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        print(f"Removed: {img_path}")
                        break
    
    if labels_without_images:
        print(f"\nFound {len(labels_without_images)} labels without images:")
        for lbl in sorted(labels_without_images):
            print(f"  - {lbl}")
        
        response = input("\nWould you like to remove these labels without images? (y/n): ").lower()
        if response == 'y':
            for lbl in labels_without_images:
                label_path = os.path.join(labels_dir, f"{lbl}.txt")
                if os.path.exists(label_path):
                    os.remove(label_path)
                    print(f"Removed: {label_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO training')
    parser.add_argument('--src', required=True, help='Source directory containing COCO format dataset')
    parser.add_argument('--dest', required=True, help='Destination directory for YOLO format dataset')
    parser.add_argument('--skip-validation', action='store_true', help='Skip dataset validation')
    args = parser.parse_args()

    # Create destination directory structure
    print("Creating directory structure...")
    images_dir, labels_dir = create_directory_structure(args.dest)

    # Copy images
    src_images = os.path.join(args.src, "images", "default")
    if not os.path.exists(src_images):
        print(f"Error: Source images directory not found at {src_images}")
        return
    
    print("Copying images...")
    image_mapping = copy_images(src_images, images_dir)

    # Convert annotations
    print("Converting annotations...")
    coco_json = os.path.join(args.src, "annotations", "instances_default.json")
    if not os.path.exists(coco_json):
        print(f"Error: COCO annotations not found at {coco_json}")
        return

    # Create a modified version of the COCO JSON with sanitized filenames
    import json
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Update image filenames in the COCO JSON
    for img in coco_data['images']:
        original_name = img['file_name']
        if original_name in image_mapping:
            img['file_name'] = image_mapping[original_name]
        else:
            print(f"Warning: Image {original_name} not found in mapping")
    
    # Write temporary JSON with updated filenames
    temp_json = os.path.join(args.dest, 'temp_annotations.json')
    with open(temp_json, 'w') as f:
        json.dump(coco_data, f)

    # Convert using the temporary JSON with sanitized names
    converter = COCO2YOLOBB(temp_json, args.dest)
    converter.run()

    # Clean up temporary file
    os.remove(temp_json)

    # Validate dataset unless explicitly skipped
    if not args.skip_validation:
        print("\nValidating dataset...")
        images_without_labels, labels_without_images = validate_dataset(images_dir, labels_dir)
        handle_mismatches(images_dir, labels_dir, images_without_labels, labels_without_images)

    print(f"""
Dataset preparation complete!
Output structure:
{args.dest}/
    ├── all_labels/  # YOLO format annotation files
    ├── all_images/  # Copied image files
    ├── data.yaml    # YOLO training configuration
    └── test.yaml    # YOLO testing configuration

You can now proceed with class merging if needed using:
python tools/merge_classes.py --save {args.dest} --data {os.path.join(args.dest, 'data.yaml')} --newclasses <comma-separated-classes>
""")

if __name__ == "__main__":
    main() 