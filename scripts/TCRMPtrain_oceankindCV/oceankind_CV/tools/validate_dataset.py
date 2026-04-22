#!/usr/bin/env python3

"""
Author: LO
Created: 1 March 2025

===
Validates that all images have corresponding labels and vice versa.
Optionally removes unmatched files.
===
"""

import os
import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='Validate YOLO dataset image-label pairs')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory containing all_images and all_labels')
    args = parser.parse_args()

    # Check directory structure
    images_dir = os.path.join(args.dataset, "all_images")
    labels_dir = os.path.join(args.dataset, "all_labels")

    if not os.path.exists(images_dir):
        print(f"Error: all_images directory not found at {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Error: all_labels directory not found at {labels_dir}")
        return

    print("\nValidating dataset...")
    images_without_labels, labels_without_images = validate_dataset(images_dir, labels_dir)
    handle_mismatches(images_dir, labels_dir, images_without_labels, labels_without_images)

if __name__ == "__main__":
    main() 