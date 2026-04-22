#!/usr/bin/env python3

"""
Author: LO
Created: 1 March 2025

===
TEST THIS
Restructures CVAT YOLOv8 Segmentation format directories into standardized format:
1. Moves all label files from nested directories into a single all_labels directory
2. Updates data.yaml with correct paths and validation set
3. Creates test.yaml for testing configuration
===
"""

import os
import shutil
import argparse
import yaml
from pathlib import Path

def clean_directory(directory):
    """Remove directory and its contents if it exists"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def move_label_files(src_dir, dest_dir):
    """Move all label files from nested directories into a single directory"""
    print("\nMoving label files...")
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Walk through all subdirectories
    for root, _, files in os.walk(os.path.join(src_dir, "labels")):
        for file in files:
            if file.endswith('.txt'):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                
                # Handle filename collisions
                counter = 1
                base_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1]
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")
                    counter += 1
                
                try:
                    shutil.move(src_path, dest_path)
                    print(f"Moved: {file}")
                except Exception as e:
                    print(f"Warning: Failed to move {file}: {str(e)}")

def update_yaml_files(dataset_path):
    """Update data.yaml and create test.yaml with correct configurations"""
    print("\nUpdating YAML configurations...")
    
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    test_yaml_path = os.path.join(dataset_path, "test.yaml")
    
    # Read existing data.yaml if it exists
    if os.path.exists(data_yaml_path):
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        data = {}
    
    # Update data.yaml
    data["path"] = str(dataset_path)
    data["train"] = "train"
    data["val"] = "valid"  # Add validation set
    
    # Write updated data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Updated: {data_yaml_path}")
    
    # Create test.yaml
    test_data = data.copy()
    test_data["val"] = "test"  # Change validation to test
    
    with open(test_yaml_path, 'w') as f:
        yaml.dump(test_data, f, sort_keys=False)
    print(f"Created: {test_yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='Restructure CVAT YOLOv8 Segmentation format directories')
    parser.add_argument('--src', required=True, help='Source directory containing CVAT YOLOv8 Segmentation format dataset')
    args = parser.parse_args()
    
    # Validate source directory
    if not os.path.exists(args.src):
        print(f"Error: Source directory {args.src} does not exist")
        return
    
    # Create all_labels directory
    all_labels_dir = os.path.join(args.src, "all_labels")
    
    # Move label files
    move_label_files(args.src, all_labels_dir)
    
    # Update YAML files
    update_yaml_files(args.src)
    
    print(f"""
Dataset restructuring complete!
Output structure:
{args.src}/
    ├── all_labels/  # All label files in a single directory
    ├── data.yaml    # Updated training configuration
    └── test.yaml    # Testing configuration

You can now proceed with data splitting using:
python tools/bal_train_test_split.py --src {args.src} --valid 0.2 --test 0.1
""")

if __name__ == "__main__":
    main() 