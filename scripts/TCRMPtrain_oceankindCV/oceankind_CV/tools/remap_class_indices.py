#!/usr/bin/env python3

"""
Author: LO
Created: March 2025

Remaps class indices in YOLO label files from [1,2] to [0,1] and updates the data.yaml file.
Creates backups before making any changes.
"""

import os
import sys
import shutil
import argparse
import yaml
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Remap YOLO class indices and update YAML config')
    parser.add_argument('--data', required=True, help='Path to data.yaml file')
    parser.add_argument('--backup', action='store_true', help='Create backup of original files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    return parser.parse_args()

def backup_directory(src_dir, backup_dir):
    """Create a backup of the directory"""
    if os.path.exists(backup_dir):
        print(f"Backup directory already exists: {backup_dir}")
        return False
    try:
        shutil.copytree(src_dir, backup_dir)
        print(f"✓ Created backup: {backup_dir}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

def remap_file(file_path, class_map, dry_run=False):
    """Remap class indices in a single file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        changes = False
        
        for line in lines:
            if not line.strip():
                new_lines.append(line)
                continue
                
            parts = line.strip().split()
            if not parts:
                new_lines.append(line)
                continue
                
            try:
                old_class = int(parts[0])
                if old_class in class_map:
                    new_class = class_map[old_class]
                    new_line = f"{new_class} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
                    changes = True
                else:
                    new_lines.append(line)
            except ValueError:
                new_lines.append(line)
                
        if changes and not dry_run:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            
        return changes
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(dir_path, class_map, dry_run=False):
    """Process all label files in a directory"""
    changes = False
    total_files = 0
    changed_files = 0
    
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return False
        
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.txt'):
                total_files += 1
                file_path = os.path.join(root, file)
                if remap_file(file_path, class_map, dry_run):
                    changed_files += 1
                    changes = True
                
    print(f"✓ Processed {dir_path}: {changed_files}/{total_files} files modified")
    return changes

def update_yaml(yaml_path, class_map, dry_run=False):
    """Update the YAML file with new class indices"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        # Create new names mapping
        old_names = data.get('names', {})
        new_names = {}
        
        for old_idx, name in old_names.items():
            if old_idx in class_map:
                new_names[class_map[old_idx]] = name
        
        if dry_run:
            print("\nYAML changes that would be made:")
            print(f"Old names: {old_names}")
            print(f"New names: {new_names}")
            return True
            
        data['names'] = new_names
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
            
        print(f"✓ Updated class indices in {yaml_path}")
        return True
    except Exception as e:
        print(f"Error updating YAML: {e}")
        return False

def verify_changes(dir_path, allowed_classes):
    """Verify that all class indices are within the allowed set"""
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            class_idx = int(line.split()[0])
                            if class_idx not in allowed_classes:
                                print(f"Invalid class index {class_idx} in {file_path}:{line_num}")
                                return False
                        except (ValueError, IndexError):
                            continue
    return True

def main():
    args = parse_args()
    
    # Define class remapping
    class_map = {1: 0, 2: 1}
    allowed_classes = set(class_map.values())
    
    # Load YAML to get paths
    try:
        with open(args.data, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        sys.exit(1)
        
    base_path = config.get('path', '')
    if not base_path:
        print("Error: 'path' not found in YAML")
        sys.exit(1)
        
    # Create backup if requested
    if args.backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{base_path}_backup_{timestamp}"
        if not backup_directory(base_path, backup_path):
            print("Error creating backup, aborting")
            sys.exit(1)
    
    # Process each directory
    dirs_to_process = ['train/labels', 'valid/labels', 'test/labels']
    changes_made = False
    
    print("\nProcessing label files...")
    for dir_name in dirs_to_process:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            if process_directory(dir_path, class_map, args.dry_run):
                changes_made = True
    
    if not changes_made and not args.dry_run:
        print("\nNo changes needed in label files")
        sys.exit(0)
    
    # Verify changes if not in dry run
    if not args.dry_run:
        print("\nVerifying changes...")
        for dir_name in dirs_to_process:
            dir_path = os.path.join(base_path, dir_name)
            if os.path.exists(dir_path):
                if not verify_changes(dir_path, allowed_classes):
                    print("Verification failed! Please restore from backup")
                    sys.exit(1)
        print("✓ All changes verified")
    
    # Update YAML file
    if update_yaml(args.data, class_map, args.dry_run):
        if args.dry_run:
            print("\nDry run completed successfully")
        else:
            print("\nAll changes completed successfully")
    else:
        print("\nError updating YAML file")
        sys.exit(1)

if __name__ == '__main__':
    main() 