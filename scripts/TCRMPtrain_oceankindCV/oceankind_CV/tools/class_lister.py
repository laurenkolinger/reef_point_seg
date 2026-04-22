#!/usr/bin/env python3

"""
Author: Lauren Olinger
Created: March 2025

===
Generates a clean class listing format from COCO instances.json for class merging
Can optionally pre-fill mappings from an existing Families_class_changes file
Classes can be sorted by ID (default) or alphabetically by name
===
"""

import json
import argparse
import os
import sys
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

def find_instances_json(directory: str) -> str:
    """Find instances_default.json or data.yaml in the directory"""
    # First try data.yaml
    yaml_path = os.path.join(directory, "data.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
        
    # Fallback to instances_default.json
    json_path = os.path.join(directory, "annotations", "instances_default.json")
    if os.path.exists(json_path):
        return json_path
        
    sys.exit(f"Error: Could not find data.yaml or instances_default.json in {directory}")

def parse_class_changes(file_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse class mappings from a Families_class_changes file
    Returns a dictionary with both ID mappings and name mappings
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Find the class_change dictionary using regex
        dict_match = re.search(r'class_change\s*=\s*{([^}]+)}', content, re.DOTALL)
        if not dict_match:
            sys.exit("Could not find class_change dictionary in the file")
            
        dict_str = dict_match.group(1)
        
        # Parse the mappings and names
        id_mappings = {}  # old_id -> new_id
        name_mappings = {}  # class_name -> new_id
        
        for line in dict_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Extract the key-value pair and class name from comment
            # Match patterns like: '0': '9', # AENA_Normal -> Spotted Eagle Ray
            match = re.match(r"'(\d+)'\s*:\s*'(\d+|9)',\s*#\s*([A-Za-z0-9_]+)\s*->", line)
            if match:
                old_id, new_id, class_name = match.groups()
                id_mappings[old_id] = new_id
                name_mappings[class_name] = new_id
                
        if not id_mappings:
            sys.exit("No valid class mappings found in the file")
                
        return {'id_mappings': id_mappings, 'name_mappings': name_mappings}
        
    except Exception as e:
        sys.exit(f"Error parsing Families_class_changes file: {e}")

def load_instances_json(json_path: str) -> Dict:
    """Load and parse the COCO instances.json or YOLO data.yaml file"""
    try:
        if json_path.endswith('.yaml') or json_path.endswith('.yml'):
            with open(json_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            with open(json_path, 'r') as f:
                data = json.load(f)
        return data
    except Exception as e:
        sys.exit(f"Error loading file: {e}")

def extract_categories(data: Dict, sort_by_name: bool = False) -> List[Dict]:
    """
    Extract category information from instances.json or data.yaml
    
    Args:
        data: The loaded JSON data or YAML data
        sort_by_name: If True, sort categories alphabetically by name instead of by ID
    """
    try:
        # Check if this is YOLO data.yaml format
        if 'names' in data:
            categories = []
            for id_str, name in data['names'].items():
                categories.append({
                    'id': int(id_str),
                    'name': name
                })
        else:
            # Fallback to COCO format
            categories = data['categories']
            
        # Sort either by ID (default) or by name
        if sort_by_name:
            return sorted(categories, key=lambda x: x['name'].lower())
        else:
            return sorted(categories, key=lambda x: x['id'])
    except KeyError:
        sys.exit("Error: Input file does not contain valid category information")

def extract_class_name(name: str) -> str:
    """Extract the base class name before any modifiers"""
    # Remove common modifiers like _Normal, _Dark, etc.
    base_name = name.split('_')[0]
    return base_name

def generate_class_table(categories: List[Dict], output_path: str, existing_mappings: Optional[Dict[str, Dict[str, str]]] = None, sort_by_name: bool = False):
    """Generate a clean text file with class information and mapping template"""
    
    # Calculate maximum lengths for formatting
    max_id_len = max(len(str(cat['id'])) for cat in categories)
    max_name_len = max(len(cat['name']) for cat in categories)
    max_id_len = max(max_id_len, len("Class ID"))
    max_name_len = max(max_name_len, len("Current Class Name"))
    
    # Create header
    header = f"""# Class Mapping Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# Instructions:
# 1. First, define your new target classes in the "New Class Definitions" section below
#    - New class IDs MUST start at 0 and be sequential (0, 1, 2, etc.)
#    - Give each new class a descriptive label
#    - Add more rows as needed, maintaining sequential IDs
# 2. Then, in the "Class Mapping" section:
#    - Review the current classes
#    - In the "Map To Class" column, enter the new class ID from the definitions above
#    - Leave blank to keep the class as is
#    - Use 'remove' to exclude this class
#
# New Class Definitions:
# {'-' * 40}
# New Class ID | Class Label
# {'-' * 40}
# 0           | __________
# 1           | __________
# 2           | __________
# 3           | __________
# 4           | __________
# 5           | __________
# 6           | __________
# 7           | __________
# 8           | __________
# 9           | __________
# {'-' * 40}
#
# Class Mapping:
# {'-' * (max_id_len + max_name_len + 20)}
# {'Class ID'.ljust(max_id_len)} | {'Current Class Name'.ljust(max_name_len)} | Map To Class
# {'-' * (max_id_len + max_name_len + 20)}"""

    # Create class listing
    rows = []
    for cat in categories:
        class_id = str(cat['id'])
        class_name = cat['name']
        
        # Get mapping based on sort mode
        mapping = '__________'
        if existing_mappings:
            if sort_by_name:
                # When sorting by name, use name mappings
                if class_name in existing_mappings['name_mappings']:
                    mapping = existing_mappings['name_mappings'][class_name]
            else:
                # When sorting by ID, use ID mappings
                if class_id in existing_mappings['id_mappings']:
                    mapping = existing_mappings['id_mappings'][class_id]
        
        # Convert '9' to 'remove' for better clarity
        mapping = 'remove' if mapping == '9' else mapping
        row = f"{class_id.ljust(max_id_len)} | {class_name.ljust(max_name_len)} | {mapping}"
        rows.append(row)

    # Add footer with examples and note about existing mappings
    footer = f"""
# {'-' * (max_id_len + max_name_len + 20)}
#
### Example 1: Simple class merging
### New Class Definitions:
### New Class ID | Class Label
### 0           | Vehicle
### 1           | Animal
### 2           | Person
###
### Class Mapping:
### Class ID | Current Class Name     | Map To Class
### 0        | car                    | 0
### 1        | truck                  | 0
### 2        | motorcycle             | 0
### 3        | dog                    | 1
### 4        | cat                    | 1
### 5        | person                 | 2
###
### Example 2: Class removal and complex merging
### New Class Definitions:
### New Class ID | Class Label
### 0           | Fish
### 1           | Coral
###
### Class Mapping:
### Class ID | Current Class Name     | Map To Class
### 0        | grouper               | 0
### 1        | snapper               | 0
### 2        | parrotfish            | 0
### 3        | soft_coral            | 1
### 4        | hard_coral            | 1
### 5        | background            | remove
###
#
# The examples above show:
# 1. Simple merging of vehicle types, animals, and person classes
# 2. Complex merging with class removal (background class)
"""

    if existing_mappings:
        footer += """
# Note: Existing mappings from Families_class_changes have been pre-filled.
# Please review these mappings and ensure they align with your new class definitions.
# Any blank mappings (__________) need to be filled in or marked as 'remove'.
"""

    # Write to file
    with open(output_path, 'w') as f:
        f.write(header + '\n')
        f.write('\n'.join(rows))
        f.write(footer)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a clean class listing format from COCO instances_default.json')
    
    parser.add_argument('--dir', 
                       help='Directory containing annotations/instances_default.json',
                       required=True)
    
    parser.add_argument('--output',
                       help='Path to output text file (default: class_mapping.txt)',
                       default='class_mapping.txt')
    
    parser.add_argument('--existing-mappings',
                       help='Path to existing Families_class_changes file to pre-fill mappings',
                       default=None)
    
    parser.add_argument('--sort-by-name',
                       help='Flag to sort classes alphabetically by name instead of by ID (no value needed)',
                       action='store_true')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find and load instances_default.json
    json_path = find_instances_json(args.dir)
    print(f"Found instances_default.json at: {json_path}")
    
    # Load existing mappings if provided
    existing_mappings = None
    if args.existing_mappings:
        print(f"Loading existing mappings from: {args.existing_mappings}")
        existing_mappings = parse_class_changes(args.existing_mappings)
        print(f"Found {len(existing_mappings['id_mappings'])} existing mappings")
    
    # Load and process data
    data = load_instances_json(json_path)
    categories = extract_categories(data, args.sort_by_name)
    
    # Generate output
    generate_class_table(categories, args.output, existing_mappings, args.sort_by_name)
    print(f"Generated class mapping template at: {args.output}")
    if existing_mappings:
        print("Existing mappings have been pre-filled. Please add class labels to complete the mapping.")
    else:
        print("Edit this file to specify your class mappings")

if __name__ == '__main__':
    main() 