#!/usr/bin/env python3

"""
Author: LO
Created: March 2024

Generate training configuration and metadata file for YOLO training
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
from pathlib import Path

def arg_parse():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate training configuration')
    parser.add_argument("--data", dest="data_yaml",
            help="Path to data.yaml file", required=True, type=str)
    parser.add_argument("--out", dest="out_dir",
            help="Output directory for training", required=True, type=str)
    return parser.parse_args()

def count_files(directory):
    """Count files in directory recursively"""
    try:
        directory = Path(directory)
        if not directory.exists():
            print(f"Warning: Directory does not exist: {directory}")
            return 0
        # Count only files, not directories
        return sum(1 for _ in directory.rglob('*') if _.is_file())
    except Exception as e:
        print(f"Warning: Error counting files in {directory}: {e}")
        return 0

def validate_yaml_structure(config):
    """Validate required fields in YAML configuration"""
    required_fields = ['path', 'train', 'val', 'names']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in YAML: {', '.join(missing_fields)}")

def resolve_path(base_path, relative_path):
    """Resolve a path relative to the base path"""
    try:
        base = Path(base_path)
        # Handle both absolute and relative paths
        full_path = Path(relative_path) if Path(relative_path).is_absolute() else base / relative_path
        return full_path
    except Exception as e:
        print(f"Warning: Error resolving path {relative_path} relative to {base_path}: {e}")
        return None

def count_class_distribution(labels_path):
    """Count the distribution of classes in a labels directory"""
    distribution = {}
    if not labels_path or not os.path.exists(labels_path):
        return distribution
        
    for label_file in os.listdir(labels_path):
        if not label_file.endswith('.txt'):
            continue
            
        with open(os.path.join(labels_path, label_file), 'r') as f:
            for line in f:
                if line.strip():
                    class_id = line.strip().split()[0]
                    distribution[class_id] = distribution.get(class_id, 0) + 1
                    
    return distribution

def get_dataset_stats(data_yaml):
    """Get statistics about the dataset"""
    try:
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
            print("\nLoaded class names from data.yaml:")
            print(config.get('names', {}))
        
        # Validate YAML structure
        validate_yaml_structure(config)
        
        # Get base path
        base_path = Path(config['path'])
        if not base_path.exists():
            print(f"Warning: Base path does not exist: {base_path}")
        
        # Resolve paths
        train_images_path = resolve_path(base_path, config['train'])
        train_labels_path = resolve_path(base_path, config['train'].replace('images', 'labels'))
        val_images_path = resolve_path(base_path, config['val'])
        val_labels_path = resolve_path(base_path, config['val'].replace('images', 'labels'))
        
        # Handle test path if present
        test_images_path = None
        test_labels_path = None
        if 'test' in config:
            test_images_path = resolve_path(base_path, config['test'])
            test_labels_path = resolve_path(base_path, config['test'].replace('images', 'labels'))
        
        # Count files and class distributions
        train_dist = count_class_distribution(train_labels_path)
        val_dist = count_class_distribution(val_labels_path)
        test_dist = count_class_distribution(test_labels_path) if test_labels_path else {}
        
        stats = {
            'num_classes': len(config['names']),
            'class_names': config['names'],
            'train_images': count_files(train_images_path),
            'train_labels': count_files(train_labels_path),
            'train_distribution': train_dist,
            'val_images': count_files(val_images_path),
            'val_labels': count_files(val_labels_path),
            'val_distribution': val_dist,
            'test_images': count_files(test_images_path) if test_images_path else 0,
            'test_labels': count_files(test_labels_path) if test_labels_path else 0,
            'test_distribution': test_dist
        }
        
        # Print debug information
        print("\nDataset paths:")
        print(f"Base path: {base_path}")
        print(f"Train images: {train_images_path}")
        print(f"Train labels: {train_labels_path}")
        print(f"Validation images: {val_images_path}")
        print(f"Validation labels: {val_labels_path}")
        if test_images_path:
            print(f"Test images: {test_images_path}")
            print(f"Test labels: {test_labels_path}")
        
        return stats, config
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        raise

def generate_config_file(out_dir, data_yaml, stats, config):
    """Generate the configuration file with metadata and parameters"""
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'train_config.txt')
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            # Write header with ASCII art
            f.write("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         YOLO Training Configuration                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
            
            # Metadata section
            f.write("\n┌─ METADATA "+"─"*67+"┐\n")
            f.write(f"│ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"│ Data YAML: {os.path.abspath(data_yaml)}\n")
            f.write(f"│ Output Directory: {os.path.abspath(out_dir)}\n")
            f.write(f"│ Python Executable: {sys.executable}\n")
            f.write("└" + "─"*78 + "┘\n")
            
            # Dataset statistics
            f.write("\n┌─ DATASET STATISTICS "+"─"*59+"┐\n")
            f.write(f"│ Number of Classes: {stats['num_classes']}\n")
            f.write("│ Class Names and Distribution:\n")
            
            # Show distribution for each class across splits
            for class_id in range(stats['num_classes']):
                # Try both string and integer keys
                class_name = stats['class_names'].get(str(class_id)) or stats['class_names'].get(class_id)
                if class_name is None:
                    print(f"Warning: No name found for class {class_id}")
                    class_name = f"class_{class_id}"
                
                train_count = stats['train_distribution'].get(str(class_id), 0)
                val_count = stats['val_distribution'].get(str(class_id), 0)
                test_count = stats['test_distribution'].get(str(class_id), 0)
                total = train_count + val_count + test_count
                
                f.write(f"│   • Class {class_id} ({class_name}):\n")
                f.write(f"│     - Total: {total} instances\n")
                f.write(f"│     - Training: {train_count} instances\n")
                f.write(f"│     - Validation: {val_count} instances\n")
                if stats['test_images'] > 0:
                    f.write(f"│     - Test: {test_count} instances\n")
            f.write("│\n")
            
            f.write("│\n│ Dataset Split:\n")
            f.write(f"│   • Training Set:    {stats['train_images']} images, {stats['train_labels']} labels\n")
            f.write(f"│   • Validation Set:  {stats['val_images']} images, {stats['val_labels']} labels\n")
            if stats['test_images'] > 0:
                f.write(f"│   • Test Set:        {stats['test_images']} images, {stats['test_labels']} labels\n")
            
            # Calculate and display ratios
            if stats['train_images'] > 0:
                train_ratio = stats['train_images'] / (stats['train_images'] + stats['val_images']) * 100
                val_ratio = stats['val_images'] / (stats['train_images'] + stats['val_images']) * 100
                f.write("│\n│ Split Ratios:\n")
                f.write(f"│   • Training:    {train_ratio:.1f}%\n")
                f.write(f"│   • Validation:  {val_ratio:.1f}%\n")
                if stats['test_images'] > 0:
                    test_ratio = stats['test_images'] / (stats['train_images'] + stats['val_images'] + stats['test_images']) * 100
                    f.write(f"│   • Test:        {test_ratio:.1f}%\n")
            f.write("└" + "─"*78 + "┘\n")
            
            # Training parameters
            f.write("\n┌─ TRAINING PARAMETERS "+"─"*58+"┐\n")
            f.write("│ Instructions:\n")
            f.write("│ • Edit the values below to customize your training\n")
            f.write("│ • DO NOT modify the parameter names or remove any lines\n")
            f.write("│ • Leave a value unchanged if you're unsure about its impact\n")
            f.write("│ • Values must be on the same line as the parameter name\n")
            f.write("└" + "─"*78 + "┘\n\n")
            
            # Parameter categories
            parameters = {
                "Model Configuration": {
                    "model": {
                        "value": "yolov8m.pt",
                        "description": "Model type (options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)",
                        "impact": "Larger models are more accurate but slower and require more memory"
                    },
                    "imgsz": {
                        "value": 512,
                        "description": "Input image size (pixels)",
                        "impact": "Larger sizes may improve accuracy for small objects but require more memory"
                    }
                },
                "Training Schedule": {
                    "epochs": {
                        "value": 500,
                        "description": "Number of training epochs",
                        "impact": "More epochs allow for better convergence but increase training time"
                    },
                    "patience": {
                        "value": 50,
                        "description": "Early stopping patience (epochs without improvement)",
                        "impact": "Lower values may stop training early, higher values give more chances to improve"
                    },
                    "batch": {
                        "value": -1,
                        "description": "Batch size (-1 for auto-batch)",
                        "impact": "Larger batches are more efficient but require more memory"
                    }
                },
                "Optimization": {
                    "workers": {
                        "value": 8,
                        "description": "Number of worker threads for data loading",
                        "impact": "More workers can speed up training but use more CPU resources"
                    },
                    "lr0": {
                        "value": 0.01,
                        "description": "Initial learning rate",
                        "impact": "Higher values may train faster but risk instability"
                    },
                    "lrf": {
                        "value": 0.01,
                        "description": "Final learning rate factor",
                        "impact": "Controls how much the learning rate decreases during training"
                    }
                },
                "Data Augmentation": {
                    "scale": {
                        "value": 0.2,
                        "description": "Image scale augmentation factor (0-1)",
                        "impact": "Higher values increase scale variation in training"
                    },
                    "flipud": {
                        "value": 0.5,
                        "description": "Vertical flip probability (0-1)",
                        "impact": "Helps model learn orientation invariance"
                    },
                    "fliplr": {
                        "value": 0.5,
                        "description": "Horizontal flip probability (0-1)",
                        "impact": "Helps model learn orientation invariance"
                    }
                }
            }
            
            for category, params in parameters.items():
                f.write(f"┌─ {category} " + "─"*(77-len(category)) + "┐\n")
                for name, param in params.items():
                    f.write("│\n")
                    f.write(f"│ {param['description']}\n")
                    f.write(f"│ Impact: {param['impact']}\n")
                    f.write(f"│ {name}: {param['value']}\n")
                f.write("└" + "─"*78 + "┘\n\n")
                
    except UnicodeEncodeError:
        # Fallback to simple ASCII if UTF-8 fails
        with open(config_path, 'w') as f:
            # Write header with simple ASCII
            f.write("="*80 + "\n")
            f.write("|" + " "*30 + "YOLO Training Configuration" + " "*29 + "|\n")
            f.write("="*80 + "\n\n")
            
            # Metadata section
            f.write("METADATA:\n")
            f.write("-"*40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data YAML: {os.path.abspath(data_yaml)}\n")
            f.write(f"Output Directory: {os.path.abspath(out_dir)}\n")
            f.write(f"Python Executable: {sys.executable}\n\n")
            
            # Dataset statistics
            f.write("DATASET STATISTICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Number of Classes: {stats['num_classes']}\n")
            f.write("Class Names and Distribution:\n")
            
            # Show distribution for each class across splits
            for class_id in range(stats['num_classes']):
                class_name = stats['class_names'].get(str(class_id)) or stats['class_names'].get(class_id)
                if class_name is None:
                    print(f"Warning: No name found for class {class_id}")
                    class_name = f"class_{class_id}"
                
                train_count = stats['train_distribution'].get(str(class_id), 0)
                val_count = stats['val_distribution'].get(str(class_id), 0)
                test_count = stats['test_distribution'].get(str(class_id), 0)
                total = train_count + val_count + test_count
                
                f.write(f"  * Class {class_id} ({class_name}):\n")
                f.write(f"    - Total: {total} instances\n")
                f.write(f"    - Training: {train_count} instances\n")
                f.write(f"    - Validation: {val_count} instances\n")
                if stats['test_images'] > 0:
                    f.write(f"    - Test: {test_count} instances\n")
            f.write("\n")
            
            # Dataset split information
            f.write("\nDataset Splits:\n")
            f.write(f"  * Training Set:    {stats['train_images']} images, {stats['train_labels']} labels\n")
            f.write(f"  * Validation Set:  {stats['val_images']} images, {stats['val_labels']} labels\n")
            if stats['test_images'] > 0:
                f.write(f"  * Test Set:        {stats['test_images']} images, {stats['test_labels']} labels\n")
            
            # Calculate and display ratios
            if stats['train_images'] > 0:
                train_ratio = stats['train_images'] / (stats['train_images'] + stats['val_images']) * 100
                val_ratio = stats['val_images'] / (stats['train_images'] + stats['val_images']) * 100
                f.write("\nSplit Ratios:\n")
                f.write(f"  * Training:    {train_ratio:.1f}%\n")
                f.write(f"  * Validation:  {val_ratio:.1f}%\n")
                if stats['test_images'] > 0:
                    test_ratio = stats['test_images'] / (stats['train_images'] + stats['val_images'] + stats['test_images']) * 100
                    f.write(f"  * Test:        {test_ratio:.1f}%\n")
            f.write("\n")
            
            # Training parameters
            f.write("\nTRAINING PARAMETERS:\n")
            f.write("-"*40 + "\n")
            f.write("Instructions:\n")
            f.write("* Edit the values below to customize your training\n")
            f.write("* DO NOT modify the parameter names or remove any lines\n")
            f.write("* Leave a value unchanged if you're unsure about its impact\n")
            f.write("* Values must be on the same line as the parameter name\n\n")
            
            # Parameter categories
            for category, params in parameters.items():
                f.write(f"\n{category}:\n")
                f.write("-"*40 + "\n")
                for name, param in params.items():
                    f.write(f"\n{param['description']}\n")
                    f.write(f"Impact: {param['impact']}\n")
                    f.write(f"{name}: {param['value']}\n")
                f.write("\n")
    
    return config_path

def main():
    args = arg_parse()
    
    try:
        # Get dataset statistics
        print("Analyzing dataset...")
        stats, config = get_dataset_stats(args.data_yaml)
        
        # Generate configuration file
        print("Generating configuration file...")
        config_path = generate_config_file(args.out_dir, args.data_yaml, stats, config)
        
        print("\nConfiguration generated successfully!")
        print(f"Configuration file: {config_path}")
        print("\nNext steps:")
        print("1. Review and edit the training parameters in the configuration file if needed")
        print("2. Run training with:")
        print(f"   python training/train.py --config {config_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 