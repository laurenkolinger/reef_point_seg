#!/usr/bin/env python3

"""
Author: Matthew Tavatgis, Updated by SM, LO
Created: 9th Feb 2024

Use Ultralytics to train a YOLO model with configuration file
"""

import sys
import argparse
import os
import yaml
from ultralytics import YOLO
from datetime import datetime

def arg_parse():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLO Model')
    parser.add_argument("--config", dest="config",
            help="Path to training configuration file", required=True, type=str)
    parser.add_argument("--project", dest="project_dir",
            help="Directory to store training results (defaults to same directory as config file)", default=None, type=str)
    return parser.parse_args()

def load_config(config_path):
    """Load and parse the configuration file"""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
        
    config = {}
    current_section = None
    
    try:
        # Try UTF-8 first
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and purely decorative lines
                if not line or line.startswith('╔') or line.startswith('╚') or line.startswith('═') or line.startswith('║'):
                    continue
                
                # Skip section headers and footers
                if line.startswith('┌') or line.startswith('└'):
                    continue
                
                # Remove vertical bar and trim
                if line.startswith('│'):
                    line = line[1:].strip()
                
                if ':' in line:
                    key, value = [x.strip() for x in line.split(':', 1)]
                    # Remove bullet points and other formatting
                    key = key.replace('•', '').replace('*', '').strip()
                    value = value.strip()
                    
                    # Convert key to lowercase for comparison
                    key_lower = key.lower()
                    
                    # Handle metadata fields
                    if key_lower == 'data yaml':
                        config['data_yaml'] = value
                        continue
                    elif key_lower == 'output directory':
                        config['output_dir'] = value
                        continue
                    
                    # Handle training parameters
                    if key_lower in ['model', 'imgsz', 'epochs', 'patience', 'batch', 
                                   'workers', 'lr0', 'lrf', 'scale', 'flipud', 'fliplr']:
                        try:
                            # Convert to appropriate type
                            if key_lower in ['imgsz', 'epochs', 'patience', 'workers']:
                                config[key_lower] = int(value)
                            elif key_lower == 'batch':
                                config[key_lower] = int(value) if value != '-1' else -1
                            elif key_lower in ['lr0', 'lrf', 'scale', 'flipud', 'fliplr']:
                                config[key_lower] = float(value)
                            else:
                                config[key_lower] = value.strip()
                        except ValueError as e:
                            print(f"Error parsing value for {key}: {value}")
                            sys.exit(1)
    except UnicodeDecodeError:
        # Fallback to reading with default encoding
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and decorative lines
                if not line or line.startswith('=') or line.startswith('-'):
                    continue
                
                # Remove vertical bar and trim
                if line.startswith('|'):
                    line = line[1:].strip()
                
                if ':' in line:
                    key, value = [x.strip() for x in line.split(':', 1)]
                    # Remove formatting
                    key = key.replace('*', '').strip()
                    value = value.strip()
                    
                    # Convert key to lowercase for comparison
                    key_lower = key.lower()
                    
                    # Handle metadata fields
                    if key_lower == 'data yaml':
                        config['data_yaml'] = value
                        continue
                    elif key_lower == 'output directory':
                        config['output_dir'] = value
                        continue
                    
                    # Handle training parameters
                    if key_lower in ['model', 'imgsz', 'epochs', 'patience', 'batch', 
                                   'workers', 'lr0', 'lrf', 'scale', 'flipud', 'fliplr']:
                        try:
                            # Convert to appropriate type
                            if key_lower in ['imgsz', 'epochs', 'patience', 'workers']:
                                config[key_lower] = int(value)
                            elif key_lower == 'batch':
                                config[key_lower] = int(value) if value != '-1' else -1
                            elif key_lower in ['lr0', 'lrf', 'scale', 'flipud', 'fliplr']:
                                config[key_lower] = float(value)
                            else:
                                config[key_lower] = value.strip()
                        except ValueError as e:
                            print(f"Error parsing value for {key}: {value}")
                            sys.exit(1)
    
    # Print debug information
    print("\nParsed configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    return config

def main():
    # Parse arguments
    args = arg_parse()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Set project directory to config file directory if not specified
    if args.project_dir is None:
        args.project_dir = os.path.dirname(os.path.abspath(args.config))
    
    # Validate required fields
    required_fields = ['data_yaml', 'output_dir', 'model']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        print(f"Error: Missing required fields in configuration: {', '.join(missing_fields)}")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*80)
    print("YOLO Training Configuration")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration File: {os.path.abspath(args.config)}")
    print(f"Data YAML: {config['data_yaml']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Project Directory: {os.path.abspath(args.project_dir)}")
    print(f"Model: {config['model']}")
    
    print("\nTraining Parameters:")
    print("-"*40)
    for key in ['imgsz', 'epochs', 'patience', 'batch', 'workers', 'lr0', 'lrf', 
                'scale', 'flipud', 'fliplr']:
        if key in config:
            print(f"{key}: {config[key]}")
    print("="*80 + "\n")

    try:
        # Create project directory if it doesn't exist
        os.makedirs(args.project_dir, exist_ok=True)
        
        # Load model
        print("Loading model...")
        model = YOLO(config['model'])
        print("Model loaded successfully")

        # Start training
        print("\nStarting training...")
        metrics = model.train(
            data=config['data_yaml'],
            epochs=config.get('epochs', 500),
            patience=config.get('patience', 50),
            batch=config.get('batch', -1),
            imgsz=config.get('imgsz', 512),
            save=True,
            device=0,
            workers=config.get('workers', 8),
            project=args.project_dir,
            name=os.path.basename(config['output_dir']),
            val=True,
            lr0=config.get('lr0', 0.01),
            lrf=config.get('lrf', 0.01),
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            pose=12.0,
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,
            hsv_h=0.2,
            hsv_s=0.3,
            hsv_v=0.3,
            degrees=0.0,
            translate=0.0,
            scale=config.get('scale', 0.2),
            shear=0.0,
            perspective=0.0,
            flipud=config.get('flipud', 0.5),
            fliplr=config.get('fliplr', 0.5),
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            auto_augment="randaugment",
            erasing=0.0,
            crop_fraction=1.0
        )
        
        print("\nTraining completed successfully")
        print(f"Training duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nFinal Metrics:")
        print("-"*40)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
