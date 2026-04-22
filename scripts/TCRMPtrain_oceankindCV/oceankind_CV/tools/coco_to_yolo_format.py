#!/usr/bin/env python3

"""
Author: Serena Mou (Modified by LO)
Created: 23 July 2024

===
Converts the Segment Anything Model (SAM) masks from CVAT in COCO format into YOLO compatible bounding boxes
If there are multiple jsons and the classes do not all match, a "all_classes_dict.yaml" will be required
===

"""

import json
import os
import yaml
import glob
import csv
import sys
import argparse
from pathlib import Path

class COCO2YOLOBB():
    def __init__(self, json_file, save_location):
        self.in_files = json_file
        self.save_location = save_location

    def get_info(self, data):
        """Extract information from COCO format data"""
        try:
            # List all the classes
            categories = data["categories"]
            classes = [category["name"] for category in categories]
            
            # List all the image filenames
            images = data["images"]
            img_names = [image["file_name"] for image in images]
            img_size = []
            for image in images:
                w = image["width"]
                h = image["height"]
                img_size.append([w,h])
            
            # For each annotation, get the class, image ID and the bbox
            annotations = data["annotations"]
            cls = [int(annotation["category_id"])-1 for annotation in annotations]
            img_ids = [int(annotation["image_id"])-1 for annotation in annotations]
            bbxs = [annotation["bbox"] for annotation in annotations]
            
            # Handle both SAM and regular COCO formats
            im_sz = []
            for annotation in annotations:
                if "segmentation" in annotation and isinstance(annotation["segmentation"], dict) and "size" in annotation["segmentation"]:
                    im_sz.append(annotation["segmentation"]["size"])
                else:
                    # Use image size if segmentation size not available
                    img_id = int(annotation["image_id"]) - 1
                    im_sz.append([images[img_id]["height"], images[img_id]["width"]])
            
            return classes, img_names, cls, img_ids, bbxs, im_sz
            
        except Exception as e:
            print(f"ERROR processing COCO format: {str(e)}")
            print("Check that the JSON file is in COCO 1.0 format.")
            raise

    def write_yaml(self, classes):
        """Generate YAML files without prompting for overwrites"""
        yaml_path = os.path.join(self.save_location, "data.yaml")
        test_yaml_path = os.path.join(self.save_location, "test.yaml")

        # Dictionary of {0: class0, 1: class1...}
        cls_dict = {k:v for k,v in enumerate(classes)}

        data = {
            "path": self.save_location,
            "train": "train",
            "val": "valid",
            "names": cls_dict
        }

        test = {
            "path": self.save_location,
            "train": "train",
            "val": "test",
            "names": cls_dict
        }

        with open(yaml_path, 'w') as outfile:
            yaml.dump(data, outfile, sort_keys=False)

        with open(test_yaml_path, 'w') as outfile:
            yaml.dump(test, outfile, sort_keys=False)

    def bbx_converter(self, bbx_raw, im_sz):
        """Convert COCO format bounding box to YOLO format with validation"""
        try:
            [xl, yl, w, h] = bbx_raw
            [fh, fw] = im_sz
            
            # Validate inputs
            if fw <= 0 or fh <= 0:
                raise ValueError(f"Invalid image dimensions: {fw}x{fh}")
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid box dimensions: {w}x{h}")
            
            # Convert to YOLO format (normalized)
            xn = (xl + (w/2))/fw
            yn = (yl + (h/2))/fh
            wn = (w/fw)
            hn = (h/fh)
            
            # Validate outputs
            if not (0 <= xn <= 1 and 0 <= yn <= 1 and 0 <= wn <= 1 and 0 <= hn <= 1):
                raise ValueError(f"Normalized coordinates out of range: {xn}, {yn}, {wn}, {hn}")
            
            return [xn, yn, wn, hn]
        except Exception as e:
            print(f"Error converting bounding box {bbx_raw} with image size {im_sz}: {str(e)}")
            raise

    def write_txt(self, classes, img_names, cls, img_ids, bbxs, im_sz, loop):
        """Write YOLO format annotation files"""
        out_folder = os.path.join(self.save_location, "all_labels")
        os.makedirs(out_folder, exist_ok=True)

        if loop == 0:
            self.write_yaml(classes)

        for i, name in enumerate(img_names):
            try:
                # Use pathlib for reliable filename handling
                path = Path(name)
                out_txt_name = path.stem + '.txt'
                
                # Get all annotations for this image
                all_im_idx = [j for j in range(len(img_ids)) if img_ids[j] == i]
                lines = []

                for idx in all_im_idx:
                    try:
                        idx_class = cls[idx]
                        bbox = self.bbx_converter(bbxs[idx], im_sz[idx])
                        lines.append((idx_class, *bbox))
                    except Exception as e:
                        print(f"Warning: Skipping annotation for image {name}, index {idx}: {str(e)}")
                        continue

                # Write annotations to file
                out_path = os.path.join(out_folder, out_txt_name)
                with open(out_path, 'w') as f:
                    for line in lines:
                        write_line = "%d %0.4f %0.4f %0.4f %0.4f" % line
                        f.write("%s\n" % write_line)
                
            except Exception as e:
                print(f"Warning: Failed to process image {name}: {str(e)}")
                continue

    def run(self):
        """Main conversion process"""
        try:
            # Handle single JSON file
            if os.path.isfile(self.in_files):
                all_in = [self.in_files]
            else:
                try:
                    all_in = glob.glob(self.in_files)
                except Exception as e:
                    print(f"ERROR: Failed to process input path pattern: {str(e)}")
                    return

            if not all_in:
                print("ERROR: No input files found")
                return

            # Process each JSON file
            for i, data_path in enumerate(all_in):
                try:
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    
                    classes, img_names, cls, img_ids, bbxs, im_sz = self.get_info(data)
                    self.write_txt(classes, img_names, cls, img_ids, bbxs, im_sz, i)
                    
                except Exception as e:
                    print(f"ERROR processing {data_path}: {str(e)}")
                    continue

        except Exception as e:
            print(f"ERROR in conversion process: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Convert from COCO SAM annotation to YOLO format')
    parser.add_argument("--json", dest="json_file", help="Path to JSON file or regex to JSON files", required=True)
    parser.add_argument("--save", dest="save_location", help="Path to save labels", required=True)
    args = parser.parse_args()

    converter = COCO2YOLOBB(args.json_file, args.save_location)
    converter.run()
    print("Conversion complete")

if __name__ == '__main__':
    main()
