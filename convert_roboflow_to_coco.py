#!/usr/bin/env python3
"""
Convert Roboflow format (individual JSON per image) to COCO format.

The data in /workspace/sam3_lora/data/ appears to be in Roboflow format where:
- Each image has a corresponding JSON file
- JSON contains image info and annotations for that image

This script converts to standard COCO format with a single _annotations.coco.json file.
"""

import json
import os
import glob
from pathlib import Path


def convert_roboflow_to_coco(data_dir: str, output_file: str = "_annotations.coco.json"):
    """
    Convert Roboflow format to COCO format.

    Args:
        data_dir: Directory containing images and JSON files
        output_file: Output COCO JSON file name
    """
    # Find all JSON files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} JSON files")

    # Initialize COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Track categories
    category_map = {}
    category_id = 1
    annotation_id = 1
    image_id = 1

    # Process each JSON file
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Get image info
            if 'image' in data:
                image_info = data['image']

                # Add image entry
                coco_image = {
                    "id": image_id,
                    "file_name": image_info.get('file_name', os.path.basename(json_file).replace('.json', '')),
                    "height": image_info.get('height', 0),
                    "width": image_info.get('width', 0),
                }
                coco_data['images'].append(coco_image)

                # Process annotations
                if 'annotations' in data:
                    for ann in data['annotations']:
                        # Add annotation
                        coco_ann = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "bbox": ann.get('bbox', [0, 0, 0, 0]),
                            "area": ann.get('area', 0),
                            "category_id": 1,  # Default category
                            "iscrowd": 0,
                        }

                        # Add segmentation if available
                        if 'segmentation' in ann:
                            coco_ann['segmentation'] = ann['segmentation']

                        coco_data['annotations'].append(coco_ann)
                        annotation_id += 1

                image_id += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    # Add default category
    coco_data['categories'] = [
        {"id": 1, "name": "object", "supercategory": "object"}
    ]

    # Write COCO JSON
    output_path = os.path.join(data_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"\n✓ Converted to COCO format:")
    print(f"  Output: {output_path}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")


if __name__ == "__main__":
    import sys

    # Convert train, valid, and test sets
    for split in ['train', 'valid', 'test']:
        data_dir = f'/workspace/sam3_lora/data/{split}'
        if os.path.exists(data_dir):
            print(f"\n{'='*50}")
            print(f"Converting {split} set...")
            print(f"{'='*50}")
            convert_roboflow_to_coco(data_dir)
        else:
            print(f"\nSkipping {split} (directory not found)")
