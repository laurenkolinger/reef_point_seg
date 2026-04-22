# oceankind_CV
Computer vision pipeline for processing data from CVAT to train/validate/test with ultralytics for classification, detection and segmentation. Designed for Oceankind collaborators QUT, UVI, Berkeley and Point Blue. 

---
## 0. Get started

Clone this repository 

### Install miniconda
*If you have an existing conda installation, skip this step*

Follow instructions for your platform https://docs.anaconda.com/miniconda/
(want to install from command line!!!)

### Create conda environment and install ultralytics
As per https://docs.ultralytics.com/quickstart/#install-ultralytics 

```bash
conda create --name OK_CV 
conda activate OK_CV
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 wandb scikit-learn
```

Instead of conda installing ultralytics, install from git for development

In an appropriate folder: 
```bash
# Clone the ultralytics repository
git clone https://github.com/ultralytics/ultralytics

# Navigate to the cloned directory
cd ultralytics

# Install the package in editable mode for development
pip install -e .
```

### Check installation
Make sure the install was successful. If CUDA is not available AND you have a CUDA enabled GPU, make sure it is set up correctly. 

Installing pytorch separately may help https://pytorch.org/get-started/locally/ 

```bash
python <path_to_this_repo>/tools/test_install.py
```
---
## 1. Data Preparation
### CVAT Segment Anything Mask annotation to YOLO Bounding box format:
For converting CVAT Segment Anything mask annotations to YOLO bounding box format for training:

1. Go to Project -> Open Task -> Export annotations in COCO 1.0 format

    *CVAT has YOLOv8 bounding boxes built in as a format, however it does not convert from segmentation labels to bounding boxes. Skip this conversion step this if the annotations are already in bounding box format*

2. Download the zip and extract. It should be in the format annotation/instances_default.json

    If you have many different CVAT tasks for the same model, extract all into the same folder in following format:
```
└── Raw_labels
    ├── 0-100
        └── annotations
            └── instances_default.json
    ├── 101-200
        └── annotations
            └── instances_default.json
    └── 201-300
        └── annotations
            └── instances_default.json          
```

3. To convert to YOLO format:

**Example Use**

```bash
# Run conversion script
python tools/coco_to_yolo_format.py --json <path to Dataset>/Raw_labels/*/*/*.json --save <path to Dataset>/<Dataset name>/
```

**Options**
- ```--json str``` Path to a json file OR a regex to all the json files
- ```--save str``` Path to a folder to save the labels in

**Outputs**
- ```<save>/all_labels/``` Folder in --save with bounding box labels for each image in YOLO format
- ```<save>/data.yaml``` File with the data for YOLO training
- ```<save>/test.yaml``` File with the data for YOLO testing

### CVAT Segment Anything Mask annotation to YOLO segmentation format:
For training segmentation models, download from CVAT in YOLOv8 Segmentation 1.0 format. Once unzipped, the file structure is as follows:
```
└── Dataset
    ├── labels
        └── train
            ├── im1.txt 
            ├── im2.txt
            └── imn.txt
    ├── data.yaml
    └── train.txt        
```
To standardise file structure for merging, data splitting and training:
1. Move all label files into a folder named "all_labels" (no nested folders)
2. Edit data.yaml to point the "path" to your Dataset, edit "train" and add "val". See examples/data.yaml
3. Copy the data.yaml file into a file named "test.yaml". Change "val" to point to "test". See examples/test.yaml

The data should be structured as follows:
```
└── Dataset
    ├── all_labels
        ├── im1.txt 
        ├── im2.txt
        └── imn.txt
    ├── data.yaml
    └── test.yaml      
```

---
## Free labelling software

CVAT is free, to a point. For completely free labelling software, we can use label-studio (previously labelImg). label-studio does not currently support SAM so would be more suited to bounding box level annotations. It can produce segmentation, however you will need to hand paint the mask or hand draw the polygon. This method has been tested for Bounding Boxes.  

1. Create a new conda instance - label-studio has different version dependencies than ultralytics. 

```bash
# If you're still in the OK_CV environment
conda deactivate 

# Install
conda create --name label-studio
conda activate label-studio
conda install psycopg2
pip install label-studio

# Spin up
label-studio
```

2. Follow label-studio instructions to set up project, upload data, set up labelling interface and label names, and label your data. 

3. Once fully labelled, from the Project page, select Export and select YOLO format. 

4. Unzip the data. 

5. Run `label_studio_convert.py`

**Example usage:**
```bash
python tools/label_studio_convert.py --src /Downloads/<unzipped label studio folder> --dest /Data_folder/Animals
```

**Options:**
- `--src str` Path to unzipped label-studio folder
- `--dest str` Path to desired dataset location

**Outputs:**
- `<dest>/all_images` Folder with all the images for the dataset 
- `<dest>/all_labels` Folder with all the label files for the dataset 
- `<dest>/data.yaml` YAML file for YOLO training script 
- `<dest>/test.yaml` YAML file for YOLO test script 


6. Follow instructions starting from `2. Split data`





## 1.5. (Optional) Merge Classes

If you need to merge or remap classes, use this script. There are three ways to do this:

1. Using class_lister.py (Recommended)
2. Direct dictionary input (Legacy support)
3. Manual YAML workflow

### Using class_lister.py (Recommended)
This is the most user-friendly method. First, generate a class mapping template:

```bash
# Generate a class mapping template from your instances.json
python tools/class_lister.py --json path/to/instances.json --output class_mapping.txt
```

This will create a text file with your current classes and space to specify mappings:
```
# Class Mapping Configuration
# Generated: 2024-03-25 10:30:15
# 
# Class ID | Current Class Name | Map To Class | New Class Label
# --------------------------------------------------------
# 0        | dog               | __________ | __________
# 1        | cat               | __________ | __________
# 2        | person            | __________ | __________
```

Edit this file to specify your mappings:
1. In "Map To Class": Enter the new class ID or 'remove' to exclude
2. In "New Class Label": Enter a descriptive name for new classes

For example:
```
# Class ID | Current Class Name | Map To Class | New Class Label
# 0        | dog               | 1            | animals
# 1        | cat               | 1            | animals
# 2        | person            | 2            | people
```

Then run the merge. You have two options for handling images:

1. Using default directory structure:
   ```
   dataset/
   ├── all_labels/
   └── all_images/
   ```
   ```bash
   python tools/merge_classes.py --save <output path> --labels_in <path to labels> --mapping class_mapping.txt
   ```

2. Specifying custom images directory:
   ```bash
   python tools/merge_classes.py --save <output path> --labels_in <path to labels> --mapping class_mapping.txt --images_in <path to images>
   ```

The script will:
- Process label files according to your mapping
- Copy corresponding images to the output directory
- Create data.yaml and test.yaml with updated class configuration
- Generate class_merger.yaml documenting the mapping

### Legacy Methods

#### Direct dictionary input
For backwards compatibility, you can still use the dictionary format:

```bash
# Run merging script with class dictionary
python tools/merge_classes.py --save <output path> --labels_in <path to labels> --class_dict <path to mapping file>
```

The class mapping file should contain a Python dictionary:
```python
class_change = {
    '0': '1',    # Map class 0 to new class 1
    '1': '1',    # Map class 1 to new class 1
    '2': '2',    # Map class 2 to new class 2
    '3': '-1',   # Remove class 3
}
```

#### Manual YAML workflow
For advanced users who prefer working directly with YAML:

**Generate class_merger.yaml:**
```bash
python tools/merge_classes.py --save <output path> --data <path to data.yaml> --newclasses class0,class1,class2
```

**Merge using existing class_merger.yaml:**
```bash
python tools/merge_classes.py --save <output path> --merge <path to class_merger.yaml> --labels_in <path to labels>
```

### Common Options
- ```--save str``` Output path for merged labels and configuration
- ```--labels_in str``` Path to directory containing label files
- ```--mapping str``` Path to class mapping text file (from class_lister.py)
- ```--class_dict str``` Path to Python dictionary file (legacy format)
- ```--merge str``` Path to existing class_merger.yaml
- ```--data str``` Path to data.yaml containing original classes
- ```--newclasses str``` New class names for YAML workflow

### Outputs
For all methods:
- ```<save>/all_labels/``` Processed label files with updated class IDs
- ```<save>/data.yaml``` Training configuration with updated classes
- ```<save>/test.yaml``` Testing configuration with updated classes
- ```<save>/class_merger.yaml``` Generated merger configuration

---
## 2. Split data

To process the data into train/validate/test splits for YOLO, data needs to be in the following format:
```
└── Dataset
    ├── all_labels
        ├── im1.txt
        └── im2.txt      
    ├── all_images
        ├── im1.png
        └── im2.png           
    └── data.yaml   
```

*If the labels were generated by the coco_to_yolo_format.py script, they will already be in an `all_labels` folder.*

*If the annotations were in bounding box format, download annotations from CVAT in YOLOv8 Detection 1.0 format and extract into a folder named `all_labels`.*

*Images need to placed into `all_images` folder.*


### Create training data splits
For model training the combined data sets need to be split into training, validation and testing - YOLO tools expect the following structure:
```
└──/Dataset/
    ├──/train
	    ├──/images
	    └──/labels
    ├──/valid
	    ├──/images
	    └──/labels
    └──/test/
	    ├──/images
	    └──/labels
    ├── data.yaml    # Training configuration
    └── test.yaml    # Testing configuration
```

The `bal_train_test_split.py` tool facilitates this split with several key features:
- **Stratified Sampling**: Maintains class distribution ratios across splits for balanced representation
- **Class Validation**: Ensures each class has sufficient samples (default min: 10)
- **Progress Tracking**: Visual progress bars and detailed statistics
- **Reproducibility**: Uses fixed random seed for consistent results
- **Fallback Mechanism**: Gracefully handles edge cases with random split if stratification fails
- **Automatic YAML Generation**: Creates properly configured data.yaml and test.yaml files
- **Class Discovery**: Automatically detects and configures class information from labels

**Example usage:**
```bash
# Basic usage with default parameters (outputs to same directory as source)
python tools/bal_train_test_split.py --src /Dataset --valid 0.2 --test 0.1

# Specify custom output directory
python tools/bal_train_test_split.py --src /Dataset --out /Output/Dataset_Split --valid 0.2 --test 0.1

# Advanced usage with all options
python tools/bal_train_test_split.py --src /Dataset --out /Output/Dataset_Split --valid 0.2 --test 0.1 --min_samples 15 --rand 42 --dump 5
```

**Options:**
- `--src str` Source folder to search, expects `/src/all_images`, `/src/all_labels`
- `--out str` Output directory for splits (default: same as source)
- `--valid float` Percentage of dataset to use for validation (default=0.2)
- `--test float` Percentage of dataset to use for testing (default=None)
- `--min_samples int` Minimum samples required per class (default=10)
- `--rand int` Random seed for reproducibility (default=1)
- `--dump int` Number of empty/unlabeled images to remove (default=None)

**Outputs:**
- `<out>/train/images/` Training set images (typically 70-80% of data)
- `<out>/train/labels/` YOLO format labels for training set
- `<out>/valid/images/` Validation set images (typically 10-20% of data)
- `<out>/valid/labels/` YOLO format labels for validation set
- `<out>/test/images/` Test set images (typically 10-20% of data, if enabled)
- `<out>/test/labels/` YOLO format labels for test set
- `<out>/data.yaml` Training configuration with paths and class information
- `<out>/test.yaml` Testing configuration with paths and class information

The script provides detailed statistics about:
- Class distribution before and after splitting
- Number of samples in each split
- Any classes removed due to insufficient samples
- Progress during file operations
- Final class configuration and paths in YAML files

For simple random splitting without class balancing, see the original `train_test_split.py`.

---
## 3. Training

### (Optional) Visualisation
Create a wandb account (www.wandb.ai) and login in terminal for visualisations
```bash
wandb login
```

### Training Workflow
The training process is now split into two steps:

1. **Generate Training Configuration**
   ```bash
   python training/pre_train.py --data <path_to_data.yaml> --out <output_directory>
   ```
   This will:
   - Create the output directory for training
   - Generate a configuration file with:
     - Dataset metadata and statistics
     - Default training parameters with explanations
     - Impact descriptions for each parameter
   - Example:
   ```bash
   python training/pre_train.py --data Fish_Photo_Annotations_28feb25_YOLO_merged_split/data.yaml --out Fish_Photo_Annotations_28feb25_YOLO_merged_split_model
   ```

2. **Review and Start Training**
   - Review the generated `train_config.txt` in your output directory
   - Modify training parameters if needed (the file contains explanations and impact descriptions)
   - Start training:
   ```bash
   python training/train.py --config <path_to_train_config.txt>
   ```
   - Example:
   ```bash
   python training/train.py --config Fish_Photo_Annotations_28feb25_YOLO_merged_split_model/train_config.txt
   ```

The configuration file includes:
- Metadata about the dataset
- Dataset statistics (class distribution, image counts)
- Customizable training parameters with descriptions:
  - Model type (yolov8n.pt to yolov8x.pt)
  - Image size
  - Number of epochs
  - Early stopping patience
  - Batch size
  - Learning rates
  - Data augmentation settings
  - And more...

**Note**: The training outputs will be saved in the specified output directory under the `OK_CV` project folder.

## 4. Testing

Test the weights on unseen test data. A `test.yaml` will have been generated by the `coco_to_yolo_format.py`. If one needs to be generated, see `examples/test.yaml` to use as template. It is important to use the `test.yaml`, not the previously used `data.yaml`

**Example usage:**
```bash
python <path_to_this_repo>/training/test.py --src /Dataset/test.yaml --weights /Dataset/OK_CV/Animal_Train/weights/best.pt --name Animal_Test
```
***The outputs will be saved to the pwd (present working directory- path the script is being run from).***


**Options:**
- `--src str` Source YOLO yaml file describing test dataset
- `--weights str` Path to the model weight file for evaluation
- `--name str` Model name for saving

**Outputs:**
- `/pwd/OK_CV/name/` Model test stats and outputs

---
# Helpers:

To visualise bounding box annotations, use the visualise_bb_annots.py. Only the --src argument is required if the data is organised as above. If you would like to view just a smaller subset of image/label pairs, paste the desired label files into a separate folder and add the folder name to --label_folder. The img_folder can be left as default, just the images matching the label files in the smaller subset will be shown. 

**Example usage**
```bash
python tools/visualise_bb_annots.py --src <path to dataset root dir>
```

**Options**
- `--src str` Path to the root directory of your data
- `--img_folder str` (Optional) If the folder of images is not the default <src>/all_images, add name of folder
- `--label_folder str` (Optional) If the folder of labels is not the default <src>/all_labels, add name of folder
- `--data str` (Optional) If the yaml file with class names is not the default <src>/data.yaml, add name of file
- `--scale float` (Optional) Scale the image size up or down. Default is 0.5.

**Outputs:**
OpenCV window showing images with bounding boxes and name of classes. Press any key to go to the next image, press ESC to escape. 

