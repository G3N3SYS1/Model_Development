# Segment Vehicle Lamp Model (SVLM)

## Pre-requisites
Do ensure the following dependencies have been installed:
- PyTorch (with GPU support)
- cudnn and cuda-toolkit
- conda
- Visual Studio 2019 Build Tools (if on Windows)

The follwing packages cudnn and cuda-toolkit may be easily installed using conda:
```
conda install -c conda-forge cudnn cuda-toolkit
```

## Development
Conda (Anaconda or Miniconda) is highly recommended to be used in development. A conda virtual environment should be created and dependencies should
be installed within the venv.

```bash
# Create a venv named SVLM
conda create -n svlm python

# Activate the SVLM venv
conda activate svlm

pip install -r requirements.txt

# Run the program
python svlm\main.py
```

Additional scripts are available in the `scripts/` directory which may be used for data cleaning.

## Sample Datasets

COCO dataset should be used for augmentation as it preserves bounding boxes and segmentation masks after importing into Fiftyone for augmentation. 
After augmentation, dataset will be exported in the YOLO format.

Dataset should be in the COCO format when used as input for Fiftyone as follows:

```
coco/
├─ images/
│  ├─ image1.png
├─ result.json

```

Dataset should be in the YOLO format when used as input for YOLOv8 training as follows:

```
yolo/
├─ images/
│  ├─ image1.png
├─ labels/
│  ├─ image1.txt
```
