# Segment Vehicle Lamp Model (SVLM)

## Pre-requisites
Do ensure the following dependencies have been installed:
- PyTorch (with GPU support)
- cudnn and cuda-toolkit (with Nvidia GPU)
- conda ([miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install) preferred) 
- Visual Studio 2019 Build Tools (if on Windows and not Linux)

The following packages cudnn and cuda-toolkit may be easily installed using conda:
```
conda install -c conda-forge cudnn cuda-toolkit
```
The following packages should also be installed for opencv (example displayed below only works for Debian-based distros):
```
apt-get update

apt-get -y install build-essential ffmpeg libsm6 libxext6
```

Conda (Anaconda/Miniconda) will be used as the main virtual environment manager for this project. It allows for the creation of virtual environmtn
and installation of non-python dependencies such as cudnn and cuda-toolkit.

Poetry will be used to manage Python dependencies.

## Development
Conda (Anaconda or Miniconda) is highly recommended to be used in development. A conda virtual environment should be created and dependencies should
be installed within the venv.

```bash
# Create a venv named SVLM
conda create -n svlm python

# Activate the SVLM venv
conda activate svlm

poetry install

# Install the Albumentations plugin for Fiftyone
fiftyone plugins download https://github.com/jacobmarks/fiftyone-albumentations-plugin

# Run the program
python svlm\main.py
```

Additional scripts are available in the `scripts/` directory which may be used for data cleaning.

If you're using Albumentations on Windows, you may need to manually create the tmp/ directory in C drive.

## Usage

### Augmentation

More frequent than not, a dataset may be augmented to synthetically generate new data based on existing data. This may be done by employing a number of different techniques such as cropping, and adjusting the hue, brightness, contrast or saturation of the image.

However, augmentation must be applied with caution so as to not expose the model to data that is not an accurate representation of real-world data. For instance, distorting the image too much until the model is unable to identify the presence of a car.

`svlm` enables augmentation by integrating with `Fiftyone` and `Albumentations`. Though, before proceeding with augmentation, the `config.yml` file must be updated with the paths to the dataset.

To view and augment the dataset:
```Bash
python svlm\main.py augment
```

### Export

After augmentation, the augmented images must be saved on Fiftyone in order for them to be persisted during export. Otherwise, only the original images will be exported.

The `config.yml` file must be modified to include the path of the output directory and the name of the loaded dataset on Fiftyone's database.

To export the augmented dataset to `COCODetectionDataset` format:
```Bash
python svlm\main.py export
```

### Split

A dataset should be split to 3 seperate folders, train:test:val, and then structured according to the [YOLO dataset format][yolo-dataset-format] (see [Sample Datasets](#sample-datasets) section).

To split and convert to YOLO format:
```Bash
# python svlm\main.py split IMAGES LABELS OUTPUT
python svlm\main.py split ./coco-dataset/images ./coco-dataset/labels.json ./yolo-dataset
```


## Sample Datasets

COCO dataset should be used for augmentation as it preserves bounding boxes and segmentation masks after importing into Fiftyone for augmentation.
After augmentation, dataset will be exported in the YOLO format.

> Datasets and models can be found in the STA shared drive on the NAS.

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

## Contribution

### Pre-commit tool
This project uses the pre-commit tool to maintain code quality and consistency. Before submitting a pull request or making any commits, it is important to run the pre-commit tool to ensure that your changes meet the project's guidelines.

To run the pre-commit tool, follow these steps:

1. Install pre-commit by running the following command: `poetry install`. It will not only install pre-commit but also install all the deps and dev-deps of project

2. Once pre-commit is installed, navigate to the project's root directory.

3. Run the command `pre-commit run --all-files`. This will execute the pre-commit hooks configured for this project against the modified files. If any issues are found, the pre-commit tool will provide feedback on how to resolve them. Make the necessary changes and re-run the pre-commit command until all issues are resolved.

4. You can also install pre-commit as a git hook by execute `pre-commit install`. Every time you made `git commit` pre-commit run automatically for you.

### Docstrings
All new functions and classes in `svlm` should include docstrings. This is a prerequisite for any new functions and classes to be added to the project.

`svlm` adheres to the [Sphinx Python docstring style][sphinx-docstring-style]. Please refer to the style guide while writing docstrings for your contribution.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[yolo-dataset-format]: https://docs.ultralytics.com/datasets/segment/#dataset-yaml-format
[sphinx-docstring-style]: https://google.github.io/styleguide/pyguide.html#383-functions-and-methods
