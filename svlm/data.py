import logging
import os

import fiftyone as fo
import supervision as sv
from fiftyone.types import COCODetectionDataset

DEFAULT_DATASET_TYPE = COCODetectionDataset

log = logging.getLogger(__name__)


def load(
    name,
    data_path,
    labels_path,
    dataset_type=DEFAULT_DATASET_TYPE,
):
    existing_datasets = fo.list_datasets()
    log.debug(f"Existing datasets: {existing_datasets}")

    if name not in existing_datasets:
        log.info(f"Creating a new dataset, {name}...")
        dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=data_path,
            labels_path=labels_path,
            name=name,
        )
    else:
        log.info("Loading dataset...")
        dataset = fo.load_dataset(name)

    print(dataset.view())
    return dataset


def export(
    dataset,
    export_dir: str,
    label_field,
    dataset_type=DEFAULT_DATASET_TYPE,
):
    """Export dataset from Fiftyone in the form of a COCO dataset unless
    specified otherwise, which preserves both bbox coordinates and
    segmentation masks of objects. The dataset should then be converted to
    YOLO format using data.split() before feeding to YOLOv8 for training.

    :param dataset: An instance of a Fiftyone Dataset
    :type dataset: fiftyone.Dataset
    :param export_dir: Path to export directory to store the exported dataset
    :type export_dir: str
    :param classes: An array of strings consisting of unique classes
    :type classes: list[str]
    :param label_field: Label in the dataset to export. Labels can be viewed
    using dataset.view(), e.g. segmentations or detections
    :type label_field: str
    :param dataset_type: The format of the exported dataset. Default value
    is COCODetectionDataset
    :type dataset_type: fo.types
    """
    dataset.untag_samples(dataset.distinct("tags"))

    dataset.export(
        export_dir,
        dataset_type=dataset_type,
        label_field=label_field,
        classes=dataset.default_classes,
        tolerance=0.02,
    )
    log.info(
        f"Data has been successfully exported as f{DEFAULT_DATASET_TYPE.__name__} to {export_dir}."
    )


def augment(dataset):
    session = fo.launch_app(dataset)
    # Negative number means session will be opened indefinitely
    session.wait(-1)


def split(images_directory_path, annotations_path, output_dir):
    """Convert COCO dataset to YOLO dataset and split the dataset into train,
    test, and validation sets. Recommended ratio of train:test:val is 7:2:1.

    :param images_directory_path: Path to images of COCO dataset
    (i.e. /home/kai/projects/lamp_detection/test_data/images)
    :type images_directory_path: str
    :param annotations_path: Path to labels.json of COCO dataset
    (i.e. /home/kai/projects/lamp_detection/test_data/labels.json)
    :type annotations_path: str
    :param output_dir: Path to output directory for exported dataset
    :type output_dir: str


    :return: Path to output directory
    :rtype: str
    """
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=images_directory_path,
        annotations_path=annotations_path,
        force_masks=True,
    )

    train, test = ds.split(split_ratio=0.7, random_state=42, shuffle=True)

    test, val = test.split(split_ratio=0.6, random_state=32, shuffle=True)

    train.as_yolo(
        images_directory_path=os.path.join(output_dir, "train/images"),
        annotations_directory_path=os.path.join(output_dir, "train/labels"),
        data_yaml_path=os.path.join(output_dir, "dataset.yml"),
    )

    test.as_yolo(
        images_directory_path=os.path.join(output_dir, "test/images"),
        annotations_directory_path=os.path.join(output_dir, "test/labels"),
    )

    val.as_yolo(
        images_directory_path=os.path.join(output_dir, "val/images"),
        annotations_directory_path=os.path.join(output_dir, "val/labels"),
    )
    log.info(f"Dataset converted from COCO to YOLO format at {output_dir}")


def delete(dataset_name):
    fo.delete_dataset(dataset_name)
    log.info(f"Dataset named {dataset_name} successfully deleted.")
