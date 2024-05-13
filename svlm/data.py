import os

import fiftyone as fo
import supervision as sv
from fiftyone.types import COCODetectionDataset

DEFAULT_DATASET_TYPE = COCODetectionDataset


def load(
    name,
    data_path,
    labels_path,
    dataset_type=DEFAULT_DATASET_TYPE,
):

    existing_datasets = fo.list_datasets()

    if name not in existing_datasets:
        print(f"Creating a new dataset, {name}...")
        dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            data_path=data_path,
            labels_path=labels_path,
            name=name,
        )
    else:
        print("Loading dataset...")
        dataset = fo.load_dataset(name)

    print(dataset.view())
    return dataset


def export(
    dataset,
    export_dir: str,
    classes: list[str],
    label_field,
    dataset_type=DEFAULT_DATASET_TYPE,
):
    dataset.untag_samples(dataset.distinct("tags"))

    dataset.export(
        export_dir,
        dataset_type=dataset_type,
        label_field=label_field,
        classes=classes,
    )
    print(f"Data has been successfully exported to {export_dir}.")


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
    print(f"Dataset converted from COCO to YOLO format at {output_dir}")


def delete(dataset_name):
    fo.delete_dataset(dataset_name)
    print(f"Dataset named {dataset_name} successfully deleted.")
