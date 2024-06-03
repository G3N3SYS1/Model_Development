import logging

import click
import yolo
from utils.config import Config, load_config


def init():
    """Configure the root logger of the application. The logging configuration will
    be inherited by the loggers in all modules of the application.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    logger.addHandler(ch)


@click.group
@click.option("-c", "--config", default="./config.yml", required=True)
@click.pass_context
def cli(ctx, config):
    ctx.obj = load_config(config)
    pass


@cli.command()
@click.pass_obj
def augment(conf: Config):
    import data

    dataset = data.load(
        conf.fiftyone.dataset_name,
        conf.fiftyone.augment.data_path,
        conf.fiftyone.augment.labels_path,
    )
    data.augment(dataset)


@cli.command()
@click.pass_obj
def export(conf: Config):
    import data

    dataset = data.load(
        conf.fiftyone.dataset_name,
        conf.fiftyone.augment.data_path,
        conf.fiftyone.augment.labels_path,
    )
    print("Exporting dataset to COCO format")
    data.export(
        dataset, conf.fiftyone.export.output_dir, conf.fiftyone.export.label_field
    )
    print(f"Dataset has been exported to {conf.fiftyone.export.output_dir}.")


@cli.command()
@click.pass_obj
def train(conf: Config):
    yolo.train(
        conf.train.base_model_path,
        conf.train.dataset,
        conf.train.params.imgsz,
        conf.train.params.epochs,
        conf.train.params.batch,
    )


@cli.command
@click.argument("images")
@click.argument("labels")
@click.argument("output")
def split(images, labels, output):
    import data

    data.split(images, labels, output)


@cli.command()
# @click.argument("source")
# @click.option("-o", "--output", "output_dir", type=click.Path(exists=False))
@click.pass_obj
def predict(conf: Config):
    yolo.predict(
        conf.predict.vehicle_model_path,
        conf.predict.lamp_model_path,
        conf.predict.output_dir,
        conf.predict.source,
        conf.predict.params.conf,
    )


@cli.command()
@click.argument("name")
def delete(name):
    import data

    data.delete(name)


if __name__ == "__main__":
    init()
    cli()
