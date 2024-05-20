import logging

import click
import data
import extract
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
    dataset = data.load(
        conf.dataset.name, conf.dataset.data_path, conf.dataset.labels_path
    )
    data.augment(dataset)


@cli.command()
@click.pass_obj
def export(conf: Config):
    dataset = data.load(
        conf.dataset.name, conf.dataset.data_path, conf.dataset.labels_path
    )
    print("Exporting dataset to COCO format")
    data.export(dataset, conf.dataset.export_dir, conf.dataset.label_field)
    print(f"Dataset has been exported to {conf.dataset.export_dir}.")


@cli.command()
@click.pass_obj
def train(conf: Config):
    yolo.train(
        conf.model.path,
        conf.model.source,
        conf.model.params.imgsz,
        conf.model.params.epochs,
        conf.model.params.batch,
    )


@cli.command
@click.argument("images")
@click.argument("labels")
@click.argument("output")
def split(images, labels, output):
    data.split(images, labels, output)


@cli.command()
@click.argument("source")
@click.option("-o", "--output", type=click.Path(exists=False))
@click.pass_obj
def predict(conf: Config, source, output):
    results = yolo.predict(conf.model.path, source, conf.model.params.conf)
    for r in results:
        extract.segmentation_as_image(r, output)


@cli.command()
@click.argument("name")
def delete(name):
    data.delete(name)


if __name__ == "__main__":
    init()
    cli()
