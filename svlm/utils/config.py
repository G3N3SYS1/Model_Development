from omegaconf import OmegaConf


class Dataset:
    def __init__(
        self, data_path, labels_path, name, classes, label_field, export_dir
    ) -> None:
        self.data_path = data_path
        self.labels_path = labels_path
        self.name = name
        self.classes: list[str] = classes
        self.label_field = label_field
        self.export_dir = export_dir


class Params:
    def __init__(self, imgsz, epochs, batch, conf) -> None:
        self.imgsz = imgsz
        self.epochs = epochs
        self.batch = batch
        self.conf = conf


class Model:
    def __init__(self, path, source, params) -> None:
        self.path = path
        self.source = source
        self.params: Params = params


class Config:
    def __init__(self, dataset, model) -> None:
        self.dataset: Dataset = dataset
        self.model: Model = model


conf: Config | None = None


def load_config(path):
    global conf
    conf = OmegaConf.structured(OmegaConf.load(path))

    return conf
