from omegaconf import OmegaConf


class Fiftyone:
    def __init__(self, dataset_name, augment, export) -> None:
        self.dataset_name = dataset_name
        self.augment: Augment = augment
        self.export: Export = export


class Augment:
    def __init__(self, data_path, labels_path) -> None:
        self.data_path = data_path
        self.labels_path = labels_path


class Export:
    def __init__(self, label_field, output_dir) -> None:
        self.label_field = label_field
        self.output_dir = output_dir


class Params:
    def __init__(self, imgsz, epochs, batch, conf) -> None:
        self.imgsz = imgsz
        self.epochs = epochs
        self.batch = batch
        self.conf = conf


class Train:
    def __init__(self, base_model_path, dataset, params) -> None:
        self.base_model_path: str = base_model_path
        self.dataset: str = dataset
        self.params: Params = params


class Predict:
    def __init__(
        self, vehicle_model_path, lamp_model_path, source, output_dir, params
    ) -> None:
        self.vehicle_model_path: str = vehicle_model_path
        self.lamp_model_path: str = lamp_model_path
        self.source = source
        self.output_dir: str = output_dir
        self.params: Params = params


class Config:
    def __init__(self, fiftyone, train, predict) -> None:
        self.fiftyone: Fiftyone = fiftyone
        self.train: Train = train
        self.predict: Predict = predict


conf: Config | None = None


def load_config(path):
    global conf
    conf = OmegaConf.structured(OmegaConf.load(path))

    return conf
