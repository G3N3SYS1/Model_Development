from ultralytics import YOLO


def train(model_path, source, imgsz, epochs, batch):
    # yolo segment train model=yolov8n-seg.pt data=test_data/dataset.yaml epochs=60 imgsz=640 batch=16
    model = YOLO(model_path)
    return model.train(data=source, imgsz=imgsz, epochs=epochs, batch=batch, device=0)


def predict(model_path, source, conf):
    model = YOLO(model_path)
    return model.predict(source=source, save=True, save_txt=True, conf=conf)


def val(model_path, source):
    model = YOLO(model_path)
    return model.val(data=source, split="val")
