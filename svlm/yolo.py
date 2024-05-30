import logging
import os
from pathlib import Path

import cv2
import extract
import numpy as np
from ultralytics import YOLO

log = logging.getLogger(__name__)


def train(model_path, dataset, imgsz, epochs, batch):
    # yolo segment train model=yolov8n-seg.pt data=test_data/dataset.yaml epochs=60 imgsz=640 batch=16
    model = YOLO(model_path)
    return model.train(data=dataset, imgsz=imgsz, epochs=epochs, batch=batch, device=0)


def predict(vehicle_model_path, lamp_model_path, output_dir, source, conf, stream=True):
    vehicle_model = YOLO(vehicle_model_path)
    lamp_model = YOLO(lamp_model_path)

    filename = Path(source).stem
    output_path = os.path.join(output_dir, f"{filename}_annotated.mp4")
    os.makedirs(output_dir, exist_ok=True)
    log.info(output_path)

    cap = cv2.VideoCapture(source)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        int(fps),
        (int(width), int(height)),
    )  # must be same size as input

    results = vehicle_model.predict(source=source, conf=conf, stream=stream)

    for r in results:
        crop = extract.segmentation_as_image(r, output_dir)

        if crop is None:
            out.write(r.plot())
            cv2.imshow("Lamp", r.plot())
            cv2.waitKey(1)
        else:
            lamp_res = lamp_model.predict(
                source=crop, save=True, save_txt=True, stream=True
            )
            for lr in lamp_res:
                # Pad cropped image to original resolution. To write to video output,
                # frame must have same resolution as the video
                img = lr.plot().copy()
                old_h, old_w, channels = img.shape
                result = np.full(
                    (int(height), int(width), channels), (0, 0, 0), dtype=np.uint8
                )

                x_center = (int(width) - int(old_w)) / 2
                y_center = (int(height) - int(old_h)) / 2

                # copy img image into center of result image
                result[
                    int(y_center) : int(y_center + old_h),
                    int(x_center) : int(x_center + old_w),
                ] = img

                out.write(result)
                cv2.imshow("Lamp", result)
                cv2.waitKey(1)
    out.release()


def val(model_path, source):
    model = YOLO(model_path)
    return model.val(data=source, split="val")
