import logging
import os
import tkinter
from pathlib import Path

import cv2
import extract
import numpy as np
import utils.image as image
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

    cv2.namedWindow("Lamp", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lamp", 900, 640)

    lamps = {
        "FL1": False,
        "FL2": False,
        "HL1": False,
        "HL2": False,
        "LIF": False,
        "LIS": False,
        "RIF": False,
    }

    for r in results:
        crop = extract.segmentation_as_image(r, output_dir)

        if crop is None:
            out.write(r.plot())
            cv2.imshow("Lamp", r.plot())
        else:
            lamp_res = lamp_model.predict(
                source=crop, save=True, save_txt=True, stream=True
            )
            for lr in lamp_res:

                img = lr.orig_img.copy()
                for c in lr:
                    if c.boxes is None or c.masks is None:
                        log.info(
                            "No bbox or segmentation masks found in prediction results."
                        )
                        continue
                    label = c.names[c.boxes.cls.tolist().pop()]

                    contour = (
                        np.asarray(c.masks.xy.pop()).astype(np.int32).reshape(-1, 1, 2)
                    )

                    bbox = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)

                    # Obtained a cropped image of a specific lamp instead of whole car
                    iso_crop = image.crop(img, contour, bbox)

                    cv2.imshow("New lamp", iso_crop)

                    label = label.split("_", 1)[0]

                    # Calculate mean pixel value of cropped image of lamp
                    lamps[label] = image.isbright(img, contour, label)

                    log.info(f"{label} is lit up: {lamps[label]}")

                # Pad cropped image to original resolution. To write to video output,
                # frame must have same resolution as the video
                h, w, _ = r.orig_img.shape
                img = image.pad(lr.plot().copy(), (w, h))

                out.write(img)
                y = 300
                for key in lamps.keys():
                    y += 50
                    cv2.putText(
                        img,
                        f"{key} is lit: {lamps[key]}",
                        (1600, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("Lamp", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()
    out.release()


def val(model_path, source):
    model = YOLO(model_path)
    return model.val(data=source, split="val")
