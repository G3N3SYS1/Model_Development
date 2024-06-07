import logging
import os
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
    
    cv2.namedWindow("Lamp", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lamp", 900, 640)
    width_offset = 750

    lamps = {
        "FL1": False,
        "FL2": False,
        "HL1": False,
        "HL2": False,
        "LIF": False,
        "LIS": False,
        "RIF": False,
        "DRL": False,
    }
    
    current_id = 0
    prev_id = 0
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        crop_frame = frame[0:int(height), 0+width_offset:int(width)]

        results = vehicle_model.track(source=crop_frame, conf=conf, stream=stream, persist=True, verbose=False, tracker= "custom_tracker.yaml")

    

        
        for r in results:
            cv2.line(frame,(width_offset,0), (width_offset,int(height)), (255, 0, 0), 5)

            if r.boxes.id is not None:
                try:
                    current_id = r.boxes.id.numpy().astype(np.int32)[0]
                    if current_id > prev_id:
                        prev_id = current_id
                        #    reset
                        lamps = {
                            "FL1": False,
                            "FL2": False,
                            "HL1": False,
                            "HL2": False,
                            "LIF": False,
                            "LIS": False,
                            "RIF": False,
                            "DRL": False,
                        }
                except Exception as e:
                    print(e)
                    pass
            #     for box in r.boxes:
            #         id = box.id.numpy().squeeze().astype(np.int32)
            #         x1, y1, x2, y2 = box.xyxy.cpu().squeeze().numpy().astype(np.int32)
            #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #         x1 += width_offset
            #         x2 += width_offset
            #         cX = int((x1+x2)/2)
            #         cY = int((y1+y2)/2)
            #         cv2.putText(frame,str(id),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.imshow("Lamp", frame)
            # out.write(frame)

            # crop = None
        
            # if sum(1 for v in lamps.values() if v) < 8:
            #     crop = extract.segmentation_as_image(r, output_dir)
            # else: crop = None

            crop = extract.segmentation_as_image(r, output_dir)

            if crop is None:
                out.write(frame)
                cv2.imshow("Lamp", frame)
            else:
                lamp_res = lamp_model.predict(
                    source=crop, save=True, save_txt=True, stream=True, verbose=False
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

                        if contour.size == 0 or bbox.size == 0:
                            log.info(
                                "No valid bbox or segmentation masks could be parsed from the results."
                            )
                            continue

                        # Obtained a cropped image of a specific lamp instead of whole car
                        iso_crop = image.crop(img, contour, bbox)

                        cv2.imshow("Lamp View", iso_crop)

                        label = label.split("_", 1)[0]

                        # Calculate mean pixel value of cropped image of lamp
                        if not lamps[label]:
                            lamps[label] = image.isbright(img, contour, label)

                        log.info(f"{label} is lit up: {lamps[label]}")

                    # Pad cropped image to original resolution. To write to video output,
                    # frame must have same resolution as the video
                    h, w, _ = r.orig_img.shape
                    img = image.pad(lr.orig_img.copy(), (width, height))

                    y = 300

                    cv2.putText(img,str(f"ID:{current_id}"),(1600,y),cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
                    
                    for key in lamps.keys():
                        y += 50
                        cv2.putText(
                            img,
                            f"{key} is lit: {lamps[key]}",
                            (1600, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0) if lamps[key] else (0, 0, 255)  ,
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.imshow("Lamp", img)
                    out.write(img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()
    out.release()


def val(model_path, source):
    model = YOLO(model_path)
    return model.val(data=source, split="val")
