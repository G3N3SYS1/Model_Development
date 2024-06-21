import logging
import os
from pathlib import Path

import cv2
import extract
import numpy as np
import utils.image as image
from ultralytics import YOLO
from utils.lamp import track

log = logging.getLogger(__name__)


def train(model_path, dataset, imgsz, epochs, batch):
    # yolo segment train model=yolov8n-seg.pt data=test_data/dataset.yaml epochs=60 imgsz=640 batch=16
    model = YOLO(model_path)
    return model.train(data=dataset, imgsz=imgsz, epochs=epochs, batch=batch, device=0)


def predict(vehicle_model_path, lamp_model_path, output_dir, source,
             vehicle_conf, lamp_conf, d_angle, ref_pt, mean_light_thresh, stream=True):
    vehicle_model = YOLO(vehicle_model_path)
    lamp_model = YOLO(lamp_model_path)

    filename = Path(source).stem
    output_dir = os.path.join(output_dir, f"{filename}")
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
    width_offset = 0

    isfrontlamp = True if d_angle == 'd1' or d_angle == 'd2' else False
    lamps = resetlamp(isfrontlamp)
    istracking = False
    ref_pt = tuple(ref_pt[d_angle].values())    
    prev_pt = ref_pt
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        crop_frame = frame[0:int(height), 0+width_offset:int(width)]
        orig_frame = np.copy(frame)
        results = vehicle_model.predict(source=crop_frame, conf=vehicle_conf, stream=stream, verbose=False)
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cv2.circle(frame, ref_pt, 5, (0,0,255), -1)
        cv2.line(frame,(width_offset,0), (width_offset,int(height)), (255, 0, 0), 5)
        crop = None
        
        for r in results:
            for c in r:
                x1, y1, x2, y2 = c.boxes.xyxy.cpu().squeeze().numpy().astype(np.int32)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # x1 += width_offset
                # x2 += width_offset
                cX = int((x1+x2)/2)
                cY = int((y1+y2)/2)
                cv2.circle(frame, (cX,cY), 5, (0,255,0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                box_xyxy = c.boxes.xyxy.cpu().squeeze().numpy().astype(np.int32)
                isreset, istracking, prev_pt, tocrop = track(
                    box_xyxy, 
                    ref_pt, 
                    istracking, 
                    prev_pt,
                    isfrontlamp,
                )

                if tocrop:
                    if isreset:
                        lamps = resetlamp(isfrontlamp)
                        img_name = Path(r.path).stem
                    else:
                        if sum(1 for v in lamps.values() if v) < len(lamps):
                            crop = extract.segmentation_as_image(c, output_dir, frame_number, orig_frame, img_name)
                            break
                        else: 
                            # Get the size of the text
                            text_size, _ = cv2.getTextSize("PASS", cv2.FONT_HERSHEY_SIMPLEX, 5, 5)
                            # Calculate the position to start writing text (centered in the bbox)
                            text_x = int((x1 + x2 - text_size[0]) / 2)
                            text_y = int((y1+ y2 + text_size[1]) / 2)
                            # Write the text on the image
                            cv2.putText(frame, "PASS", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
                else:
                    crop = None

            if crop is None:
                out.write(frame)
                cv2.imshow("Lamp", frame)
            else:
                lamp_res = lamp_model.predict(
                    source=crop, save=False, save_txt=False, stream=True, conf=lamp_conf, verbose=False
                )

                for lr in lamp_res:
                    img = lr.orig_img.copy()
                    for c in lr:
                        if c.boxes is None or c.masks is None:
                            log.info(
                                "No bbox or segmentation masks found in prediction results."
                            )
                            continue
                        label = c.names[c.boxes.cls.tolist().pop()].split("_", 1)[0]
                        
                        if "RFLR" in label or "RFLL" in label:
                            lamps[label] = True
                            continue

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

                        cv2.imshow("Lamp View", iso_crop) if "LIS" in label else None


                        # Calculate mean pixel value of cropped image of lamp
                        if not lamps[label]:
                            lamps[label] = image.isbright(img, contour, label, mean_light_thresh)

                        log.info(f"{label} is lit up: {lamps[label]}")

                    # Pad cropped image to original resolution. To write to video output,
                    # frame must have same resolution as the video
                    h, w, _ = r.orig_img.shape
                    img = image.pad(lr.plot(), (width, height))

                    y = 300

                    
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


def resetlamp(isfrontlamp: bool):
    print("Reset lamp status..")
    if isfrontlamp:
        lamps = {
            "FLR": False,
            "FLL": False,
            "HLL": False,
            "HLR": False,
            "LIF": False,
            "RIS": False,
            "RIF": False,
            "DRL": False,

        }
    else:
        lamps = {
            "SLL": False,
            "SLR": False,
            "CSL": False,
            "RVLL": False,
            "RVLR": False,
            "LIR": False,
            "LIS": False,
            "RIR": False,
            "RLL": False,
            "RLR": False,
            "RFLL": False,
            "RFLR": False
        }
    return lamps

def process_C(c, lamps):
    if c.boxes is None or c.masks is None:
        log.info(
            "No bbox or segmentation masks found in prediction results."
        )
        return None
    label = c.names[c.boxes.cls.tolist().pop()]
    print(f"label: {label}")


    contour = (
        np.asarray(c.masks.xy.pop()).astype(np.int32).reshape(-1, 1, 2)
    )

    bbox = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)

    if contour.size == 0 or bbox.size == 0:
        log.info(
            "No valid bbox or segmentation masks could be parsed from the results."
        )
        return None
    img = c.orig_img.copy()
    # Obtained a cropped image of a specific lamp instead of whole car
    iso_crop = image.crop(img, contour, bbox)

    cv2.imshow("Lamp View", iso_crop) if "RLR" in label else None

    label = label.split("_", 1)[0]

    # Calculate mean pixel value of cropped image of lamp
    if not lamps[label]:
        lamps[label] = image.isbright(img, contour, label)

    log.info(f"{label} is lit up: {lamps[label]}")
    return lamps[label]