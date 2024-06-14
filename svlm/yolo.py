import logging
import os
from pathlib import Path

import cv2
import extract
import numpy as np
import utils.image as image
from ultralytics import YOLO
import math

log = logging.getLogger(__name__)


def train(model_path, dataset, imgsz, epochs, batch):
    # yolo segment train model=yolov8n-seg.pt data=test_data/dataset.yaml epochs=60 imgsz=640 batch=16
    model = YOLO(model_path)
    return model.train(data=dataset, imgsz=imgsz, epochs=epochs, batch=batch, device=0)


def predict(vehicle_model_path, lamp_model_path, output_dir, source, conf, stream=True):
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

    # lamps = {
    #     "RVLL": False,
    #     "FL2": False,
    #     "HL1": False,
    #     "HL2": False,
    #     "LIF": False,
    #     "LIS": False,
    #     "RIF": False,
    #     "DRL": False,
    # }
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
    
    current_id = 0
    prev_id = 0
    trigger_centroid = (1400,700)
    tracking = False
    prev_centroid = {
        "centroid": trigger_centroid,
        "in_bbox" : False
        }

    
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        crop_frame = frame[0:int(height), 0+width_offset:int(width)]

        results = vehicle_model.predict(source=crop_frame, conf=conf, stream=stream, verbose=False)
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cv2.circle(frame, trigger_centroid, 5, (0,0,255), -1)

        
        for r in results:
            cv2.line(frame,(width_offset,0), (width_offset,int(height)), (255, 0, 0), 5)
            if r.boxes is not None:
                try:
                    # current_id = r.boxes.id.numpy().astype(np.int32)[0]
                    # if current_id > prev_id:
                    #     prev_id = current_id
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
                # for box in r.boxes:
                if len(r.boxes):
                    for box in r.boxes:
                        current_in_bbox = in_box(trigger_centroid, box.xyxy.cpu().squeeze().numpy().astype(np.int32))
                        x1, y1, x2, y2 = box.xyxy.cpu().squeeze().numpy().astype(np.int32)
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1 += width_offset
                        x2 += width_offset
                        cX = int((x1+x2)/2)
                        cY = int((y1+y2)/2)


                        if  current_in_bbox:
                            if tracking:
                                prev_centroid["centroid"] = (cX,cY)
                            else:
                                tracking = True
                                prev_centroid["centroid"] = (cX,cY)
                                print("Start tracking..")
                                # print(f"cY: {cY} __ prev y: {prev_centroid['centroid'][1]}")
                        else:
                            # print(f"trigger cen: {trigger_centroid[1]} __ prev y: {prev_centroid['centroid'][1]}")
                            if tracking and trigger_centroid[1] < prev_centroid["centroid"][1]:
                                tracking = False
                                print("Reset")


                        # if tracking:
                        #     print("tracking")
                        #     if cY > trigger_centroid[1]:
                        #         print("RESET1")
                        # else:
                        #     if not prev_centroid["in_bbox"]: # if prev centroid not in bbox
                        #         # if current centroid not inbbox and current cen Y more than prev cen Y by 150
                        #         if not current_in_bbox and\
                        #         more_than_distance(cY,prev_centroid["centroid"][1],150) :
                        #             print("RESET")
                        #             tracking = False
                        #         elif not tracking:
                        #             tracking = True
                        #             print("triggered")
                        #             prev_centroid["centroid"] = (cX,cY)

                            
                        # prev_centroid["in_bbox"] = current_in_bbox
                    
                    # id = box.id.numpy().squeeze().astype(np.int32)
                    # x1, y1, x2, y2 = box.xyxy.cpu().squeeze().numpy().astype(np.int32)
                    # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # x1 += width_offset
                    # x2 += width_offset
                    # cX = int((x1+x2)/2)
                    # cY = int((y1+y2)/2)
                    cv2.putText(frame,str(tracking),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
                    cv2.circle(frame, (cX,cY), 5, (0,255,0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # print(f"{is_in_box(trigger_centroid,box.xyxy.cpu().squeeze().numpy().astype(np.int32))}")
            # cv2.imshow("Lamp", r.plot())
            # out.write(r.plot())

            crop = None
        
            # if sum(1 for v in lamps.values() if v) < 8:
            #     crop = extract.segmentation_as_image(r, output_dir)
            # else: crop = None
            
            # crop = extract.segmentation_as_image(r, output_dir, frame_number)
            # out.write(crop) if crop is not None else out.write(frame)
            # cv2.imshow("Lamp", crop) if crop is not None else cv2.imshow("Lamp",frame)

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
                    img = image.pad(lr.plot().copy(), (width, height))

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

def calculate_distance(point1, point2):
    """Calculate distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def in_box(centroid, box):
    """Check if a centroid is inside a bounding box."""
    x_centroid, y_centroid = centroid
    x1, y1, x2, y2 = box
    return x1 <= x_centroid <= x2 and y1 <= y_centroid <= y2

def more_than_distance(y1, y2,dist):
    """Calculate distance between two points."""
    
    return abs(y1-y2)>dist