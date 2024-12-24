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
    # yolo segment train model=yolov8n-seg.pt images=test_data/dataset.yaml epochs=60 imgsz=640 batch=16
    model = YOLO(model_path)
    return model.train(data=dataset, imgsz=imgsz, epochs=epochs, batch=batch, device=0)


def predict(model_path, output_dir, source):
     model = YOLO(model_path)
#     lamp_model = YOLO(lamp_model_path)
#
#     filename = Path(source).stem
#     output_dir = os.path.join(output_dir, f"{filename}")
#     output_path = os.path.join(output_dir, f"{filename}_annotated.mp4")
#     os.makedirs(output_dir, exist_ok=True)
#
#     cap = cv2.VideoCapture(source)
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     out = cv2.VideoWriter(
#         output_path,
#         cv2.VideoWriter.fourcc(*"mp4v"),
#         int(fps),
#         (int(width), int(height)),
#     )  # must be same size as input
#
#     cv2.namedWindow("Lamp", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Lamp", 900, 640)
#
#     # This is only use to filter out reflectors
#     test_lamp = {
#         "RFLL" : False,
#         "RFLR" : False
#     }
#     isfrontlamp = True if d_angle == 'd1' or d_angle == 'd2' else False
#     lamps = resetlamp(isfrontlamp)
#     istracking = False
#     ref_pt = tuple(ref_pt.values())
#     prev_pt = ref_pt
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#         orig_frame = np.copy(frame)
#         results = vehicle_model.predict(source=orig_frame, conf=vehicle_conf, stream=stream, verbose=False)
#         frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
#         cv2.circle(frame, ref_pt, 5, (0,0,255), -1)
#         crop = None
#
#         for r in results:
#             for c in r:
#                 bbox = c.boxes.xyxy.cpu().squeeze().numpy().astype(np.int32)
#                 frame = image.draw_bbox(frame, bbox, isfrontlamp)
#
#                 # track vehicle
#                 isreset, istracking, prev_pt, tocrop = track(
#                     bbox,
#                     ref_pt,
#                     istracking,
#                     prev_pt,
#                     isfrontlamp,
#                 )
#
#                 if tocrop:
#                     if isreset:
#                         lamps = resetlamp(isfrontlamp)
#                         img_name = Path(r.path).stem
#
#                         # This is only use to filter out reflectors
#                         test_lamp = {
#                             "RFLL" : False,
#                             "RFLR" : False
#                         }
#                     else:
#                         if sum(1 for v in lamps.values() if v[0]) < len(lamps):
#                             crop = extract.segmentation_as_image(
#                                 c,
#                                 output_dir,
#                                 frame_number,
#                                 orig_frame,
#                                 img_name
#                             )
#                             break
#                         elif istracking:
#                             # Get the size of the text
#                             text_size, _ = cv2.getTextSize("All Pass", cv2.FONT_HERSHEY_SIMPLEX, 5, 5)
#                             # Calculate the position to start writing text (centered in the bbox)
#                             text_x = int((bbox[0] + bbox[2] - text_size[0]) / 2)
#                             text_y = int((bbox[1] + bbox[3] + text_size[1]) / 2)
#                             # Write the text on the image
#                             cv2.putText(frame, "All Pass", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
#                 else:
#                     crop = None
#
#             if crop is None:
#                 out.write(frame)
#                 cv2.imshow("Lamp", frame)
#             else:
#                 lamp_res = lamp_model.predict(
#                     source=crop, save=False, save_txt=False, stream=True, conf=lamp_conf, verbose=False
#                 )
#
#                 for lr in lamp_res:
#                     img = lr.orig_img.copy()
#                     for c in lr:
#                         if c.boxes is None or c.masks is None:
#                             log.info(
#                                 "No bbox or segmentation masks found in prediction results."
#                             )
#                             continue
#                         label = c.names[c.boxes.cls.tolist().pop()].split("_", 1)[0]
#
#
#
#                         contour = (
#                             np.asarray(c.masks.xy.pop()).astype(np.int32).reshape(-1, 1, 2)
#                         )
#
#                         bbox = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
#
#                         if label == "RIS" or label == "DRL" and c.boxes.conf.item() < 0.7:
#                             continue
#                         img = image.draw_bbox(img, bbox, isfrontlamp, label)
#                         lamps[label][1] = float(round(c.boxes.conf.item(), 2))
#
#                         if "RFLR" in label or "RFLL" in label:
#                             test_lamp[label] = True
#                             continue
#
#                         if contour.size == 0 or bbox.size == 0:
#                             log.info(
#                                 "No valid bbox or segmentation masks could be parsed from the results."
#                             )
#                             continue
#
#                         # Obtained a cropped image of a specific lamp instead of whole car
#                         iso_crop = image.crop(lr.orig_img, contour, bbox)
#
#                         cv2.imshow("Lamp View", iso_crop)# if "FLL" in label else None
#
#
#                         # Calculate mean pixel value of cropped image of lamp
#                         if not lamps[label][0]:
#                             lamps[label][0] = image.isbright(lr.orig_img, contour, label, mean_light_thresh)
#
#                         log.info(f"{label} is lit up: {lamps[label][0]}")
#
#                     # Pad cropped image to original resolution. To write to video output,
#                     # frame must have same resolution as the video
#                     h, w, _ = r.orig_img.shape
#                     img = image.pad(img, (w, h))
#
#                     y = 300
#                     for key, value in lamps.items():
#                         y += 50
#                         if key =="RIS":
#                             continue
#                         img = image.draw_texts(
#                             img,
#                             key,
#                             test_lamp[key] if key in ["RFLL", "RFLR"] else value[0],
#                             (1600, y),
#                             isfrontlamp
#                         )
#                         img = image.draw_conf(img, key, value[1], (100, y), isfrontlamp)
#                 cv2.imshow("Lamp", img)
#                 out.write(img)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             exit()
#     out.release()


def val(model_path, source):
    model = YOLO(model_path)
    return model.val(data=source, split="val")


def resetlamp(isfrontlamp: bool):
    """Reset the status of lamps (on/off) and their detection confidence scores.

    This function resets the status of lamps and their associated detection confidence 
    scores based on whether the video feed is from the front view (D1, D2) or the back 
    view (D3, D4) of the vehicle.

    :param isfrontlamp: Flag indicating whether the video feed is from the front
     view (D1, D2).
    :type isfrontlamp: bool

    :return: A dictionary containing lamp keys mapped to lists 
    [on/off status, detection confidence score]. The initial status and score 
    are set based on the front or back view.
    :rtype: dict
    """
    log.info("Reset lamp status..")
    
    if isfrontlamp:
        return {
            "FLR": [False, 0],
            "FLL": [False, 0],
            "HLL": [False, 0],
            "HLR": [False, 0],
            "LIF": [False, 0],
            "RIF": [False, 0],
            "DRL": [False, 0],
            "RIS": [True, 0],
        }
    else:
        return {
            "RVLL": [False, 0],
            "LIR": [False, 0],
            "LIS": [False, 0],
            "RLL": [False, 0],
            "SLL": [False, 0],
            "RFLL": [True, 0],
            "CSL": [False, 0],
            "RVLR": [False, 0],
            "RIR": [False, 0],
            "RLR": [False, 0],
            "SLR": [False, 0],
            "RFLR": [True, 0],
        }
