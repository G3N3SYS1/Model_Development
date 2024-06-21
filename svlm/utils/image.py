import logging
from typing import Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)



def isbright(img: np.ndarray, contour: np.ndarray, label: str, mlt: dict) -> bool:
    assert contour.size != 0, "Contour array is empty"
    assert img.size != 0, "Image array is empty"

    b_mask = np.zeros(img.shape[:2], np.uint8)
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    b_mask = np.bool_(b_mask)
    mean = np.mean(img[b_mask])
    if "LIR" in label:
        log.debug(f"Mean pixel value in segmented area: {str(mean)}")

    try:
        return bool(mean > mlt[label])
    except KeyError:
        return bool(mean > mlt["default"])


def pad(img: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    assert img.size != 0, "Image array is empty"
    width, height = resolution_wh

    old_h, old_w, channels = img.shape
    # create a black frame of width x height
    frame = np.full((int(height), int(width), channels), (0, 0, 0), dtype=np.uint8)

    x_center = (int(width) - int(old_w)) / 2
    y_center = (int(height) - int(old_h)) / 2

    # copy img image into center of frame
    frame[
        int(y_center) : int(y_center + old_h),
        int(x_center) : int(x_center + old_w),
    ] = img

    return frame


def crop(img: np.ndarray, contour: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    assert contour.size != 0, "Contour array is empty"
    assert img.size != 0, "Image array is empty"

    b_mask = np.zeros(img.shape[:2], np.uint8)

    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, img)

    x1, y1, x2, y2 = bbox
    return isolated[y1:y2, x1:x2]

def draw_bbox(img: np.ndarray, bbox: np.ndarray, isfrontlamp, key: str = None):
    colors = get_lamp_colour(isfrontlamp)
    x1, y1, x2, y2 = bbox
    
    cX = int((x1 + x2) / 2)
    cY = int((y1 + y2) / 2)
    
    if key:
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[key], 3)
    else:
        cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img

def draw_texts(img: np.ndarray, key: str, value, x, y, isfrontlamp):
    colors = get_lamp_colour(isfrontlamp)
    cv2.putText(img, f"{key}: ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[key], 2)
    text_size, _ = cv2.getTextSize(f"{key}: ", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    if key in ["RFLL", "RFLR"]:
        display_res = f"{"Detected" if value else "Undetected"}"
    else: 
        display_res = f"{"PASS" if value else "FAIL"}"
    cv2.putText(
        img, 
        display_res, 
        (x + text_size[0], y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0) if value else (0, 0, 255), 2
    )
    
    return img


def get_lamp_colour(isfrontlamp: bool = True):
    if isfrontlamp:
        return {
            "FLR": (0, 0, 255),     # Red for FLR
            "FLL": (0, 128, 255),   # Orange for FLL
            "HLL": (0, 255, 255),   # Yellow for HLL
            "HLR": (0, 255, 0),     # Green for HLR
            "LIF": (255, 255, 0),   # Cyan for LIF
            "RIS": (255, 165, 0),   # Orange-Yellow for RIS
            "RIF": (255, 102, 0),   # Orange-Red for RIF
            "DRL": (255, 0, 0)      # Bright Red for DRL
        }
    else:
        return {
            "SLL": (0, 118, 208),    # Lighter Red
            "SLR": (208, 118, 0),    # Blue
            "CSL": (170, 210, 255),  # Beige
            "RVLL": (0, 255, 255),  # Yellow-Orange
            "RVLR": (0, 204, 102),  # Green
            "LIR": (0, 102, 255),   # Orange-Red
            "LIS": (235, 30, 200),   # Pink
            "RIR": (255, 178, 102),  # Light-Blue
            "RLL": (102, 0, 204),    # Magenta
            "RLR": (204, 0, 102),    # Purple-Blue
            "RFLL": (48, 117, 213),  # Cyan
            "RFLR": (200, 213, 48),  # Turqoise
        }