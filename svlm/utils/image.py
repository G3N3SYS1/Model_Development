import logging
from typing import Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


def isbright(img: np.ndarray, contour: np.ndarray, label: str) -> bool:
    assert contour.size != 0, "Contour array is empty"
    assert img.size != 0, "Image array is empty"

    b_mask = np.zeros(img.shape[:2], np.uint8)
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    b_mask = np.bool_(b_mask)
    mean = np.mean(img[b_mask])
    log.debug(f"Mean pixel value in segmented area: {str(mean)}")

    return bool(mean > 60 if label == "LIS" else mean > 200)


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
