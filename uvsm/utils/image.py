import logging
from typing import Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)



def isbright(img: np.ndarray, contour: np.ndarray, label: str, mlt: dict) -> bool:
    """Check if the mean pixel value inside the contour area of the image is brighter 
    than a threshold.

    - Uses the provided contour to create a binary mask and calculates the mean
      pixel value within this mask.
    - Compares the mean pixel value against a threshold specified in the `mlt`
      dictionary based on the `label`.

    :param img: Image frame in the form of a NumPy array.
    :type img: np.ndarray
    :param contour: Contour points defining the region of interest.
    :type contour: np.ndarray
    :param label: Label indicating the type of area (e.g., "LIR", "default").
    :type label: str
    :param mlt: Dictionary containing thresholds for different labels.
    :type mlt: dict

    :return: True if the mean pixel value inside the contour is greater than the 
    threshold for the label, False otherwise.
    :rtype: bool
    """
    assert contour.size != 0, "Contour array is empty"
    assert img.size != 0, "Image array is empty"

    b_mask = np.zeros(img.shape[:2], np.uint8)
    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    b_mask = np.bool_(b_mask)
    mean = np.mean(img[b_mask])
    # if "LIR" in label:
    log.debug(f"Mean pixel value in segmented area: {str(mean)}")

    try:
        return bool(mean > mlt[label])
    except KeyError:
        return bool(mean > mlt["default"])


def pad(img: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """Resize and center-align an image within a black frame of specified resolution.

    - Resizes the input image to fit within the specified resolution while maintaining
      aspect ratio.
    - Places the resized image in the center of a black frame of the specified resolution.

    :param img: Image frame in the form of a NumPy array.
    :type img: np.ndarray
    :param resolution_wh: Desired width and height of the padded frame.
    :type resolution_wh: Tuple[int, int]

    :return: Padded image with the input image centered within a black frame of 
    specified resolution.
    :rtype: np.ndarray
    """
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
    """Crop the image using the bounding box coordinates and the contour to create a mask.

    - Creates a binary mask using the contour and applies it to the input image to extract
      the region of interest.
    - Uses the bounding box coordinates to define the exact area to crop from the image.

    :param img: Image frame in the form of a NumPy array.
    :type img: np.ndarray
    :param contour: Contour points defining the region of interest.
    :type contour: np.ndarray
    :param bbox: Bounding box coordinates (x1, y1, x2, y2) defining the area to crop.
    :type bbox: np.ndarray

    :return: Cropped image frame segment defined by the bounding box and contour.
    :rtype: np.ndarray
    """
    assert contour.size != 0, "Contour array is empty"
    assert img.size != 0, "Image array is empty"

    b_mask = np.zeros(img.shape[:2], np.uint8)

    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, img)

    x1, y1, x2, y2 = bbox
    return isolated[y1:y2, x1:x2]

def draw_bbox(img: np.ndarray, bbox: np.ndarray, isfrontlamp: bool, key: str = None) -> np.ndarray:
    """Draw a bounding box or a circle with the centroid on the image.

    - Draws either a rectangle with the given bounding box coordinates or a circle
      centered at the centroid of the bounding box.
    - Uses different colors for different keys based on the front or rear lamp view.

    :param img: Image frame in the form of a NumPy array.
    :type img: np.ndarray
    :param bbox: Bounding box coordinates (x1, y1, x2, y2) defining the area to draw.
    :type bbox: np.ndarray
    :param isfrontlamp: Flag indicating whether the view is from the front or rear
      of the vehicle.
    :type isfrontlamp: bool
    :param key: Optional key for selecting color (optional).
    :type key: str, optional

    :return: Image frame with the bounding box or circle drawn.
    :rtype: np.ndarray
    """
    colors = get_lamp_colour(isfrontlamp)
    x1, y1, x2, y2 = bbox
    
    cX = int((x1 + x2) / 2)
    cY = int((y1 + y2) / 2)
    
    if key:
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[key], 2)
    else:
        cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img

def draw_texts(img: np.ndarray, key: str, value: bool, point:tuple, isfrontlamp: bool) -> np.ndarray:
    """Draw text on the image indicating detection results or status.

    - Draws text labels and corresponding results (PASS/FAIL or Detected/Undetected)
      at the specified point on the image.
    - Uses different colors for different keys based on the front or rear lamp view.

    :param img: Image frame in the form of a NumPy array.
    :type img: np.ndarray
    :param key: Key or label for the text to be drawn.
    :type key: str
    :param value: Boolean value representing detection status or result
      (True for PASS/Detected, False for FAIL/Undetected).
    :type value: bool
    :param point: Coordinates (x, y) indicating where to draw the text on the image.
    :type point: tuple
    :param isfrontlamp: Flag indicating whether the view is from the front or rear of the vehicle.
    :type isfrontlamp: bool

    :return: Image frame with the text drawn.
    :rtype: np.ndarray
    """
    x, y = point
    colors = get_lamp_colour(isfrontlamp)
    cv2.putText(img, f"{key}: ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[key], 2)
    text_size, _ = cv2.getTextSize(f"{key}: ", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    if key in ["RFLL", "RFLR"]:
        display_res = f"{'Detected' if value else 'Undetected'}"
    else: 
        display_res = f"{'PASS' if value else 'FAIL'}"
    cv2.putText(
        img, 
        display_res, 
        (x + text_size[0], y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0) if value else (0, 0, 255), 2
    )
    
    return img

def draw_conf(img: np.ndarray, key: str, value: float, point: tuple, isfrontlamp: bool) -> np.ndarray:
    """Draw confidence value or score as text on the image.

    - Draws the confidence value or score at the specified point on the image for the given key.
    - Uses different colors for different keys based on the front or rear lamp view.

    :param img: Image frame in the form of a NumPy array.
    :type img: np.ndarray
    :param key: Key or label for the confidence value to be drawn.
    :type key: str
    :param value: Confidence value or score to be displayed.
    :type value: float
    :param point: Coordinates (x, y) indicating where to draw the text on the image.
    :type point: tuple
    :param isfrontlamp: Flag indicating whether the view is from the front or rear of the vehicle.
    :type isfrontlamp: bool

    :return: Image frame with the confidence value drawn.
    :rtype: np.ndarray
    """
    x, y = point
    colors = get_lamp_colour(isfrontlamp)
    cv2.putText(img, f"{key}: ", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[key], 2)
    text_size, _ = cv2.getTextSize(f"{key}: ", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(
        img, 
        str(value), 
        (x + text_size[0], y), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255) , 2
    )
    
    return img


def get_lamp_colour(isfrontlamp: bool = True)-> dict:
    """Get dictionary of colors for different lamp keys based on the front or rear view.

    - Returns a dictionary mapping lamp keys to their respective BGR color values based
      on whether the view is from the front or rear of the vehicle.

    :param isfrontlamp: Flag indicating whether the view is from the front or rear of the
      vehicle (default is True for front).
    :type isfrontlamp: bool

    :return: Dictionary mapping lamp keys to BGR color tuples.
    :rtype: dict
    """
    if isfrontlamp:
        return {
            "FLR": (0, 0, 255),     # Red
            "FLL": (0, 128, 255),   # Orange
            "HLL": (0, 255, 255),   # Yellow
            "HLR": (0, 255, 0),     # Green
            "LIF": (255, 255, 0),   # Light blue
            "RIS": (255, 165, 0),   # Not used
            "RIF": (208, 118, 0),   # Dark blue
            "DRL": (204, 0, 102),    # Purple-Blue
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