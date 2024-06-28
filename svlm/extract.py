import logging
import os

import cv2
import numpy as np
import utils.image as image
from ultralytics.engine.results import Results

log = logging.getLogger(__name__)


def segmentation_as_image(res: Results, output_dir: str, frame_number: int, img: np.ndarray, img_name: str, save_all: bool = False):
    """Extract segmentation mask from prediction results in the form of a black/white binary mask which is then used
    to crop the segmented object from the original image. The cropped image will be exported as a png file with transparent
    background to the specified path stored in output_dir.

    https://docs.ultralytics.com/guides/isolating-segmentation-objects/#isolate-with-transparent-pixels-sub-options

    :param res: Prediction results object obtained from YOLOv8, containing bounding boxes and segmentation masks.
    :type res: Results
    :param output_dir: Path to the directory where the cropped images will be saved.
    :type output_dir: str
    :param frame_number: Frame number or identifier associated with the input image.
    :type frame_number: int
    :param img: Original input image from which the segmentation mask is extracted.
    :type img: np.ndarray
    :param img_name: Name or identifier of the input image.
    :type img_name: str
    :param save_all: Saving all frames to `output_dir`. Default `False`
    :type save_all: bool
    """
    if res.boxes is None or res.masks is None:
        log.info("No bbox or segmentation masks found in prediction results.")
        return

    label = res.names[res.boxes.cls.tolist().pop()]
    # Create contour mask
    contour = np.asarray(res.masks.xy.pop()).astype(np.int32).reshape(-1, 1, 2)

    # isolated = np.dstack([img, b_mask]) # Transparent background by adding alpha channel

    bbox = res.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
    iso_crop = image.crop(img, contour, bbox)

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")

    os.makedirs(output_dir, exist_ok=True)

    if save_all:
        export_name = f"{img_name}_{label}_{int(frame_number)}.png"
    else:
        export_name = f"{img_name}_{label}.png"
    output_dir = os.path.join(output_dir, export_name)

    # When cv2.imshow, background still apparent. Only when saved as png will be transparent
    _ = cv2.imwrite(output_dir, iso_crop)
    log.info(f"Image successfully cropped and exported to {output_dir}")
    return iso_crop
