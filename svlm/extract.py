import logging
import os

import cv2
import numpy as np
import utils.image as image
from ultralytics.engine.results import Results

log = logging.getLogger(__name__)


def segmentation_as_image(c: Results, output_dir, frame_number, img, img_name):
    """Extract segmentation mask from prediction results in the form of a black/white binary mask which is then used
    to crop the segmented object from the original image. The cropped image will be exported as a png file with transparent
    background to the specified path stored in output_dir.

    https://docs.ultralytics.com/guides/isolating-segmentation-objects/#isolate-with-transparent-pixels-sub-options

    :param res: Prediction results obtained from YOLOv8
    :type res: Results
    :param output_dir: Path to directory to store the cropped images
    :type output_dir: str
    """
    if c.boxes is None or c.masks is None:
        log.info("No bbox or segmentation masks found in prediction results.")
        return

    label = c.names[c.boxes.cls.tolist().pop()]
    # Create contour mask
    contour = np.asarray(c.masks.xy.pop()).astype(np.int32).reshape(-1, 1, 2)

    # isolated = np.dstack([img, b_mask]) # Transparent background by adding alpha channel

    bbox = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
    iso_crop = image.crop(img, contour, bbox)

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")

    os.makedirs(output_dir, exist_ok=True)

    # export_name = f"{img_name}_{label}_{int(frame_number)}.png"
    export_name = f"{img_name}_{label}.png"
    output_dir = os.path.join(output_dir, export_name)

    # When cv2.imshow, background still apparent. Only when saved as png will be transparent
    _ = cv2.imwrite(output_dir, iso_crop)
    log.info(f"Image successfully cropped and exported to {output_dir}")
    return iso_crop
