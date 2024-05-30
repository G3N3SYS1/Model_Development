import logging
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics.engine.results import Results

log = logging.getLogger(__name__)


def segmentation_as_image(res: Results, output_dir):
    """Extract segmentation mask from prediction results in the form of a black/white binary mask which is then used
    to crop the segmented object from the original image. The cropped image will be exported as a png file with transparent
    background to the specified path stored in output_dir.

    https://docs.ultralytics.com/guides/isolating-segmentation-objects/#isolate-with-transparent-pixels-sub-options

    :param res: Prediction results obtained from YOLOv8
    :type res: Results
    :param output_dir: Path to directory to store the cropped images
    :type output_dir: str
    """
    for r in res:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem

        for c in r:
            if c.boxes is None or c.masks is None:
                log.info("No bbox or segmentation masks found in prediction results.")
                return

            label = c.names[c.boxes.cls.tolist().pop()]
            b_mask = np.zeros(img.shape[:2], np.uint8)

            # Create contour mask
            contour = np.asarray(c.masks.xy.pop()).astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)

            # isolated = np.dstack([img, b_mask]) # Transparent background by adding alpha channel

            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            iso_crop = isolated[y1:y2, x1:x2]

            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), "output")

            os.makedirs(output_dir, exist_ok=True)

            export_name = f"{img_name}_{label}.png"
            output_dir = os.path.join(output_dir, export_name)

            # When cv2.imshow, background still apparent. Only when saved as png will be transparent
            _ = cv2.imwrite(output_dir, iso_crop)
            log.info(f"Image successfully cropped and exported to {output_dir}")
            return iso_crop
