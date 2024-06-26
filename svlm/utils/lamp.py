import numpy as np
import operator

def track(
        box: np.ndarray,
        ref_pt: tuple,
        istracking: bool,
        prev_pt: tuple,
        isfrontlamp: bool
    ) -> tuple[bool, bool, tuple, bool]:
    """Track the status of a bounding box, determine when to crop for lamp segmentation,
    and when to reset the tracking status.

    - Crop for lamp segmentation only when the reference point is inside the bounding box.
    - Reset condition occurs when the y-coordinate of the centroid of the bounding box
    is greater or lower (based on front or rear view) than the y-coordinate of the reference point.

    :param box: Bounding box coordinates of the detected vehicle in the format (x1, y1, x2, y2).
    :type box: np.ndarray
    :param ref_pt: Reference point or trigger point that determines reset conditions.
    :type ref_pt: tuple (x, y)
    :param istracking: Flag indicating whether the bounding box is currently being tracked.
    :type istracking: bool
    :param prev_pt: Previous centroid point (last seen point) of the bounding box.
    :type prev_pt: tuple (x, y)
    :param isfrontlamp: Flag indicating whether the view is from the front or rear of the vehicle.
    :type isfrontlamp: bool

    :return: A tuple containing:
             - isreset: Flag indicating if the tracking status should be reset.
             - istracking: Updated tracking status based on the current conditions.
             - prev_pt: Updated previous centroid point (last seen point).
             - tocrop: Flag indicating whether to perform lamp segmentation crop.

    :rtype: tuple (bool, bool, tuple, bool)
    """
    isreset=False
    x1, y1, x2, y2 = box
    cX = int((x1+x2)/2)
    cY = int((y1+y2)/2)
    det_pt = (cX,cY)
    reset_condition = operator.lt if isfrontlamp else operator.gt

    if inbox(ref_pt, box):
        # Case: Reference point is inside the bounding box
        tocrop = True

        if not istracking and not reset_condition(ref_pt[1], prev_pt[1]):
            # Start tracking if not already tracking and not needing reset
            isreset = True
            istracking = True
            print("Start tracking..")
        elif reset_condition(ref_pt[1], prev_pt[1]):
            # Stop tracking if reset condition is met
            istracking = False

        # Update the last seen point
        prev_pt = det_pt 
    else:
         # Case: Reference point is outside the bounding box
        tocrop = False
        
        if istracking and reset_condition(ref_pt[1], prev_pt[1]):
            # Stop tracking if reset condition is met
            # Reset flag set to true
            istracking = False
            isreset = True
            

    return isreset, istracking, prev_pt, tocrop


def inbox(centroid, box):
    """Check if a centroid is inside a bounding box.
    
    param centroid: Centroid coordinates (x, y).
    :type centroid: tuple
    :param box: Bounding box coordinates (x1, y1, x2, y2).
    :type box: np.ndarray

    :return: True if the centroid is inside the bounding box, False otherwise.
    :rtype: bool
    """
    x_centroid, y_centroid = centroid
    x1, y1, x2, y2 = box
    return x1 <= x_centroid <= x2 and y1 <= y_centroid <= y2
