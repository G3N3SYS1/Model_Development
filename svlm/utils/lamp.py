import numpy as np
import operator

def track(box: np.ndarray, ref_pt, istracking, prev_pt, isfrontlamp) -> bool:
    isreset=False
    x1, y1, x2, y2 = box
    cX = int((x1+x2)/2)
    cY = int((y1+y2)/2)
    det_pt = (cX,cY)    
    reset_condition = operator.lt if isfrontlamp else operator.gt

    if inbox(ref_pt, box):
        tocrop = True
        if not istracking and not reset_condition(ref_pt[1], prev_pt[1]):
            isreset = True
            istracking = True
            print("Start tracking..")
        elif reset_condition(ref_pt[1], prev_pt[1]):
            istracking = False
        prev_pt = det_pt 
    else:
        tocrop = False
        if istracking and reset_condition(ref_pt[1], prev_pt[1]):
            istracking = False
            isreset = True
            

    return isreset, istracking, prev_pt, tocrop


def inbox(centroid, box):
    """Check if a centroid is inside a bounding box."""
    x_centroid, y_centroid = centroid
    x1, y1, x2, y2 = box
    return x1 <= x_centroid <= x2 and y1 <= y_centroid <= y2
