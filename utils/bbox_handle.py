def get_center(bbox):
    x1, y1, x2, y2 = bbox
    x_center = int( (x1 + x2)/2)
    y_center = int( (y1 + y2)/2)
    return x_center, y_center

def get_foot(bbox):
    x1, y1, x2, y2 = bbox
    x_center_bottom = int( (x1 + x2) / 2)
    return x_center_bottom, y2

def euclid_measure(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)** 0.5

def xy_measure(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_width(bbox):
    return bbox[2] - bbox[0]

def get_height(bbox):
    return bbox[3] - bbox[1]