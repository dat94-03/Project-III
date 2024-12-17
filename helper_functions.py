import pickle
from shapely.geometry import Polygon

#find the center point of the slot
def find_polygon_center(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    center_x = int(sum(x_coords) / len(points))
    center_y = int(sum(y_coords) / len(points))
    return center_x, center_y

#save selected slot data
def save_object(polygon,src):
    with open("object/{src}.obj".format(src=src), "wb") as f:
        pickle.dump(polygon, f)
#load selected slot data
def load_object(src):
    try:
        with open("object/{src}.obj".format(src=src), "rb") as f:
            return pickle.load(f)
    except:
        save_object([],src)
        with open("object/{src}.obj".format(src=src), "rb") as f:
            return pickle.load(f)

#check if center of car slot in a car boundary
def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        p1x, p1y = polygon[i]
        p2x, p2y = polygon[(i + 1) % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x:
                inside = not inside
    return inside


def get_label_name(n):
    label = {
        0: "pedestrian",
        1: "people",
        2: "bicycle",
        3: "car",
        4: "van",
        5: "truck",
        6: "tricycle",
        7: "awning-tricycle",
        8: "bus",
        9: "motor",
    }
    return label[n]
