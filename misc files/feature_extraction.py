import json
import os

import numpy as np


def get_distance(r):
    frame_path = r["frame_path"]

    lidar_path = frame_path.replace('/camera/', '/lidar/')
    lidar_path = lidar_path.replace('_camera_frontcenter_', '_lidar_frontcenter_')
    lidar_path = lidar_path.replace('.png', '.npz')
    # Load lidar data from npz file
    lidar_data = np.load(lidar_path)

    # Extract relevant data from npz file
    lidar_points = lidar_data['points']
    lidar_rows = lidar_data['row']
    lidar_cols = lidar_data['col']
    lidar_depths = lidar_data['depth']

    # print(lidar_rows)
    # print(lidar_cols)

    # Combine rows and cols into a list of (row, col) tuples
    lidar2d = list(zip(lidar_rows, lidar_cols))

    # Define a function to calculate the distance between two points
    def distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Define the target point
    target_point = np.array([600, 1200])

    results = []
    for box in r['rois']:
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        target_point = np.array([x, y])

        # Find the index of the nearest point in the lidar array
        distances = [distance(np.array(point), target_point) for point in lidar2d]
        nearest_point_index = np.argmin(distances)

        # Get the lidar point corresponding to the nearest row and col
        lidar_point = lidar_points[nearest_point_index]
        results.append(lidar_depths[nearest_point_index])
    # print("Nearest lidar point:", lidar_point)
    # print("distance",lidar_depths[nearest_point_index])
    return results


def get_areas(r):
    areas = []
    for i in range(len(r['rois'])):
        mask = r['masks'][:, :, i]
        area = sum(sum(mask * 1))
        areas.append(area)
    return areas


def get_relpos(r):
    rows = []
    for box in r["rois"]:
        x = (box[0] + box[2]) / 2
        if x < 960:
            rows.append(-1)
        else:
            rows.append(1)
        y = (box[1] + box[3]) / 2
        # col.append[y]
    return rows


def get_times(object_deletion_dict, frame_active_objects, frame_path):
    times = []
    present_frame = int(frame_path.split("/")[-1].split("_")[-1].split(".")[0])
    for i in frame_active_objects:
        last_frame = int(object_deletion_dict[i].split("/")[-1].split("_")[-1].split(".")[0])
        time = (last_frame - present_frame) / 30
        times.append(time)
    return times


def get_busdata(frame_path, root_dir):
    scene = frame_path.split('/')[-4]
    file = frame_path.split('/')[-1].split('_')[0]

    json_file = json_path = os.path.join(root_dir, scene, 'bus', file + '_bus_signals.json')

    with open(json_file, 'r') as f:
        data = json.load(f)

    my_frame = frame_path.split('/')[-1].split('.')[0] + '.json'

    ax = 0;
    ay = 0;
    speed = 0
    for i in range(len(data)):

        if data[i]['frame_name'] == my_frame:
            ax = sum(data[i]['flexray']['acceleration_x']['values']) / len(
                data[i]['flexray']['acceleration_x']['values'])
            ay = sum(data[i]['flexray']['acceleration_y']['values']) / len(
                data[i]['flexray']['acceleration_y']['values'])
            speed = sum(data[i]['flexray']['vehicle_speed']['values']) / len(
                data[i]['flexray']['vehicle_speed']['values'])
            break

    return ax, ay, speed

def feature_extractor(finres, object_deletion_dict, active_objects,root_dir_bus):
    X_distance = []
    X_accx = []
    X_accy = []
    X_speed = []
    X_maskarea = []
    X_relpos = []
    X_obj = []
    Y_time = []

    root_dir = os.path.append(root_dir_bus,' /camera_lidar_semantic_bus/')
    for i, r in enumerate(finres):
        X_distance = np.append(X_distance, get_distance(r))

        ax, ay, speed = get_busdata(r['frame_path'], root_dir)

        accx = ax * np.ones(len(r['rois']))

        # accx = np.array([ax for _ in range(len(r['rois']))])

        X_obj

        accy = np.array([ay for _ in range(len(r['rois']))])

        speed_ = np.array([speed for _ in range(len(r['rois']))])

        # print(X_accx, accx.shape)
        X_accx = np.append(X_accx, accx)
        X_accy = np.append(X_accy, accy)
        X_speed = np.append(X_speed, speed_)

        X_maskarea = np.append(X_maskarea, get_areas(r))
        X_relpos = np.append(X_relpos, get_relpos(r))
        Y_time = np.append(Y_time, get_times(object_deletion_dict, active_objects[i], r['frame_path']))

    print(X_distance.shape)
    print(X_maskarea.shape)
    print(X_accx.shape)
    print(X_speed.shape)
    print(X_accy.shape)
    print(X_relpos.shape)
    X = np.stack((X_distance, X_speed, X_maskarea, X_accx, X_accy, X_relpos))

    Y = Y_time

    return X, Y


X, Y = feature_extractor(finres, object_deletion_dict, active_objects)
print(X.shape, Y.shape)

