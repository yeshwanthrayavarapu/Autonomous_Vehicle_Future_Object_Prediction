
#selects files from the data set
#number of scenes canot be greater than the number of scenes in data set (we gave a sample of 2 scenes in data set)

import re
import os
import random

working_dir=os.getcwd()
dataset_subset_dir=os.path.join(working_dir,"Dataset/camera_lidar_semantic_bboxes")
root_dir=dataset_subset_dir
# returns the paths of the frames from random scenes from a2d2 dataset
def frame_selection(root_dir, num_scenes, num_frames):
    root_dir = root_dir
    num_scenes_to_select = num_scenes
    num_frames_to_select = num_frames
    starting_frame = None

    scene_dirs = [d for d in os.listdir(root_dir) if d != ".DS_Store"]
    selected_scene_dir = random.sample(scene_dirs, num_scenes_to_select)[0]

    scene_dir_path = os.path.join(root_dir, selected_scene_dir)
    camera_dir_path = os.path.join(scene_dir_path, "camera", "cam_front_center")
    label_dir_path = os.path.join(scene_dir_path, "label", "cam_front_center")
    label3d_dir_path = os.path.join(scene_dir_path, "label3D", "cam_front_center")
    lidar_dir_path = os.path.join(scene_dir_path, "lidar", "cam_front_center")

    frames = []
    bboxes = []
    for file in os.listdir(camera_dir_path):
        if file.endswith(".png"):
            file_prefix = file.split("_")[3]
            file_prefix = os.path.splitext(file_prefix)[0]
            scene = re.sub(r"(?<=\d)_(?=\d)", "", selected_scene_dir)
            if all([os.path.isfile(os.path.join(camera_dir_path, file)),
                    os.path.isfile(os.path.join(label_dir_path, f"{scene}_label_frontcenter_{file_prefix}.png")),
                    os.path.isfile(os.path.join(label3d_dir_path, f"{scene}_label3D_frontcenter_{file_prefix}.json")),
                    os.path.isfile(os.path.join(lidar_dir_path, f"{scene}_lidar_frontcenter_{file_prefix}.npz"))]):
                frames.append(os.path.join(camera_dir_path, file))
    frames.sort()
    if starting_frame is None:
        starting_frame = random.choice(frames)

    starting_frame_index = frames.index(starting_frame)

    selected_frames = []
    for i in range(num_frames_to_select):
        selected_frames.append(frames[(starting_frame_index + i) % len(frames)])
    for j in selected_frames:
        j = j.replace('/camera/', '/label3D/')
        j = j.replace('_camera_frontcenter_', '_label3d_frontcenter_')
        j = j.replace('.png', '.json')
        bboxes.append(j)

    return selected_frames

