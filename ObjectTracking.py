import cv2
import numpy as np
import skimage
import os

from instance_segmentation import random_colors, class_names
from tracking import ObjectTracker

ROOT_DIR = os.getcwd()


# take the image and results and apply the mask, box, and Label

def display_tracks(image, boxes, ids, classid, names):
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        label = names[classid[i]]
        y1, x1, y2, x2 = boxes[i]
        caption = '{0}'.format(ids[i])
        caption1 = '{0}'.format(label)

        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption1, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    return image


def track_objects(finres):
    tracker = ObjectTracker(min_dist=50, max_skip=3, line_length=10, object_id=1)
    active_objects = []
    previous_list = []
    object_deletion_dict = dict()
    k = 0
    for i, r in enumerate(finres):
        centroids = []
        for box in r['rois']:
            x = (box[0] + box[2]) / 2
            y = (box[1] + box[3]) / 2
            # print(x,y)
            centroid = np.array([[x], [y]])
            centroids.append(centroid)
        if (len(centroids) > 0):
            tracker.Update(centroids)
        # print('Present centroids', centroids)
        # print(tracker.deleted_objects)
        # print(tracker.objects)

        tracked_objects = [obj.object_id for obj in tracker.objects]
        tracked_centroids = [obj.centroid for obj in tracker.objects]

        centroid_list = []
        for cen in centroids:
            x = np.array(cen).tolist()
            centroid_list.append([x[0][0], x[1][0]])

        tracked_centroid_list = []
        for track in tracked_centroids:
            x = np.array(track).tolist()
            tracked_centroid_list.append([x[0][0], x[1][0]])

        mapped_tracked_objects = []
        for centroid in centroid_list:
            if centroid in tracked_centroid_list:
                index = tracked_centroid_list.index(centroid)
                mapped_tracked_objects.append(tracked_objects[index])
            else:
                mapped_tracked_objects.append(0)

        # print(mapped_tracked_objects)

        current_list = mapped_tracked_objects
        active_objects.append(mapped_tracked_objects)
        diff_list = list(set(previous_list) - set(current_list))

        if diff_list != []:
            for c in diff_list:
                object_deletion_dict[c] = finres[i - 1]["frame_path"]

        previous_list = []
        if current_list != []:
            for c in current_list:
                previous_list.append(c)

        # for the objects in the last frame of the scene

        image = skimage.io.imread(r['frame_path'])

        trc = display_tracks(image, r['rois'], mapped_tracked_objects, r['class_ids'], class_names)

        name = '{0}.jpg'.format(10 * k)
        k += 1

        name = os.path.join(ROOT_DIR, "tracking_results", name)
        skimage.io.imsave(name, trc)
        print('writing to file:{0}'.format(name))

    for tracked_object in previous_list:
        object_deletion_dict[tracked_object] = finres[i]["frame_path"]

    return object_deletion_dict, active_objects
