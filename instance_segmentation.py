import os
#import sys

import cv2
import matplotlib
import numpy as np
import skimage.io

#ROOT_DIR = os.path.abspath("../")
#   sys.path.append(ROOT_DIR)
 #   sys.path.append(os.getcwd())
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
#% matplotlib inline
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
from Mask_RCNN.samples.coco import coco



ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
RESULTS_PATH = os.path.join(ROOT_DIR, "results")
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               ]


#generates random color for each instance
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

#apply mask to image
def apply_mask(image, mask, color, alpha=0.5):

    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


#take the image and results and apply the mask, box, and Label

def display_instances(image, boxes, masks, ids, names, scores):

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    return image


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1


#perform instance segmentation on list of frames selected and saves them to results folder

def instance_segmentation(selected_frames):
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    batch_size=20
    if len(selected_frames)<20:
        batch_size=len(selected_frames)
    else:
        batch_size=20

    config = InferenceConfig()
    config.IMAGES_PER_GPU=batch_size
    config.BATCH_SIZE=batch_size
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    images = []
    output_path = []
    finres = []
    k = 1
    for i, frame_path in enumerate(selected_frames):
        image = skimage.io.imread(frame_path)
        images.append(image)
        if len(images) == batch_size or i == len(selected_frames) - 1:
            print('Predicted')
            results = model.detect(images, verbose=0)
            for j, r in enumerate(results):
                r.update({"frame_path": selected_frames[i - batch_size + j + 1]})
                finres.append(r)
            for j, r in enumerate(results):
                image = skimage.io.imread(selected_frames[i - batch_size + j + 1])
                res = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
                name = '{0}.jpg'.format(k)
                k += 1
                name = os.path.join(RESULTS_PATH, name)
                output_path.append(name)
                skimage.io.imsave(name, res)
                print('writing to file:{0}'.format(name))
            images = []
    return finres


