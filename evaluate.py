import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import torch.cuda

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1, box2: Lists or tuples with 4 elements each [x, y, width, height].

    Returns:
    - IoU: Intersection over Union (float).
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)

    inter_area = inter_width * inter_height
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def frame_iou(ground_truth, prediction):
    '''
    Parameters:
    - ground_truth: Ground truth bounding box list
    - prediction: Prediction bounding box list

    Returns:
    - Frame IoU
    '''
    if len(ground_truth) == 0:
        return 1
    elif len(prediction) == 0:
        return 0
    
    gt_to_pred_iou = np.zeros((len(ground_truth), len(prediction)))
    for i, truth in enumerate(ground_truth):
        for j, pred in enumerate(prediction):
            if pred.cls == truth.cls:
                gt_to_pred_iou[i][j] = iou(truth.xyxy[0], pred.xyxy[0])

    gt_to_pred_iou = gt_to_pred_iou.max(axis=1)
    return gt_to_pred_iou.mean()

def sort_nicely( l ):
    """ 
        From https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
        Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def name2index(frame_name):
    return int(frame_name.split('.')[0].split('_')[1])

def calculate_accuracy(ground_truth_dir, eval_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(os.path.join(PATH_STEM, '..', 'yolov8x.pt')).to(device)
    
    eval_dir_idx = 0
    eval_dir_list = sort_nicely(os.listdir(eval_dir))

    allframes_iou = np.zeros_like(os.listdir(ground_truth_dir), dtype=float)
    eval_iou = np.zeros_like(os.listdir(ground_truth_dir), dtype=float)

    eval_result = None

    for i, frame_name in enumerate(sort_nicely(os.listdir(ground_truth_dir))):
        frame_idx = name2index(frame_name)

        # Start ground truth and eval at the same frame
        if frame_idx < name2index(eval_dir_list[0]):
            continue

        if eval_dir_idx+1 < len(eval_dir_list) and frame_idx >= name2index(eval_dir_list[eval_dir_idx+1]):
            eval_dir_idx += 1
            # es_frame = cv2.imread(os.path.join(eval_dir, eval_dir_list[eval_dir_idx]))
            es_frame = np.load(os.path.join(eval_dir, eval_dir_list[eval_dir_idx]))
            eval_result = model.predict(es_frame, verbose=False)
        
        print(frame_idx, name2index(eval_dir_list[eval_dir_idx]))

        # gt_frame = cv2.imread(os.path.join(ground_truth_dir, frame_name))
        gt_frame = np.load(os.path.join(ground_truth_dir, frame_name))
        ground_truth_result = model.predict(gt_frame, verbose=False)
        allframes_result = model.predict(gt_frame, verbose=False)

        if eval_result is None:
            # Use first frame for eval if we haven't predicted yet
            eval_result = allframes_result

        allframes_iou[i] = frame_iou(ground_truth_result[0].boxes, allframes_result[0].boxes)
        eval_iou[i] = frame_iou(ground_truth_result[0].boxes, eval_result[0].boxes)

    return eval_iou, allframes_iou

if __name__ == '__main__':
    PATH_STEM = os.path.dirname(__file__)
    GROUND_TRUTH_DIR = 'ny_driving/30fps'
    EVAL_DIR = 'ny_driving/25fps'
    ecostream_iou, allframes_iou = calculate_accuracy(os.path.join(PATH_STEM, GROUND_TRUTH_DIR), os.path.join(PATH_STEM, EVAL_DIR))
    
    print(f'Ecostream IoU by frame: {ecostream_iou.reshape(-1, 1)}')
    print(f'All frames IoU by frame: {allframes_iou.reshape(-1, 1)}')
    print(f'EcoStream IoU: {round(ecostream_iou.mean(), 4)}')
    print(f'Sending all frames IoU: {round(allframes_iou.mean(), 4)}')