import cv2
import os
import numpy as np
import re

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
        return 1.0 if len(prediction) == 0 else 0.0
    elif len(prediction) == 0:
        return 0.0
    
    gt_to_pred_iou = np.zeros((len(ground_truth), len(prediction)))
    for i, truth in enumerate(ground_truth):
        for j, pred in enumerate(prediction):
            if pred.cls == truth.cls:
                gt_to_pred_iou[i][j] = iou(truth.xyxy[0], pred.xyxy[0])

    gt_to_pred_iou = gt_to_pred_iou.max(axis=1)
    return gt_to_pred_iou.mean()

def frame_iou_dynamic(ground_truth, prediction):
    '''
    Parameters:
    - ground_truth: Ground truth bounding box list
    - prediction: Prediction bounding box list

    Returns:
    - Frame IoU
    '''
    target_classes = [
        0, # Person
        1, # Bicycle
        2, # Car
        3, # Motorcycle
        4, # Airplane
        5, # Bus
        6, # Train
        7, # Truck
    ]
    # Filter ground_truth and prediction to include only target classes
    ground_truth = [gt for gt in ground_truth if gt.cls in target_classes]
    prediction = [pred for pred in prediction if pred.cls in target_classes]

    # Handle edge cases
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) == 0 else 0.0
    if len(prediction) == 0:
        return 0.0

    # Initialize IoU matrix
    gt_to_pred_iou = np.zeros((len(ground_truth), len(prediction)))

    # Calculate IoU for matching classes
    for i, truth in enumerate(ground_truth):
        for j, pred in enumerate(prediction):
            if pred.cls == truth.cls:
                gt_to_pred_iou[i][j] = iou(truth.xyxy[0], pred.xyxy[0])

    # Get the best IoU for each ground truth and calculate mean
    max_ious = gt_to_pred_iou.max(axis=1)
    return max_ious.mean()

def sort_nicely( l ):
    """ 
        From https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
        Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def name2index(frame_name):
    return int(frame_name[5:-4])

def decode_from_path(decoder, dir, name):
    # enc_frame = np.load(os.path.join(dir, name)).astype(np.uint8)
    with open(os.path.join(dir, name), 'rb') as f:
        byte_data = f.read()
    enc_frame = np.frombuffer(byte_data, dtype=np.uint8)
    decoded_frame = decoder.process_frame(enc_frame)
    return cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)