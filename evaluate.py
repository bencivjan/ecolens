import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import torch.cuda
import time
from ffenc_uiuc.h264_encoder import ffdec

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
    return int(frame_name[5:-4])

def decode_from_path(decoder, dir, name):
    # enc_frame = np.load(os.path.join(dir, name)).astype(np.uint8)
    with open(os.path.join(dir, name), 'rb') as f:
        byte_data = f.read()
    enc_frame = np.frombuffer(byte_data, dtype=np.uint8)
    decoded_frame = decoder.process_frame(enc_frame)
    return cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

def calculate_accuracy(ground_truth_dir, eval_dir, log_file=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(os.path.join(os.path.dirname(__file__), 'yolov8x.pt')).to(device)

    decoder = ffdec()
    
    eval_dir_idx = 0
    eval_dir_list = sort_nicely(os.listdir(eval_dir))

    eval_iou = np.zeros_like(os.listdir(ground_truth_dir), dtype=float)

    eval_result = None

    for i, frame_name in enumerate(sort_nicely(os.listdir(ground_truth_dir))):
        frame_idx = name2index(frame_name)

        # Start ground truth and eval at the same frame
        if frame_idx < name2index(eval_dir_list[0]):
            continue

        # Decode first frame if we have not yet
        if eval_dir_idx == 0 and not eval_result:
            es_frame = decode_from_path(decoder, eval_dir, eval_dir_list[eval_dir_idx])
            eval_result = model.predict(es_frame, verbose=False)

        # Decode next eval frame if it is equal to the ground truth frame index
        if eval_dir_idx+1 < len(eval_dir_list) and frame_idx >= name2index(eval_dir_list[eval_dir_idx+1]):
            eval_dir_idx += 1
            es_frame = decode_from_path(decoder, eval_dir, eval_dir_list[eval_dir_idx])
            eval_result = model.predict(es_frame, verbose=False)

        gt_frame = cv2.imread(os.path.join(ground_truth_dir, frame_name))
        ground_truth_result = model.predict(gt_frame, verbose=False)

        # cv2.imshow('Ground Truth', ground_truth_result[0].plot())
        # cv2.imshow('Eval', eval_result[0].plot())
        # cv2.waitKey(1)

        print(frame_idx, name2index(eval_dir_list[eval_dir_idx]))

        eval_iou[i] = frame_iou(ground_truth_result[0].boxes, eval_result[0].boxes)

        if log_file and i > 0 and i % 1800 == 0: # Log every 60 seconds at 30fps
            with open(log_file, mode='a') as file:
                file.write(f'Frame {frame_idx}: {eval_iou[:i].mean()}\n')

    return eval_iou

if __name__ == '__main__':
    # IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'filter-images/zero-filter-3000/flashdrive')
    # GROUND_TRUTH_DIR = f'{os.path.dirname(__file__)}/filter-images/ground-truth-JH-full'
    # LOG_FILE = f'{os.path.dirname(__file__)}/accuracy-{time.time()}.csv'
    # # ecostream_iou, allframes_iou = calculate_accuracy(os.path.join(PATH_STEM, GROUND_TRUTH_DIR), os.path.join(PATH_STEM, EVAL_DIR))

    # print(IMAGE_DIR)
    # with open(LOG_FILE, mode='w') as file:
    #     file.write('Frequency,Filter,Threshold,Frame Bitrate,Average IoU\n')

    # batch_names = ['1.5', '1.8', '2.1', '2.4']

    # for batch_name in batch_names:
    #     BATCH_DIR = os.path.join(IMAGE_DIR, batch_name)
    #     for img_directory in sort_nicely(os.listdir(BATCH_DIR)):

    #         img_path = os.path.join(BATCH_DIR, img_directory)

    #         if os.path.isdir(img_path):
    #             print(img_directory)
    #             iou_ = calculate_accuracy(GROUND_TRUTH_DIR, img_path)
    #             print(f'Batch {batch_name}: {img_directory} IoU: {round(iou_.mean(), 4)}')
    #             with open(LOG_FILE, mode='a') as file:
    #                 freq, filter, thresh, bitrate = img_directory.split('-')
    #                 file.write(f'{freq},{filter},{thresh},{bitrate},{iou_.mean():.4f}\n')
            

    # SINGLE CONFIG
    LOG_FILE = f'{os.path.dirname(__file__)}/JH-full.csv'
    GROUND_TRUTH_DIR = f'{os.path.dirname(__file__)}/filter-images/ground-truth-JH-full'
    img_path = os.path.join(os.path.dirname(__file__), 'filter-images', 'JH-full-1.5-pixel-0.0200-1000')
    # img_path = os.path.join(os.path.dirname(__file__), 'filter-images', 'JH', '1.5', '1.5-pixel-0.0200-1000')
    iou_ = calculate_accuracy(GROUND_TRUTH_DIR, img_path, 'JH-full-accuracy-log.txt')
    print(f'{os.path.basename(img_path)} IoU: {round(iou_.mean(), 4)}')
    with open(LOG_FILE, mode='a') as file:
        file.write('Frequency,Filter,Threshold,Frame Bitrate,Average IoU\n')
        img_dir_name = os.path.basename(img_path)
        _, _, freq, filter, thresh, bitrate = img_dir_name.split('-')
        file.write(f'{freq},{filter},{thresh},{bitrate},{iou_.mean():.4f}\n')