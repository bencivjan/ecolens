import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
import torch.cuda
import time
from ffenc_uiuc.h264_encoder import ffdec
from utils import *

def calculate_accuracy(ground_truth_dir, eval_dir, log_file=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(os.path.join(os.path.dirname(__file__), 'yolov8x.pt')).to(device)

    decoder = ffdec()
    
    eval_dir_idx = 0
    eval_dir_list = sort_nicely(os.listdir(eval_dir))

    eval_iou = np.zeros_like(os.listdir(ground_truth_dir), dtype=float)

    eval_result = None

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # ground_truth_output = cv2.VideoWriter('ground_truth_video.mp4', fourcc, 30.0, (1920, 1080))
    # eval_output = cv2.VideoWriter('eval_video.mp4', fourcc, 30.0, (1920, 1080))

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
        # ground_truth_output.write(ground_truth_result[0].plot())
        # eval_output.write(ground_truth_result[0].plot())

        print(frame_idx, name2index(eval_dir_list[eval_dir_idx]))

        eval_iou[i] = frame_iou_dynamic(ground_truth_result[0].boxes, eval_result[0].boxes)

        if log_file and i > 0 and i % 1800 == 0: # Log every 60 seconds at 30fps
            with open(log_file, mode='a') as file:
                file.write(f'Frame {frame_idx}: {eval_iou[:i].mean()}\n')

    # ground_truth_output.release()
    # eval_output.release()

    return eval_iou

if __name__ == '__main__':
    # IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'filter-images/JH/')
    # GROUND_TRUTH_DIR = f'{os.path.dirname(__file__)}/filter-images/ground-truth-JH'
    # LOG_FILE = f'{os.path.dirname(__file__)}/accuracy-JH-dynamic-1.5-1.8-{time.time()}.csv'

    # print(IMAGE_DIR)
    # with open(LOG_FILE, mode='w') as file:
    #     file.write('Frequency,Filter,Threshold,Frame Bitrate,Average IoU\n')

    # # batch_names = ['1.5', '1.8', '2.1', '2.4']
    # batch_names = ['1.5', '1.8']

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
    LOG_FILE = f'{os.path.dirname(__file__)}/JH-night-full.csv'
    CHECKPOINT_LOG_FILE = f'{os.path.dirname(__file__)}/JH-night-full-accuracy-log.txt'
    GROUND_TRUTH_DIR = f'{os.path.dirname(__file__)}/filter-images/ground-truth-JH-night-full'

    # img_path = '/media/ben/UBUNTU 22_0/JH-night-full/1.5-pixel-0.0200-1000'
    img_path = os.path.join(os.path.dirname(__file__), 'filter-images', 'JH-night-full-1.5-pixel-0.0100-2400')
    iou_ = calculate_accuracy(GROUND_TRUTH_DIR, img_path, CHECKPOINT_LOG_FILE)
    print(f'{os.path.basename(img_path)} IoU: {round(iou_.mean(), 4)}')
    with open(LOG_FILE, mode='a') as file:
        file.write('Frequency,Filter,Threshold,Frame Bitrate,Average IoU\n')
        img_dir_name = os.path.basename(img_path)
        _, _, freq, filter, thresh, bitrate = img_dir_name.split('-')
        file.write(f'{freq},{filter},{thresh},{bitrate},{iou_.mean():.4f}\n')