import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import re
import cv2
import numpy as np
from ultralytics import YOLO
import torch.cuda
import time
from tqdm import tqdm
from ffenc_uiuc.h264_encoder import ffdec, ffenc
from diff_processor import PixelDiff
from utils import frame_iou_dynamic, sort_nicely, name2index, decode_from_path

class VideoConfiguration:
    def __init__(self, thresh, frame_bitrate):
        self.features = None
        self.prev_features = None
        self.filter = PixelDiff(thresh=thresh)
        self.frame_bitrate = frame_bitrate
        self.bb = None

class Evaluator:

    def __init__(self, ground_truth_dir):
        self.ground_truth_dir = ground_truth_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(os.path.join(os.path.dirname(__file__), '../yolov8x.pt')).to(self.device)
        self.decoder = ffdec()
        self.h264_processor_dict = {}
        self.fps = 30
        self.MAX_BITRATE = 3000

    def modify_frame_bitrate(self, frame, frame_bitrate):
        bitrate = frame_bitrate * self.fps
        if bitrate not in self.h264_processor_dict:
            encoder = ffenc(frame.shape[1], frame.shape[0], self.fps)
            encoder.change_settings(bitrate, self.fps)
            self.h264_processor_dict[bitrate] = (encoder, ffdec())

        enc_frame = self.h264_processor_dict[bitrate][0].process_frame(frame)
        return self.h264_processor_dict[bitrate][1].process_frame(enc_frame)

    def evaluate_configs(self, configs):
        if len(configs) == 0:
            return []
        video_configs = [VideoConfiguration(thresh=float(c[0]), frame_bitrate=int(c[1])) for c in configs]
        print([(c.filter.thresh, c.frame_bitrate) for c in video_configs])
        ious = [[] for _ in configs]
        # Iterate through ground truth
        for i, frame_name in tqdm(enumerate(sort_nicely(os.listdir(self.ground_truth_dir))), total=len(os.listdir(self.ground_truth_dir))):
            frame_idx = name2index(frame_name)
            # READ AS JPEG
            frame = cv2.imread(os.path.join(self.ground_truth_dir, frame_name))
            if frame is None:
                raise AssertionError(f'Unable to read image from {self.ground_truth_dir}')
            # READ AS H264
            # frame = decode_from_path(self.decoder, self.ground_truth_dir, frame_name)

            bb = self.model.predict(frame, verbose=False)[0].boxes

            # If frame satisfies filter, run inference & update bounding box
            for j, vc in enumerate(video_configs):
                vc.features = vc.filter.get_frame_feature(frame)

                if vc.prev_features is None:
                    vc.prev_features = vc.features
                dis = vc.filter.cal_frame_diff(vc.features, vc.prev_features)
                if not vc.bb:
                    vc.bb = bb

                if dis > vc.filter.thresh:
                    if vc.frame_bitrate == self.MAX_BITRATE:
                        vc.bb = bb
                    else:
                        temp_frame = self.modify_frame_bitrate(frame, vc.frame_bitrate)
                        vc.bb = self.model.predict(temp_frame, verbose=False)[0].boxes
                    vc.prev_features = vc.features

                # Calculate IoU based on bounding boxes
                iou = frame_iou_dynamic(bb, vc.bb)
                ious[j].append(iou)

        return [sum(config_iou) / len(config_iou) for config_iou in ious]
    

if __name__ == "__main__":
    evaluator = Evaluator('../filter-images/JH/1.5/1.5-pixel-0.0000-3000')

    # TEST BASIC EVAL
    # print(evaluator.evaluate_configs([[0.01, 3000], [0.02, 3000], [0.03, 3000]]))

    # TEST LOWER BITRATE
    # print(evaluator.evaluate_configs([[0.01, 1000]]))

    # TEST MIXED BITRATE
    # print(evaluator.evaluate_configs([[0.01, 3000], [0.01, 1000], [0.02, 3000], [0.02, 1600], [0.02, 1000]]))

    print(evaluator.evaluate_configs([[0.03, 400], [0.03, 100]]))