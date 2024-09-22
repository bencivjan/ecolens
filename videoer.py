import argparse
from pathlib import Path

from diff_processor import PixelDiff, AreaDiff, CornerDiff, EdgeDiff
from video_processor import VideoProcessor
from ffenc_uiuc.h264_encoder import ffenc, ffdec
import cv2, time


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='southampton')
    parser.add_argument('-s', '--subset_pattern', type=str, default='raw000')
    parser.add_argument('-y', '--yes', action='store_true')
    return parser


if __name__ == '__main__':
    # args = make_parser().parse_args()
    # dataset_name = args.dataset_name
    # subset_pattern = args.subset_pattern

    # dataset_root = '/mnt/shared/dataset'
    # segment_root = Path(dataset_root) / dataset_name / subset_pattern
    # segments = [f for f in sorted(segment_root.iterdir()) if f.match('segment???.mp4')]

    # videoer = Videoer(dataset_root=dataset_root,
    #                   dataset_name=dataset_name,
    #                   subset_pattern=subset_pattern)

    # dps = [
    #     PixelDiff(thresh=0.01),
    #     AreaDiff(thresh=0.01),
    #     CornerDiff(thresh=0.01),
    #     EdgeDiff(thresh=0.01)
    # ]

    VIDEO = '../distributed-360-streaming/videos/ny_driving.nut'
    cap = cv2.VideoCapture(VIDEO)
    diff_processor = EdgeDiff(thresh=0)
    test_start_time = time.time()
    total_frames = 0
    target_fps = 15
    target_frame_duration = 1.0 / target_fps

    # while True:
    #     try:
    #         diff_results = diff_processor.process_video(VIDEO)
    #         total_frames += diff_results['num_total_frames']
    #         print(diff_results)
    #         # if time.time() - test_start_time > 75:
    #         if total_frames > 5000:
    #             break
    #     except:
    #         break

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    encoder = ffenc(int(width), int(height), int(fps))

    ret, prev_frame = cap.read()
    prev_feat = diff_processor.get_frame_feature(prev_frame)
    total_frames = 0

    # ==== For basic encoding ====
    # while True:
    #     try:
    #         start_time = time.time()
    #         ret, frame = cap.read()
    #         if not ret:
    #             print('Restarting video')
    #             _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #             continue
    #         total_frames += 1
    #         _ = encoder.process_frame(frame)
            
    #         elapsed_time = time.time() - start_time
    #         sleep_time = target_frame_duration - elapsed_time
    #         if sleep_time > 0:
    #             time.sleep(sleep_time)
    #     except:
    #         break
    
    # ==== For frame filtering ====
    while True:
        try:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print('Restarting video')
                _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            total_frames += 1
            feat = diff_processor.get_frame_feature(frame)
            dis = diff_processor.cal_frame_diff(feat, prev_feat)
            if dis > diff_processor.thresh:
                prev_feat = feat
                _ = encoder.process_frame(frame)
            
            elapsed_time = time.time() - start_time
            sleep_time = target_frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        except:
            break
        
    total_time = time.time() - test_start_time
    print('Finished profiling')
    print(f'{total_frames} frames')
    print(f'{total_time} seconds')
    print(f'{total_frames / total_time} fps')