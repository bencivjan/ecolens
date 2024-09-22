import cv2
import subprocess
import os
import numpy as np
from time import sleep,time,localtime,strftime,monotonic
# from multiprocessing import Process, Queue, Array
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from queue import Full
from diff_processor import PixelDiff, AreaDiff, EdgeDiff
from ffenc_uiuc.h264_encoder import ffenc, ffdec
from TC66C import TC66C

# Scaling governor must be set to userspace
# `sh -c 'sudo cpufreq-set -g userspace'`
def set_cpu_freq(cpu_freq):
    with open('/sys/devices/system/cpu/cpufreq/policy0/scaling_governor', 'r') as file:
        assert file.read().strip() == 'userspace', 'Scaling governor must be set to userspace\n`sudo cpufreq-set -g userspace`'
    
    cpu_freq = str(cpu_freq)
    print(f'Setting cpu freqency to {cpu_freq} KHz')

    command = f"echo {cpu_freq} | sudo tee /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)

    if result.returncode == 0:
        print("Successfully set cpu frequency")
    else:
        print("Failed to set cpu frequency")
        print(f"Error: {result.stderr}")

def throttle(target_fps, start_time):
    if target_fps == 0:
        raise ArithmeticError
    # Calculate the time to wait between frames
    frame_time = 1.0 / target_fps

    elapsed_time = monotonic() - start_time
    time_to_wait = frame_time - elapsed_time
    if time_to_wait > 0:
        sleep(time_to_wait)
        return True
    else:
        return False
    
def read_frames(cap: cv2.VideoCapture, shmem_name: str, cur_frame_idx: mp.Value, target_fps: int):
    # Test
    
    existing_shm = SharedMemory(name=shmem_name)
    shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=existing_shm.buf)


    _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = 0
    while True:
        now = monotonic()
        ret, frame = cap.read()
        if not ret:
            print('============ Finished video ============')
            with cur_frame_idx.get_lock():
                cur_frame_idx.value = -1 # Signal that video has ended
            break
        with cur_frame_idx.get_lock():
            shared_array[:,:,:] = frame
            cur_frame_idx.value = total_frames

        throttle(target_fps, now)
        total_frames += 1

def filter_frames(diff_processor, shmem_name: str, cur_frame_idx: mp.Value, encoding_queue: mp.Queue, width: int, height: int):
    # _, prev_frame = current_frame_queue.get()
    sleep(0.5)
    existing_shm = SharedMemory(name=shmem_name)
    shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=existing_shm.buf)

    with cur_frame_idx.get_lock():
        # prev_frame = np.frombuffer(cur_frame.get_obj(), dtype=np.uint8).reshape((height, width, 3))
        prev_frame = shared_array.copy()
    prev_feat = diff_processor.get_frame_feature(prev_frame)

    while True:
        with cur_frame_idx.get_lock():
            # frame = np.frombuffer(cur_frame.get_obj(), dtype=np.uint8).reshape((height, width, 3))
            frame = shared_array.copy()
            frame_idx = cur_frame_idx.value
            # print(f'Filter: frame_idx, {frame_idx}')
            if frame_idx == -1:
                encoding_queue.put((None, None))
                break
        feat = diff_processor.get_frame_feature(frame)
        dis = diff_processor.cal_frame_diff(feat, prev_feat)
        # print(f'Filter: dis, {dis}')

        if dis > diff_processor.thresh:
            prev_feat = feat
            print('========= Put Frame ===========')
            encoding_queue.put((frame_idx, frame))

def encode_frames(encoding_queue: mp.Queue, bitrate, fps, width, height, save_dir):
    encoder = ffenc(width, height, fps)
    decoder = ffdec()
    encoder.change_settings(bitrate, fps)

    while True:
        frame_idx, frame = encoding_queue.get()
        print(type(frame_idx), type(frame))
        print(f'Encode: ====== Received frame ========')
        # print(frame)
        if frame is None:
            break
        frame_cpy = frame.copy()
        encoded_frame = encoder.process_frame(frame_cpy)
        decoded_frame = decoder.process_frame(encoded_frame)
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

        # Save raw image
        filename = os.path.join(save_dir, f'frame{frame_idx}.npy')
        with open(filename, 'wb') as file:
            np.save(filename, decoded_frame)
    print('out of loop')


def generate_ground_truth(cap, width, height, bitrate, fps, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    encoder = ffenc(width, height, fps)
    decoder = ffdec()
    encoder.change_settings(bitrate, fps)

    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        encoded_frame = encoder.process_frame(frame)
        decoded_frame = decoder.process_frame(encoded_frame)
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

        filename = os.path.join(save_dir, f'frame{frame_idx}.npy')
        # with open(filename, 'wb') as file:
        #     file.write(decoded_frame.tobytes())
        np.save(filename, decoded_frame)
        frame_idx += 1

# def encode_video(cap, diff_processor, encoder, decoder, save_dir, target_fps, start):
#     total_frames = 0
#     _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     ret, prev_frame = cap.read()
#     prev_feat = diff_processor.get_frame_feature(prev_frame)

#     while True:
#         now = monotonic()
#         # print(f'{(now - start):5.1f}')
#         ret, frame = cap.read()
#         if not ret:
#             print('Finished video')
#             break
#         feat = diff_processor.get_frame_feature(frame)
#         dis = diff_processor.cal_frame_diff(feat, prev_feat)
#         if dis > diff_processor.thresh:
#             prev_feat = feat
#             encoded_frame = encoder.process_frame(frame)
#             decoded_frame = decoder.process_frame(encoded_frame)
#             decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

#             # Save frame to jpg
#             if not cv2.imwrite(os.path.join(save_dir, f'frame{total_frames}.jpg'), decoded_frame):
#                 raise Exception('Failed to write image')

#         throttle(target_fps, now)
#         total_frames += 1
    
#     print('Finished profiling')
#     print(f'{total_frames} frames')
    
#     return total_frames

def class2str(cls):
    if cls == PixelDiff:
        return 'pixel'
    elif cls == AreaDiff:
        return 'area'
    elif cls == EdgeDiff:
        return 'edge'
    else:
        raise Exception('Unknown class')
    
def read_energy(TC66, out_file, start, interval=1):
    with open(out_file,'w') as f:
        f.write('Time[S],Volt[V],Current[A],Power[W]\n')

        while True:
            now = monotonic()-start
            pd = TC66.Poll()
            s = '{:5.1f},{:07.5f},{:07.5f},{:07.5f}'.format(
                now,
                pd.Volt, 
                pd.Current,
                pd.Power)
            f.write(s+'\n')

            print(s)
            elapsed = (monotonic()-start) - now
            if elapsed < interval:
                sleep(interval - elapsed)

if __name__ == '__main__':
    print('Running')

    LOG_FILE = './timestamps.csv'
    VIDEO = '../distributed-360-streaming/videos/ny_driving.nut'
    FREQUENCIES = [1500000, 1800000, 2100000, 2400000]
    FILTERS = [PixelDiff, AreaDiff, EdgeDiff]
    # Batch 1: [0.1, 0.2, 0.3]
    # Batch 2: [0.4, 0.5, 0.6]
    # Batch 3: [0.7, 0.8, 0.9]
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    FRAME_BITRATES = [100, 400, 700, 1000, 1300, 1600, 1900, 2100, 2400, 2700, 3000] # kbps

    TC66 = TC66C('/dev/ttyACM0')
    cap = cv2.VideoCapture(VIDEO)
    target_fps = 25
    target_frame_duration = 1.0 / target_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # encoder = ffenc(width, height, fps)
    # decoder = ffdec()

    with open(LOG_FILE, mode='w') as file:
        file.write('Frequency,Filter,Threshold,Frame Bitrate,FPS,Start Time,End Time,Avg Energy\n')

    total_exp_start_time = monotonic()

    mp.Process(target=read_energy, args=(TC66,
                                      'TC66_'+strftime('%Y%m%d%H%M%S',localtime())+'.csv',
                                      total_exp_start_time)).start()

    cur_frame_idx = mp.Value('i')
    shm = SharedMemory(create=True, size=height*width*3)
    encoding_queue = mp.Queue()

    gt_dir = os.path.join(os.path.dirname(__file__), 'ground-truth')
    generate_ground_truth(cap, width, height, 3000, 30, gt_dir)

    for frequency in [1500000]:

        set_cpu_freq(frequency)

        for filter in [PixelDiff]:
            for threshold in THRESHOLDS:

                diff_processor = filter(thresh=threshold)

                for frame_bitrate in FRAME_BITRATES:
                    bitrate = frame_bitrate * fps
                    # encoder.change_settings(bitrate, fps)
                    save_dir = os.path.join(os.path.dirname(__file__), f'{frequency / 1_000_000}-{class2str(filter)}-{threshold}-{frame_bitrate}')
                    os.makedirs(save_dir, exist_ok=True)

                    cur_exp_start_time = monotonic()

                    # num_frames = encode_video(cap, diff_processor, encoder, decoder, save_dir, target_fps, total_exp_start_time)
                    read_frames_pid = mp.Process(target=read_frames, args=(cap, shm.name, cur_frame_idx, target_fps))
                    filter_frames_pid = mp.Process(target=filter_frames, args=(diff_processor, shm.name, cur_frame_idx, encoding_queue, width, height))
                    encode_frames_pid = mp.Process(target=encode_frames, args=(encoding_queue, bitrate, fps, width, height, save_dir))

                    read_frames_pid.start()
                    filter_frames_pid.start()
                    # ret, frame = cap.read()
                    # encoding_queue.put((0, frame))
                    encode_frames_pid.start()

                    try:
                        read_frames_pid.join()
                        filter_frames_pid.join()
                        encode_frames_pid.join()
                    except KeyboardInterrupt:
                        read_frames_pid.terminate()
                        filter_frames_pid.terminate()
                        encode_frames_pid.terminate()
                    finally:
                        # Clean up shared memory
                        shm.close()
                        shm.unlink()


                    cur_exp_end_time = monotonic()
                    tot_time = cur_exp_end_time - cur_exp_start_time

                    with open(LOG_FILE, mode='a') as file:
                            freq_ghz = frequency / 1_000_000
                            filter_str = class2str(filter)
                            start_time = cur_exp_start_time - total_exp_start_time
                            end_time = cur_exp_end_time - total_exp_start_time

                            file.write(f'{freq_ghz},{filter_str},{threshold},{frame_bitrate},{fps:.3f},{start_time:.3f},{end_time:.3f}\n')

                    sleep(2)
