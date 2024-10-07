import cv2
import subprocess
import os
import numpy as np
from time import sleep,time,localtime,strftime,monotonic
from datetime import datetime
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
    
def read_frames(cap: cv2.VideoCapture, shmem_name: str, cur_frame_idx: mp.Value, target_fps: int, ret_queue: mp.Queue, frame_drop: mp.Queue):
    existing_shm = SharedMemory(name=shmem_name)
    shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=existing_shm.buf)
    total_frames_set = set() # Set to keep track of frame dropping

    _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = 0
    read_start_time = monotonic()
    while True:
        now = monotonic()
        ret, frame = cap.read()
        if not ret:
            print('============ Finished video ============')
            ret_queue.put(total_frames / (monotonic() - read_start_time))
            frame_drop.put(total_frames_set)
            cur_frame_idx.value = -1 # Signal that video has ended
            return
        with cur_frame_idx.get_lock():
            shared_array[:,:,:] = frame
            cur_frame_idx.value = total_frames

        throttle(target_fps, now)
        total_frames_set.add(total_frames)
        total_frames += 1

def filter_frames(diff_processor, filter_shmem_name: str, filter_frame_idx: mp.Value, encoding_shmem_name: str, encoding_frame_idx: mp.Value, width: int, height: int, frame_drop: mp.Queue):
    filter_shm = SharedMemory(name=filter_shmem_name)
    encoding_shm = SharedMemory(name=encoding_shmem_name)
    filter_shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=filter_shm.buf)
    encoding_shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=encoding_shm.buf)

    filtered_input_set = set() # Set to keep track of frame dropping
    filter_output_set = set()

    max_dis = 0
    min_dis = 99999999

    while filter_frame_idx.value != 0: # Read frame process has not set first frame yet
        sleep(0.01)

    with filter_frame_idx.get_lock():
        prev_frame = filter_shared_array.copy()
        prev_idx = filter_frame_idx.value
    with encoding_frame_idx.get_lock():
        encoding_frame_idx.value = prev_idx
        encoding_shared_array[:,:,:] = prev_frame # Always encode the first frame we see
    prev_feat = diff_processor.get_frame_feature(prev_frame)

    while True:
        with filter_frame_idx.get_lock():
            frame = filter_shared_array.copy()
            frame_idx = filter_frame_idx.value
        if frame_idx == -1:
            frame_drop.put(filtered_input_set)
            frame_drop.put(filter_output_set)
            filter_frame_idx.value = -2
            encoding_frame_idx.value = -1
            # print(f'MAX DIFFERENCE {max_dis}')
            # print(f'MIN DIFFERENCE {min_dis}')
            return
        elif frame_idx == prev_idx:
            sleep(0.01)
            continue

        feat = diff_processor.get_frame_feature(frame)
        dis = diff_processor.cal_frame_diff(feat, prev_feat)
        filtered_input_set.add(frame_idx)

        # print(prev_idx, frame_idx)
        # print(f'difference: {dis}')
        max_dis = max(dis, max_dis)
        min_dis = min(dis, min_dis)

        if dis > diff_processor.thresh:
            prev_feat = feat
            prev_idx = frame_idx # TODO: Move this out of if statement?
            with encoding_frame_idx.get_lock():
                encoding_frame_idx.value = frame_idx
                encoding_shared_array[:,:,:] = frame
            filter_output_set.add(frame_idx)

def encode_frames(encoding_shmem_name: str, encoding_frame_idx: mp.Value, bitrate: int, fps: int, width: int, height: int, save_dir: str, frame_drop: mp.Queue):
    encoder = ffenc(width, height, fps)
    decoder = ffdec()
    encoder.change_settings(bitrate, fps)

    encoding_shm = SharedMemory(name=encoding_shmem_name)
    encoding_shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=encoding_shm.buf)

    encoded_frames_set = set() # Set to keep track of frame dropping

    prev_frame_idx = -2 # Use -2 to avoid conflict with -1 signaling termination

    while True:
        with encoding_frame_idx.get_lock():
            enc_frame_idx = encoding_frame_idx.value
            frame = encoding_shared_array.copy()
        # print(f'encoding_frame_idx.value: {enc_frame_idx}')
        if enc_frame_idx == -1:
            frame_drop.put(encoded_frames_set)
            encoding_frame_idx.value = -2
            return
        elif prev_frame_idx == enc_frame_idx:
            sleep(0.01)
            continue
        prev_frame_idx = enc_frame_idx
        # print(f'Encode: Saving frame {enc_frame_idx}')
        encoded_frame = encoder.process_frame(frame)
        decoded_frame = decoder.process_frame(encoded_frame)
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

        encoded_frames_set.add(enc_frame_idx)

        # Save raw image
        filename = os.path.join(save_dir, f'frame{enc_frame_idx}.npy')
        np.save(filename, decoded_frame)

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

def class2str(cls):
    if cls == PixelDiff:
        return 'pixel'
    elif cls == AreaDiff:
        return 'area'
    elif cls == EdgeDiff:
        return 'edge'
    else:
        raise Exception('Unknown class')
    
def read_energy(TC66, start, energy_readings, out_file='TC66_'+strftime('%Y%m%d%H%M%S',localtime())+'.csv', interval=1):
    if out_file:
        f = open(out_file,'w')
        f.write('Time[S],Volt[V],Current[A],Power[W]\n')

    try:
        while True:
            now = monotonic()-start
            pd = TC66.Poll()
            s = '{:5.1f},{:07.5f},{:07.5f},{:07.5f}'.format(
                now,
                pd.Volt, 
                pd.Current,
                pd.Power)
            if out_file:
                f.write(s+'\n')

            energy_readings.put(pd.Power)

            print(s)
            elapsed = (monotonic()-start) - now
            if elapsed < interval:
                sleep(interval - elapsed)
    except KeyboardInterrupt:
        if out_file:
            f.close()

def get_average_energy(energy_readings):
    power_sum = 0
    num_readings = energy_readings.qsize()
    print(f'Number of energy readings: {num_readings}')

    for _ in range(num_readings):
        power_sum += energy_readings.get()
    
    return power_sum / num_readings if num_readings > 0 else 0

if __name__ == '__main__':
    print('Running')

    current_time = datetime.now()
    LOG_FILE = f'./test-{current_time.strftime("%Y%m%d%H%M%S")}.csv'
    VIDEO = './videos/ny_driving.nut'
    FREQUENCIES = [1500000, 1800000, 2100000, 2400000]
    FILTERS = [PixelDiff, AreaDiff, EdgeDiff]
    # Batch 1: [0.1, 0.2, 0.3]
    # Batch 2: [0.4, 0.5, 0.6]
    # Batch 3: [0.7, 0.8, 0.9]
    # THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    PIXEL_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10 ]
    AREA_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10 ]
    EDGE_THRESHOLDS = [0.00, 0.0004, 0.0008, 0.0012, 0.0016, 0.0020, 0.0024, 0.0028, 0.0032, 0.0036, 0.0040]

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

    # encoding_queue = mp.Queue()
    energy_readings = mp.Queue()

    mp.Process(target=read_energy, args=(TC66,
                                      total_exp_start_time, energy_readings),
                                      kwargs = {'out_file': None,
                                                'interval': 0.25}).start()

    # gt_dir = os.path.join(os.path.dirname(__file__), 'ground-truth')
    # generate_ground_truth(cap, width, height, 3000, 30, gt_dir)

    try:
        for frequency in [1800000]:

            set_cpu_freq(frequency)

            filter_shm = SharedMemory(create=True, size=height*width*3)
            encoding_shm = SharedMemory(create=True, size=height*width*3)

            filter_frame_idx = mp.Value('i')
            encoding_frame_idx = mp.Value('i')
            filter_frame_idx.value = -2
            encoding_frame_idx.value = -2

            return_values = mp.Queue()
            frame_drop = mp.Queue() # For frame index sets at each step

            for filter in FILTERS:
            # for filter in [AreaDiff]:

                if filter == PixelDiff:
                    thresholds = PIXEL_THRESHOLDS
                elif filter == AreaDiff:
                    thresholds = AREA_THRESHOLDS
                elif filter == EdgeDiff:
                    thresholds = EDGE_THRESHOLDS
                else:
                    raise ValueError('ERROR: Filter not recognized')

                for threshold in thresholds:
                # for threshold in [0.00]:

                    diff_processor = filter(thresh=threshold)

                    for frame_bitrate in FRAME_BITRATES:
                    # for frame_bitrate in [3000] * 2:
                        bitrate = frame_bitrate * fps
                        save_dir = os.path.join(os.path.dirname(__file__), 'flashdrive', f'{frequency / 1_000_000}-{class2str(filter)}-{threshold:.4f}-{frame_bitrate}')
                        # save_dir = os.path.join('/home', 'bencivjan', 'Desktop', 'flashdrive', 'batch1', f'{frequency / 1_000_000}-{class2str(filter)}-{threshold}-{frame_bitrate}')
                        os.makedirs(save_dir, exist_ok=True)

                        cur_exp_start_time = monotonic()

                        # num_frames = encode_video(cap, diff_processor, encoder, decoder, save_dir, target_fps, total_exp_start_time)
                        read_frames_pid = mp.Process(target=read_frames, args=(cap, filter_shm.name, filter_frame_idx, target_fps, return_values, frame_drop))
                        filter_frames_pid = mp.Process(target=filter_frames, args=(diff_processor, filter_shm.name, filter_frame_idx, encoding_shm.name, encoding_frame_idx,  width, height, frame_drop))
                        encode_frames_pid = mp.Process(target=encode_frames, args=(encoding_shm.name, encoding_frame_idx, bitrate, fps, width, height, save_dir, frame_drop))

                        get_average_energy(energy_readings) # Clear out readings collected before experiment

                        read_frames_pid.start()
                        filter_frames_pid.start()
                        encode_frames_pid.start()

                        read_frames_pid.join()
                        filter_frames_pid.join()
                        encode_frames_pid.join()

                        real_fps = return_values.get()

                        total_frames_set = frame_drop.get()
                        filter_input_set = frame_drop.get()
                        filter_output_set = frame_drop.get()
                        encoded_frames_set = frame_drop.get()
                        filter_input_dropped_frames = total_frames_set - filter_input_set
                        filtered_frames = filter_input_set - filter_output_set
                        encoder_dropped_frames = filter_output_set - encoded_frames_set
                        total_dropped_frames = total_frames_set - encoded_frames_set
                        print(f'Filter input dropped frames: {len(filter_input_dropped_frames)}')
                        print(f'Filtered frames: {len(filtered_frames)}')
                        print(f'Encoder dropped frames: {len(encoder_dropped_frames)}')
                        print(f'Total dropped frames: {len(total_dropped_frames)}')

                        cur_exp_end_time = monotonic()
                        tot_time = cur_exp_end_time - cur_exp_start_time
                        average_energy = get_average_energy(energy_readings)

                        with open(LOG_FILE, mode='a') as file:
                                freq_ghz = frequency / 1_000_000
                                filter_str = class2str(filter)
                                start_time = cur_exp_start_time - total_exp_start_time
                                end_time = cur_exp_end_time - total_exp_start_time

                                file.write(f'{freq_ghz},{filter_str},{threshold:4f},{frame_bitrate},{real_fps:.3f},{start_time:.3f},{end_time:.3f},{average_energy}\n')
                        
                        sleep(10)
            
            # Clean up shared memory
            filter_shm.close()
            filter_shm.unlink()
            encoding_shm.close()
            encoding_shm.unlink()
            filter_shm = None
            encoding_shm = None

    except Exception as e:
        print(f'ERROR: {e}')
        with open('errors.out', 'w') as f:
            f.write(str(e))
    finally:
        # Clean up shared memory
        if filter_shm:
            filter_shm.close()
            filter_shm.unlink()
        if encoding_shm:
            encoding_shm.close()
            encoding_shm.unlink()
