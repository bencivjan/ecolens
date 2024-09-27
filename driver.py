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
    existing_shm = SharedMemory(name=shmem_name)
    shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=existing_shm.buf)

    _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = 0
    while True:
        now = monotonic()
        ret, frame = cap.read()
        if not ret:
            print('============ Finished video ============')
            cur_frame_idx.value = -1 # Signal that video has ended
            return
        with cur_frame_idx.get_lock():
            shared_array[:,:,:] = frame
            cur_frame_idx.value = total_frames

        throttle(target_fps, now)
        total_frames += 1

def filter_frames(diff_processor, filter_shmem_name: str, filter_frame_idx: mp.Value, encoding_shmem_name: str, encoding_frame_idx: mp.Value, width: int, height: int):
    filter_shm = SharedMemory(name=filter_shmem_name)
    encoding_shm = SharedMemory(name=encoding_shmem_name)
    filter_shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=filter_shm.buf)
    encoding_shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=encoding_shm.buf)

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

        # print(prev_idx, frame_idx)
        # print(f'difference: {dis}')
        max_dis = max(dis, max_dis)
        min_dis = min(dis, min_dis)

        if dis > diff_processor.thresh:
            prev_feat = feat
            prev_idx = frame_idx
            with encoding_frame_idx.get_lock():
                encoding_frame_idx.value = frame_idx
                encoding_shared_array[:,:,:] = frame

def encode_frames(encoding_shmem_name: str, encoding_frame_idx: mp.Value, bitrate: int, fps: int, width: int, height: int, save_dir: str):
    encoder = ffenc(width, height, fps)
    decoder = ffdec()
    encoder.change_settings(bitrate, fps)

    encoding_shm = SharedMemory(name=encoding_shmem_name)
    encoding_shared_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=encoding_shm.buf)

    prev_frame_idx = -2 # Use -2 to avoid conflict with -1 signaling termination

    while True:
        with encoding_frame_idx.get_lock():
            enc_frame_idx = encoding_frame_idx.value
            frame = encoding_shared_array.copy()
        # print(f'encoding_frame_idx.value: {enc_frame_idx}')
        if enc_frame_idx == -1:
            encoding_frame_idx.value = -2
            return
        elif prev_frame_idx == enc_frame_idx:
            sleep(0.05)
            continue
        prev_frame_idx = enc_frame_idx
        # print(f'Encode: Saving frame {enc_frame_idx}')
        encoded_frame = encoder.process_frame(frame)
        decoded_frame = decoder.process_frame(encoded_frame)
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_RGB2BGR)

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
    
def read_energy(TC66, out_file, start, energy_readings, interval=1):
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

            energy_readings.put(pd.Power)

            print(s)
            elapsed = (monotonic()-start) - now
            if elapsed < interval:
                sleep(interval - elapsed)

def get_average_energy(energy_readings):
    power_sum = 0
    num_readings = energy_readings.qsize()
    print(f'Number of energy readings: {num_readings}')

    for _ in range(num_readings):
        power_sum += energy_readings.get()
    
    return power_sum / num_readings if num_readings > 0 else 0

if __name__ == '__main__':
    print('Running')

    LOG_FILE = './1_5ghz.csv'
    VIDEO = './videos/ny_driving.nut'
    FREQUENCIES = [1500000, 1800000, 2100000, 2400000]
    FILTERS = [PixelDiff, AreaDiff, EdgeDiff]
    # Batch 1: [0.1, 0.2, 0.3]
    # Batch 2: [0.4, 0.5, 0.6]
    # Batch 3: [0.7, 0.8, 0.9]
    # THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    PIXEL_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]
    AREA_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]
    EDGE_THRESHOLDS = [0.0, 0.0004, 0.0008, 0.0012, 0.0016, 0.002, 0.0024, 0.0028, 0.0032, 0.0036, 0.004]

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

    filter_frame_idx = mp.Value('i')
    encoding_frame_idx = mp.Value('i')
    filter_frame_idx.value = -2
    encoding_frame_idx.value = -2

    filter_shm = SharedMemory(create=True, size=height*width*3)
    encoding_shm = SharedMemory(create=True, size=height*width*3)
    # encoding_queue = mp.Queue()
    energy_readings = mp.Queue()

    mp.Process(target=read_energy, args=(TC66,
                                      'TC66_'+strftime('%Y%m%d%H%M%S',localtime())+'.csv',
                                      total_exp_start_time, energy_readings),
                                      kwargs = {'interval': 0.25}).start()

    # gt_dir = os.path.join(os.path.dirname(__file__), 'ground-truth')
    # generate_ground_truth(cap, width, height, 3000, 30, gt_dir)

    try:
        for frequency in [1500000]:

            set_cpu_freq(frequency)

            for filter in FILTERS:
                
                if filter == PixelDiff:
                    thresholds = PIXEL_THRESHOLDS
                elif filter == AreaDiff:
                    thresholds = AREA_THRESHOLDS
                elif filter == EdgeDiff:
                    thresholds = EDGE_THRESHOLDS
                else:
                    raise ValueError('ERROR: Filter not recognized')

                for threshold in thresholds:

                    diff_processor = filter(thresh=threshold)

                    for frame_bitrate in FRAME_BITRATES:
                        bitrate = frame_bitrate * fps
                        save_dir = os.path.join(os.path.dirname(__file__), 'flashdrive', '1.5', f'{frequency / 1_000_000}-{class2str(filter)}-{threshold}-{frame_bitrate}')
                        # save_dir = os.path.join('/home', 'bencivjan', 'Desktop', 'flashdrive', 'batch1', f'{frequency / 1_000_000}-{class2str(filter)}-{threshold}-{frame_bitrate}')
                        os.makedirs(save_dir, exist_ok=True)

                        cur_exp_start_time = monotonic()

                        # num_frames = encode_video(cap, diff_processor, encoder, decoder, save_dir, target_fps, total_exp_start_time)
                        read_frames_pid = mp.Process(target=read_frames, args=(cap, filter_shm.name, filter_frame_idx, target_fps))
                        filter_frames_pid = mp.Process(target=filter_frames, args=(diff_processor, filter_shm.name, filter_frame_idx, encoding_shm.name, encoding_frame_idx,  width, height))
                        encode_frames_pid = mp.Process(target=encode_frames, args=(encoding_shm.name, encoding_frame_idx, bitrate, fps, width, height, save_dir))

                        get_average_energy(energy_readings) # Clear out readings collected before experiment

                        read_frames_pid.start()
                        filter_frames_pid.start()
                        encode_frames_pid.start()

                        read_frames_pid.join()
                        filter_frames_pid.join()
                        encode_frames_pid.join()

                        cur_exp_end_time = monotonic()
                        tot_time = cur_exp_end_time - cur_exp_start_time
                        average_energy = get_average_energy(energy_readings)

                        with open(LOG_FILE, mode='a') as file:
                                freq_ghz = frequency / 1_000_000
                                filter_str = class2str(filter)
                                start_time = cur_exp_start_time - total_exp_start_time
                                end_time = cur_exp_end_time - total_exp_start_time

                                file.write(f'{freq_ghz},{filter_str},{threshold},{frame_bitrate},{fps:.3f},{start_time:.3f},{end_time:.3f},{average_energy}\n')

                        sleep(10)
    except Exception as e:
        print(f'ERROR: {e}')
        with open('errors.out', 'w') as f:
            f.write(e)
    finally:
        # Clean up shared memory
        filter_shm.close()
        filter_shm.unlink()
        encoding_shm.close()
        encoding_shm.unlink()
