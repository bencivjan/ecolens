import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import cv2
import numpy as np
import logging
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from profiler import set_cpu_freq, throttle, filter_frames, encode_frames, \
                        read_energy, get_average_energy
from utils import sort_nicely
from time import sleep,time,localtime,strftime,monotonic
from diff_processor import PixelDiff
from TC66C import TC66C

class StreamSimulator:
    def __init__(self, image_dir, log_file) -> None:
        self.image_dir = image_dir
        self.log_file = log_file

        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(message)s')
        logging.info('Frame Range Start, Frame Range End, Configuration, Average Energy')

    # Dummy method to prevent deadlock by reading queue while simulation runs
    def read_queue(self, save_queue) -> None:
        frame, idx = save_queue.get()
        while frame is not None:
            # filename = os.path.join(os.path.dirname(__file__), 'images', f'frame{idx}.npy')
            filename = os.path.join(os.path.dirname(__file__), 'frameoutput.npy')
            np.save(filename, frame)
            frame, idx = save_queue.get()

    def simulate(self, config_list):
        TC66 = TC66C('/dev/ttyACM0')
        height = 1080
        width = 1920
        fps = 30
        target_fps = 25 # same as profiler

        try:

            filter_shm = SharedMemory(create=True, size=height*width*3)
            filter_shm_array = np.ndarray((height, width, 3), dtype=np.uint8, buffer=filter_shm.buf)
            encoding_shm = SharedMemory(create=True, size=height*width*3)

            filter_frame_idx = mp.Value('i')
            encoding_frame_idx = mp.Value('i')
            filter_frame_idx.value = -2
            encoding_frame_idx.value = -2

            save_queue = mp.Queue() # Encoded frames to save
            return_values = mp.Queue()
            frame_drop = mp.Queue() # For frame index sets at each step
            energy_readings = mp.Queue()

            total_exp_start_time = monotonic()

            energy_pid = mp.Process(target=read_energy, args=(TC66,
                                        total_exp_start_time, energy_readings),
                                        kwargs = {'out_file': None,
                                                    'interval': 0.25,
                                                    'verbose': False})
            energy_pid.start()
        
            set_cpu_freq(1500000)

            for start, end, threshold, bitrate in config_list:
                filter_frames_pid = mp.Process(target=filter_frames, args=(PixelDiff(thresh=threshold), filter_shm.name, filter_frame_idx, encoding_shm.name, encoding_frame_idx,  width, height, frame_drop))
                encode_frames_pid = mp.Process(target=encode_frames, args=(encoding_shm.name, encoding_frame_idx, bitrate, fps, width, height, save_queue, frame_drop))
                read_queue_pid = mp.Process(target=self.read_queue, args=(save_queue,))

                get_average_energy(energy_readings) # Clear out readings collected before experiment
                
                filter_frames_pid.start()
                encode_frames_pid.start()
                read_queue_pid.start()
                
                for i, frame_name in tqdm(enumerate(sort_nicely(os.listdir(self.image_dir))[start:end]), total=end - start, desc=f'{start}-{end}'):
                    now = monotonic()
                    frame = cv2.imread(os.path.join(self.image_dir, frame_name))

                    with filter_frame_idx.get_lock():
                        filter_shm_array[:,:,:] = frame
                        filter_frame_idx.value = i

                    throttle(target_fps, now)
                filter_frame_idx.value = -1 # Signal that video has ended

                filter_frames_pid.join()
                encode_frames_pid.join()
                read_queue_pid.join()

                average_energy = get_average_energy(energy_readings)
                logging.info(f'({start}, {end}, ({threshold}, {bitrate}), {average_energy}),')

            # This function won't return because of the energy reading process, use CTL C
            return
            
        finally:
            # Clean up shared memory
            if filter_shm:
                filter_shm.close()
                filter_shm.unlink()
                filter_shm = None
            if encoding_shm:
                encoding_shm.close()
                encoding_shm.unlink()
                encoding_shm = None

            print('Done')

if __name__ == '__main__':
    JH_DAY = f'{os.path.dirname(__file__)}/../ground-truth-videos/JH-full'
    LOG = f'{os.path.dirname(__file__)}/ecolens-JH-day-0.9-energy.csv'

    jh_day_90 = [
        # start index, end index, threshold, bitrate
        (0, 1950, 0.01, 700),
        (1950, 7800, 0.02, 3000),
        (7800, 9750, 0.02, 2700),
        (9750, 11700, 0.04, 1900),
        (11700, 13650, 0.03, 3000),
        (13650, 15600, 0.02, 2700),
        (15600, 17400, 0.01, 3000),
        (17550, 18017, 0.03, 3000)
    ]

    jh_day_85 = [
        (0, 1950, 0.03, 1000),
        (1950, 3900, 0.02, 1000),
        (3900, 5850, 0.03, 700),
        (5850, 7800, 0.02, 3000),
        (7800, 11700, 0.02, 400),
        (11700, 13650, 0.02, 3000),
        (13650, 15600, 0.03, 400),
        (15600, 17550, 0.03, 1900),
        (17550, 18017, 0.04, 1900),
    ]
    
    LOG = f'{os.path.dirname(__file__)}/ecolens-JH-day-0.9-energy.csv'
    simulator = StreamSimulator(JH_DAY, LOG)
    simulator.simulate(jh_day_90)

    sleep(5)

    simulator = StreamSimulator(JH_DAY, LOG)
    simulator.simulate(jh_day_85)

    print("Exiting")