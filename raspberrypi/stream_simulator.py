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
from diff_processor import PixelDiff, EdgeDiff
from TC66C import TC66C

class StreamSimulator:
    def __init__(self, image_dir, log_file) -> None:
        self.image_dir = image_dir
        self.log_file = log_file

    # Dummy method to prevent deadlock by reading queue while simulation runs
    def read_queue(self, save_queue) -> None:
        frame, idx = save_queue.get()
        while frame is not None:
            # filename = os.path.join(os.path.dirname(__file__), 'images', f'frame{idx}.npy')
            filename = os.path.join(os.path.dirname(__file__), 'frameoutput.npy')
            np.save(filename, frame)
            frame, idx = save_queue.get()

    def simulate(self, config_list, processor=PixelDiff):
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(message)s', force=True)
        logging.info('Frame Range Start, Frame Range End, Configuration, Average Energy')

        TC66 = TC66C('/dev/ttyACM0')
        height = 1080
        width = 1920
        fps = 30
        target_fps = 20 # same as profiler

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
                                                    'interval': 0.125,
                                                    'verbose': False})
            energy_pid.start()

            for start, end, (threshold, bitrate) in config_list:
                threshold, bitrate = float(threshold), int(bitrate)
                print(threshold, bitrate)
                filter = processor(thresh=threshold)
                filter_frames_pid = mp.Process(target=filter_frames, args=(filter, filter_shm.name, filter_frame_idx, encoding_shm.name, encoding_frame_idx,  width, height, frame_drop))
                encode_frames_pid = mp.Process(target=encode_frames, args=(encoding_shm.name, encoding_frame_idx, bitrate, fps, width, height, save_queue, frame_drop))
                read_queue_pid = mp.Process(target=self.read_queue, args=(save_queue,))

                get_average_energy(energy_readings) # Clear out readings collected before experiment
                
                filter_frames_pid.start()
                encode_frames_pid.start()
                read_queue_pid.start()

                total_frames_set = set() # Set to keep track of frame dropping
                total_frames = 0
                read_start_time = monotonic()
                
                for i, frame_name in tqdm(enumerate(sort_nicely(os.listdir(self.image_dir))[start:end]), total=end - start, desc=f'{start}-{end}'):
                    now = monotonic()
                    frame = cv2.imread(os.path.join(self.image_dir, frame_name))

                    with filter_frame_idx.get_lock():
                        filter_shm_array[:,:,:] = frame
                        filter_frame_idx.value = i

                    total_frames_set.add(total_frames)
                    total_frames += 1
                    throttle(target_fps, now)
                frame_drop.put(total_frames_set)
                filter_frame_idx.value = -1 # Signal that video has ended

                print(f'Reading FPS: {total_frames / (monotonic() - read_start_time)}')

                filter_frames_pid.join()
                encode_frames_pid.join()
                read_queue_pid.join()

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

                average_energy = get_average_energy(energy_readings)
                logging.info(f'({start}, {end}, ({threshold}, {bitrate}), {average_energy}),')

            energy_pid.terminate()
            energy_pid.join()
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

def simulate_JH_day():
    JH_DAY = f'{os.path.dirname(__file__)}/../flashdrive/ground-truth-JH-full'

    # ecolens_jh_day_90 = [
    #     # start index, end index, threshold, bitrate
    #     (0, 1800, (0.01, 2400.0)),
    #     (1800, 1950, (0.01, 2400.0)),
    #     (1950, 3750, (0.02, 2400.0)),
    #     (3750, 3900, (0.02, 2400.0)),
    #     (3900, 5700, (0.02, 3000.0)),
    #     (5700, 5850, (0.02, 3000.0)),
    #     (5850, 7650, (0.01, 2400.0)),
    #     (7650, 7800, (0.01, 2400.0)),
    #     (7800, 9600, (0.02, 3000.0)),
    #     (9600, 9750, (0.02, 3000.0)),
    #     (9750, 11550, (0.02, 3000.0)),
    #     (11550, 11700, (0.02, 3000.0)),
    #     (11700, 13500, (0.02, 3000.0)),
    #     (13500, 13650, (0.02, 3000.0)),
    #     (13650, 15450, (0.02, 3000.0)),
    #     (15450, 15600, (0.02, 3000.0)),
    #     (15600, 17400, (0.01, 3000.0)),
    #     (17400, 17550, (0.01, 3000.0)),
    #     (17550, 18017, (0.02, 2400.0)),
    # ]
    # LOG = f'{os.path.dirname(__file__)}/ecolens-JH-day-0.9-energy.csv'
    # simulator = StreamSimulator(JH_DAY, LOG)
    # for _ in range(3):
    # simulator.simulate(ecolens_jh_day_90)
    # sleep(5)

    # noreprofile_jh_day_90 = [
    #     (0, 1800, (0.01, 2400.0)),
    #     (1800, 1950, (0.01, 2400.0)),
    #     (1950, 3750, (0.01, 2400.0)),
    #     (3750, 3900, (0.01, 2400.0)),
    #     (3900, 5700, (0.01, 2400.0)),
    #     (5700, 5850, (0.01, 2400.0)),
    #     (5850, 7650, (0.01, 2400.0)),
    #     (7650, 7800, (0.01, 2400.0)),
    #     (7800, 9600, (0.01, 2400.0)),
    #     (9600, 9750, (0.01, 2400.0)),
    #     (9750, 11550, (0.01, 2400.0)),
    #     (11550, 11700, (0.01, 2400.0)),
    #     (11700, 13500, (0.01, 2400.0)),
    #     (13500, 13650, (0.01, 2400.0)),
    #     (13650, 15450, (0.01, 2400.0)),
    #     (15450, 15600, (0.01, 2400.0)),
    #     (15600, 17400, (0.01, 2400.0)),
    #     (17400, 17550, (0.01, 2400.0)),
    #     (17550, 18017, (0.01, 2400.0)),
    # ]
    # LOG = f'{os.path.dirname(__file__)}/noreprofile-JH-day-0.9-energy.csv'
    # simulator = StreamSimulator(JH_DAY, LOG)
    # for _ in range(3):
    # simulator.simulate(noreprofile_jh_day_90)
    # sleep(5)

    reducto_jh_day_90 = [
        (0, 1800, (0.0016, 3000.0)),
        (1800, 1950, (0.0016, 3000.0)),
        (1950, 3750, (0.0016, 3000.0)),
        (3750, 3900, (0.0016, 3000.0)),
        (3900, 5700, (0.0016, 3000.0)),
        (5700, 5850, (0.0016, 3000.0)),
        (5850, 7650, (0.0016, 3000.0)),
        (7650, 7800, (0.0016, 3000.0)),
        (7800, 9600, (0.0016, 3000.0)),
        (9600, 9750, (0.0016, 3000.0)),
        (9750, 11550, (0.0016, 3000.0)),
        (11550, 11700, (0.0016, 3000.0)),
        (11700, 13500, (0.0016, 3000.0)),
        (13500, 13650, (0.0016, 3000.0)),
        (13650, 15450, (0.0016, 3000.0)),
        (15450, 15600, (0.0016, 3000.0)),
        (15600, 17400, (0.0016, 3000.0)),
        (17400, 17550, (0.0016, 3000.0)),
        (17550, 18017, (0.0016, 3000.0)),
    ]
    # LOG = f'{os.path.dirname(__file__)}/reducto-JH-day-0.9-energy-1.5.csv'
    # simulator = StreamSimulator(JH_DAY, LOG)
    # simulator.simulate(reducto_jh_day_90, processor=EdgeDiff)

    set_cpu_freq(2400000)
    sleep(1)

    LOG = f'{os.path.dirname(__file__)}/reducto-JH-day-0.9-energy-2.4.csv'
    simulator = StreamSimulator(JH_DAY, LOG)
    simulator.simulate(reducto_jh_day_90, processor=EdgeDiff)
    sleep(5)

def simulate_JH_night():
    JH_NIGHT = f'{os.path.dirname(__file__)}/../flashdrive/ground-truth-JH-night-full'

    # ecolens_JH_night_90 = [
    #     (0, 1800, (0.0, 400.0)),
    #     (1800, 1950, (0.0, 400.0)),
    #     (1950, 3750, (0.0, 400.0)),
    #     (3750, 3900, (0.0, 400.0)),
    #     (3900, 5700, (0.0, 3000.0)),
    #     (5700, 5850, (0.0, 3000.0)),
    #     (5850, 7650, (0.01, 2400.0)),
    #     (7650, 7800, (0.01, 2400.0)),
    #     (7800, 9600, (0.0, 3000.0)),
    #     (9600, 9750, (0.0, 3000.0)),
    #     (9750, 11550, (0.02, 400.0)),
    #     (11550, 11700, (0.02, 400.0)),
    #     (11700, 13500, (0.0, 3000.0)),
    #     (13500, 13650, (0.0, 3000.0)),
    #     (13650, 15450, (0.0, 1000.0)),
    #     (15450, 15600, (0.0, 1000.0)),
    #     (15600, 17400, (0.1, 3000.0)),
    #     (17400, 17550, (0.1, 3000.0)),
    #     (17550, 18030, (0.1, 100.0)),
    # ]

    # LOG = f'{os.path.dirname(__file__)}/ecolens-JH-night-0.9-energy.csv'
    # simulator = StreamSimulator(JH_NIGHT, LOG)
    # simulator.simulate(ecolens_JH_night_90)
    # sleep(5)

    # noreprofile_jh_night_90 = [
    #     (0, 1800, (0.0, 400.0)),
    #     (1800, 3600, (0.0, 400.0)),
    #     (3600, 5400, (0.0, 400.0)),
    #     (5400, 7200, (0.0, 400.0)),
    #     (7200, 9000, (0.0, 400.0)),
    #     (9000, 10800, (0.0, 400.0)),
    #     (10800, 12600, (0.0, 400.0)),
    #     (12600, 14400, (0.0, 400.0)),
    #     (14400, 16200, (0.0, 400.0)),
    #     (16200, 18000, (0.0, 400.0)),
    #     (18000, 18030, (0.0, 400.0)),
    # ]
    # LOG = f'{os.path.dirname(__file__)}/noreprofile-JH-night-0.9-energy.csv'
    # simulator = StreamSimulator(JH_NIGHT, LOG)
    # simulator.simulate(noreprofile_jh_night_90)
    # sleep(5)

    # baseline_jh_night = [
    #     (0, 1800, (0.00, 3000.0)),
    #     (1800, 1950, (0.00, 3000.0)),
    #     (1950, 3750, (0.00, 3000.0)),
    #     (3750, 3900, (0.00, 3000.0)),
    #     (3900, 5700, (0.00, 3000.0)),
    #     (5700, 5850, (0.00, 3000.0)),
    #     (5850, 7650, (0.00, 3000.0)),
    #     (7650, 7800, (0.00, 3000.0)),
    #     (7800, 9600, (0.00, 3000.0)),
    #     (9600, 9750, (0.00, 3000.0)),
    #     (9750, 11550, (0.00, 3000.0)),
    #     (11550, 11700, (0.00, 3000.0)),
    #     (11700, 13500, (0.00, 3000.0)),
    #     (13500, 13650, (0.00, 3000.0)),
    #     (13650, 15450, (0.00, 3000.0)),
    #     (15450, 15600, (0.00, 3000.0)),
    #     (15600, 17400, (0.00, 3000.0)),
    #     (17400, 17550, (0.00, 3000.0)),
    #     (17550, 18017, (0.00, 3000.0)),
    # ]

    # LOG = f'{os.path.dirname(__file__)}/baseline-JH-night-energy-1.5.csv'
    # simulator = StreamSimulator(JH_NIGHT, LOG)
    # simulator.simulate(baseline_jh_night)
    # sleep(5)

    # set_cpu_freq(2400000)
    # sleep(1)

    # LOG = f'{os.path.dirname(__file__)}/baseline-JH-night-energy-2.4.csv'
    # simulator = StreamSimulator(JH_NIGHT, LOG)
    # simulator.simulate(baseline_jh_night)

    reducto_jh_night_90 = [
        (0, 1800, (0.0, 3000)),
        (1800, 3600, (0.0, 3000)),
        (3600, 5400, (0.0, 3000)),
        (5400, 7200, (0.0, 3000)),
        (7200, 9000, (0.0, 3000)),
        (9000, 10800, (0.0, 3000)),
        (10800, 12600, (0.0, 3000)),
        (12600, 14400, (0.0, 3000)),
        (14400, 16200, (0.0, 3000)),
        (16200, 18000, (0.0, 3000)),
        (18000, 18030, (0.0, 3000)),
    ]

    set_cpu_freq(2400000)
    sleep(1)

    LOG = f'{os.path.dirname(__file__)}/reducto-JH-night-0.9-energy-2.4.csv'
    simulator = StreamSimulator(JH_NIGHT, LOG)
    simulator.simulate(reducto_jh_night_90, processor=EdgeDiff)


def simulate_Alma():
    ALMA = f'{os.path.dirname(__file__)}/../flashdrive/ground-truth-Alma-full'

    # ecolens_Alma_90 = [
    #     (0, 1800, (0.02, 2100.0)),
    #     (1800, 1950, (0.02, 2100.0)),
    #     (1950, 3750, (0.01, 1600.0)),
    #     (3750, 3900, (0.01, 1600.0)),
    #     (3900, 5700, (0.04, 3000.0)),
    #     (5700, 5850, (0.04, 3000.0)),
    #     (5850, 7650, (0.08, 1600.0)),
    #     (7650, 7800, (0.08, 1600.0)),
    #     (7800, 9600, (0.01, 3000.0)),
    #     (9600, 9750, (0.01, 3000.0)),
    #     (9750, 11550, (0.03, 1600.0)),
    #     (11550, 11700, (0.03, 1600.0)),
    #     (11700, 13500, (0.06, 1900.0)),
    #     (13500, 13650, (0.06, 1900.0)),
    #     (13650, 15450, (0.08, 1900.0)),
    #     (15450, 15600, (0.08, 1900.0)),
    #     (15600, 17400, (0.09, 2400.0)),
    #     (17400, 17550, (0.09, 2400.0)),
    #     (17550, 18002, (0.08, 1900.0)),
    # ]

    # sleep(5)
    # LOG = f'{os.path.dirname(__file__)}/ecolens-Alma-0.9-energy.csv'
    # simulator = StreamSimulator(ALMA, LOG)
    # for _ in range(2):
    #     simulator.simulate(ecolens_Alma_90)
    #     sleep(5)

    # noreprofile_Alma_90 = [
    #     (0, 1800, (0.02, 2100.0)),
    #     (1800, 3600, (0.02, 2100.0)),
    #     (3600, 5400, (0.02, 2100.0)),
    #     (5400, 7200, (0.02, 2100.0)),
    #     (7200, 9000, (0.02, 2100.0)),
    #     (9000, 10800, (0.02, 2100.0)),
    #     (10800, 12600, (0.02, 2100.0)),
    #     (12600, 14400, (0.02, 2100.0)),
    #     (14400, 16200, (0.02, 2100.0)),
    #     (16200, 18000, (0.02, 2100.0)),
    #     (18000, 18002, (0.02, 2100.0)),
    # ]

    # LOG = f'{os.path.dirname(__file__)}/noreprofile-Alma-0.9-energy.csv'
    # simulator = StreamSimulator(ALMA, LOG)
    # for _ in range(2):
    #     simulator.simulate(noreprofile_Alma_90)
    #     sleep(5)

    # baseline_Alma_1_5 = [
    #     (0, 1800, (0.00, 3000.0)),
    #     (1800, 3600, (0.00, 3000.0)),
    #     (3600, 5400, (0.00, 3000.0)),
    #     (5400, 7200, (0.00, 3000.0)),
    #     (7200, 9000, (0.00, 3000.0)),
    #     (9000, 10800, (0.00, 3000.0)),
    #     (10800, 12600, (0.00, 3000.0)),
    #     (12600, 14400, (0.00, 3000.0)),
    #     (14400, 16200, (0.00, 3000.0)),
    #     (16200, 18000, (0.00, 3000.0)),
    #     (18000, 18002, (0.00, 3000.0)),
    # ]

    # LOG = f'{os.path.dirname(__file__)}/baseline-Alma-energy-1.5.csv'
    # simulator = StreamSimulator(ALMA, LOG)
    # for _ in range(2):
    #     simulator.simulate(baseline_Alma_1_5)
    #     sleep(5)

    # baseline_Alma_2_4 = [
    #     (0, 1800, (0.00, 3000.0)),
    #     (1800, 3600, (0.00, 3000.0)),
    #     (3600, 5400, (0.00, 3000.0)),
    #     (5400, 7200, (0.00, 3000.0)),
    #     (7200, 9000, (0.00, 3000.0)),
    #     (9000, 10800, (0.00, 3000.0)),
    #     (10800, 12600, (0.00, 3000.0)),
    #     (12600, 14400, (0.00, 3000.0)),
    #     (14400, 16200, (0.00, 3000.0)),
    #     (16200, 18000, (0.00, 3000.0)),
    #     (18000, 18002, (0.00, 3000.0)),
    # ]

    # set_cpu_freq(2400000)
    # sleep(1)

    # LOG = f'{os.path.dirname(__file__)}/baseline-Alma-energy-2.4.csv'
    # simulator = StreamSimulator(ALMA, LOG)
    # simulator.simulate(baseline_Alma_2_4)
    # sleep(5)

    reducto_alma_90 = [
        (0, 1800, (0.004, 3000)),
        (1800, 3600, (0.004, 3000)),
        (3600, 5400, (0.004, 3000)),
        (5400, 7200, (0.004, 3000)),
        (7200, 9000, (0.004, 3000)),
        (9000, 10800, (0.004, 3000)),
        (10800, 12600, (0.004, 3000)),
        (12600, 14400, (0.004, 3000)),
        (14400, 16200, (0.004, 3000)),
        (16200, 18000, (0.004, 3000)),
        (18000, 18002, (0.004, 3000)),
    ]

    set_cpu_freq(2400000)
    sleep(1)

    LOG = f'{os.path.dirname(__file__)}/reducto-Alma-0.9-energy-2.4.csv'
    simulator = StreamSimulator(ALMA, LOG)
    simulator.simulate(reducto_alma_90, processor=EdgeDiff)

if __name__ == '__main__':
    set_cpu_freq(1500000)

    # ========= JH Day Video ========
    simulate_JH_day()

    # ========= JH Night Video ========
    # simulate_JH_night()

    # ========= Alma Video ========
    # simulate_Alma()