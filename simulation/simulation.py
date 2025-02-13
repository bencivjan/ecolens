import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from trieste.experimental.plotting import plot_mobo_points_in_obj_space
from evaluator import Evaluator
from utils import remove_tensor_duplicates
from bayesian_opt import VideoBayesianOpt

class EcoLensSimulation:
    def __init__(self, frame_dir, energy_profile, accuracy_profile, explore_time=10, exploit_time=110, log_file=None) -> None:
        self.search_space = []
        for thresh in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
            for br in [100, 400, 700, 1000, 1300, 1600, 1900, 2100, 2400, 2700, 3000]:
                self.search_space.append([thresh, br])

        self.batch_size = 4
        self.evaluator = Evaluator(frame_dir, model_path=os.path.join(os.path.dirname(__file__), '../yolov8x.pt'))
        self.energy_profile = energy_profile
        self.accuracy_profile = accuracy_profile
        self.total_frames = len(os.listdir(frame_dir))

        self.explore_time = explore_time
        self.exploit_time = exploit_time
        self.fps = 30
        
        self.log_file = log_file
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(message)s')
            logging.info('Frame Range Start, Frame Range End, Configuration, Running Accuracy, Round Accuracy')

    def run_verify_round(self, mbo: VideoBayesianOpt, frame_range: range) -> float:
        print(f'Verify round for configuration: {self.config}')
        accuracy = self.evaluator.evaluate_configs([self.config], frame_range.start, frame_range.stop)[0]
        mbo.tell_observations(tf.constant([self.config]), tf.stack([mbo.get_config_profile_energy([self.config]), tf.constant([accuracy])], axis=1))
        print(f'Accuracy: {accuracy}')
        return accuracy

    def run_explore_round(self, mbo: VideoBayesianOpt, frame_range: range, iterations: int):
        for _ in range(iterations):
            prev_dataset = mbo.ask_tell.to_result().try_get_final_dataset()
            # Since the MBO dataset has negative accuracy, we must negate to make them positive
            best_points = mbo.get_n_best_configs(6, prev_dataset.query_points, tf.stack([prev_dataset.observations[:, 0], -prev_dataset.observations[:, 1]], axis=1))[0]
            mbo_suggestions = mbo.ask_for_suggestions()
            print(f'Best Points: {best_points}')
            print(f'Suggestions: {mbo_suggestions}')

            queries = remove_tensor_duplicates(tf.concat([best_points, mbo_suggestions], axis=0))
            
            new_accuracies = self.evaluator.evaluate_configs(queries, frame_range.start, frame_range.stop)
            print(f'New Accuracies: {new_accuracies}')
            est_energies = mbo.get_config_profile_energy(queries)
            print(f'Estimated Energies: {est_energies}')
            mbo.tell_observations(queries, tf.stack([est_energies, new_accuracies], axis=1))

        # For debugging
        # mbo.ask_for_suggestions()
        # dataset = mbo.ask_tell.to_result().try_get_final_dataset()
        # plot_mobo_points_in_obj_space(dataset.observations, num_init=self.num_init_points, xlabel="Energy", ylabel="Accuracy")
        # plt.show()

    def run_exploit_round(self, frame_range: range) -> float:
        print(f'Exploit round for configuration: {self.config}')
        accuracy = self.evaluator.evaluate_configs([self.config], frame_range.start, frame_range.stop)[0]
        print(f'Accuracy: {accuracy}')
        return accuracy
    
    def select_configuration(self, mbo: VideoBayesianOpt, target_accuracy: float):
        recommended_config = mbo.get_recommended_configuration(target_accuracy)
        if recommended_config is not None:
            self.config = recommended_config

    def run(self, target_accuracy: float) -> None:
        mbo = VideoBayesianOpt(self.search_space, self.batch_size, target_accuracy=target_accuracy, data_window_size=20)
        mbo.build_dataset_from_profile(self.energy_profile, self.accuracy_profile)

        mbo.build_stacked_independent_objectives_model()

        self.config = mbo.get_recommended_configuration(target_accuracy)
        if not self.config:
            print("No valid configurations found")
            self.config = (0.0, 3000)

        self.num_init_points = 0
        print(f'Number of initial points: {self.num_init_points}')

        i = 0
        explore = False
        cur_range = range(0, self.exploit_time * self.fps)

        running_accuracy = 0

        while i < self.total_frames:
            print(f'Current range: {cur_range.start / self.fps}s to {cur_range.stop / self.fps}s')
            if not explore:
                configuration_acc = self.run_exploit_round(cur_range)
                running_accuracy += configuration_acc * (cur_range.stop - cur_range.start)
                prev_range, prev_config = cur_range, self.config
                cur_range = range(cur_range.stop, min(cur_range.stop + self.explore_time * self.fps, self.total_frames))
                explore = True
            else: # explore
                configuration_acc = self.run_verify_round(mbo, cur_range)
                running_accuracy += configuration_acc * (cur_range.stop - cur_range.start)

                # TODO: Iron out how many explore iterations to run
                self.run_explore_round(mbo, cur_range, iterations=1)

                prev_range, prev_config = cur_range, self.config # Save previous configuration for logging
                self.select_configuration(mbo, target_accuracy)
                cur_range = range(cur_range.stop, min(cur_range.stop + self.exploit_time * self.fps, self.total_frames))
                explore = False
            i = cur_range.start

            if self.log_file:
                avg_accuracy = running_accuracy / (prev_range.stop)
                logging.info(f'({prev_range.start}, {prev_range.stop}, {prev_config}, {avg_accuracy}, {configuration_acc}),')

        print(f'Final accuracy: {running_accuracy / self.total_frames}')

        dataset = mbo.ask_tell.to_result().try_get_final_dataset()
        logging.info(f'Final Dataset: {dataset}')
        print(f'Final Dataset: {dataset}')
        plot_mobo_points_in_obj_space(dataset.observations, num_init=self.num_init_points, xlabel="Energy", ylabel="Accuracy")
        plt.show()

class NoReprofileSimulation():
    def __init__(self, frame_dir, energy_profile, accuracy_profile, log_file=None) -> None:
        self.frame_dir = frame_dir
        self.energy_profile = energy_profile
        self.accuracy_profile = accuracy_profile
        self.fps = 30
        self.total_frames = len(os.listdir(frame_dir))

        self.evaluator = Evaluator(frame_dir, model_path=os.path.join(os.path.dirname(__file__), '../yolov8x.pt'))

        self.log_file = log_file
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(message)s')
            logging.info('Frame Range Start, Frame Range End, Configuration, Running Accuracy, Round Accuracy')

    def get_recommended_configuration(self, profiling_df, target_accuracy: float):
        configs = np.array([[x, y] for x, y in zip(profiling_df['Threshold'], profiling_df['Frame Bitrate'])])
        observations = np.array([[x, y] for x, y in zip(profiling_df['Avg Energy'], profiling_df['Average IoU'])])

        mask = observations[:, 1] >= target_accuracy

        configs, observations = configs[mask], observations[mask]
        sorted_idcs = observations[:,0].argsort()
        
        return tuple(configs[sorted_idcs][0])

    def run(self, target_accuracy: float) -> None:
        # Get best configuration from profiling
        profiling_df = VideoBayesianOpt.read_profiling_data(self.energy_profile, self.accuracy_profile)
        best_config = self.get_recommended_configuration(profiling_df, target_accuracy)
        print(f'Best configuration: {best_config}')

        i = 0
        time_increment = 60
        cur_range = range(0, 60 * self.fps)
        running_accuracy = 0

        while i < self.total_frames:
            print(f'Current range: {cur_range.start / self.fps}s to {cur_range.stop / self.fps}s')

            configuration_acc = self.evaluator.evaluate_configs([best_config], cur_range.start, cur_range.stop)[0]
            running_accuracy += configuration_acc * (cur_range.stop - cur_range.start)
            prev_range = cur_range
            cur_range = range(cur_range.stop, min(cur_range.stop + time_increment * self.fps, self.total_frames))

            i = cur_range.start

            if self.log_file:
                avg_accuracy = running_accuracy / (prev_range.stop)
                logging.info(f'({prev_range.start}, {prev_range.stop}, {best_config}, {avg_accuracy}, {configuration_acc}),')

        print(f'Final accuracy: {running_accuracy / self.total_frames}')

class ConfigSimulation():
    def __init__(self, frame_dir, log_file=None) -> None:
        self.frame_dir = frame_dir
        self.fps = 30
        self.total_frames = len(os.listdir(frame_dir))

        self.evaluator = Evaluator(frame_dir, model_path=os.path.join(os.path.dirname(__file__), '../yolov8x.pt'))

        self.log_file = log_file
        if self.log_file:
            logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(message)s')
            logging.info('Frame Range Start, Frame Range End, Running Accuracy, Round Accuracy, Configuration')

    def run(self, config):
        i = 0
        time_increment = 60
        cur_range = range(0, 60 * self.fps)
        running_accuracy = 0

        while i < self.total_frames:
            print(f'Current range: {cur_range.start / self.fps}s to {cur_range.stop / self.fps}s')

            configuration_acc = self.evaluator.evaluate_configs([config], cur_range.start, cur_range.stop)[0]
            running_accuracy += configuration_acc * (cur_range.stop - cur_range.start)
            prev_range = cur_range
            cur_range = range(cur_range.stop, min(cur_range.stop + time_increment * self.fps, self.total_frames))

            i = cur_range.start

            if self.log_file:
                avg_accuracy = running_accuracy / (prev_range.stop)
                logging.info(f'({prev_range.start}, {prev_range.stop}, {config}, {avg_accuracy}, {configuration_acc}),')

        print(f'Final accuracy: {running_accuracy / self.total_frames}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('simulation', type=str, default='ecolens', help='Simulation to run')
    parser.add_argument('--frame-dir', '-f', default='../filter-images/ground-truth-JH-full', help='Directory containing frames')
    parser.add_argument('--energy-profile', '-e', default="../viz/energy-JH-1.csv", help='Energy profile')
    parser.add_argument('--accuracy-profile', '-a', default="../viz/accuracy-JH-1.csv", help='Accuracy profile')
    parser.add_argument('--target-accuracy', '-t', default=0.90, type=float, help='Target accuracy')
    parser.add_argument('--explore-time', '-x', default=5, type=int, help='Explore time in seconds')
    parser.add_argument('--exploit-time', '-y', default=60, type=int, help='Exploit time in seconds')
    parser.add_argument('--log-file', '-l', default=None, type=str, help='Log file')

    args = parser.parse_args()

    if args.simulation == 'ecolens':
        simulation = EcoLensSimulation(frame_dir=args.frame_dir,
                                energy_profile=args.energy_profile,
                                accuracy_profile=args.accuracy_profile,
                                exploit_time=args.exploit_time,
                                explore_time=args.explore_time,
                                log_file=args.log_file)
        simulation.run(target_accuracy=args.target_accuracy)
    
    elif args.simulation == 'noreprofile':
        simulation = NoReprofileSimulation(frame_dir=args.frame_dir,
                                energy_profile=args.energy_profile,
                                accuracy_profile=args.accuracy_profile,
                                log_file=args.log_file)
        simulation.run(target_accuracy=args.target_accuracy)
    elif args.simulation == 'baseline':
        simulation = ConfigSimulation(frame_dir=args.frame_dir, log_file=args.log_file)
        simulation.run(config=(0.0, 3000))
    else:
        raise NotImplementedError(f'Simulation {args.simulation} not implemented')