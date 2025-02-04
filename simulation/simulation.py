import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trieste.acquisition.function import ExpectedHypervolumeImprovement, Fantasizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.multi_objective import non_dominated
from trieste.data import Dataset
from trieste.models import TrainableModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import DiscreteSearchSpace
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.experimental.plotting import plot_mobo_points_in_obj_space
from evaluator import Evaluator
from utils import remove_tensor_duplicates
from collections import deque

# Implement multi-objective Bayesian optimization for online video configuration updates

class VideoBayesianOpt:

    def __init__(self, search_space, batch_size, target_accuracy=0.90, data_window_size=20):

        self.num_objectives = 2
        self.search_space = DiscreteSearchSpace(tf.constant(search_space, dtype=tf.float64))
        
        fant_ehvi = Fantasizer(ExpectedHypervolumeImprovement())
        self.rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=fant_ehvi, num_query_points=batch_size)

        self.target_accuracy = target_accuracy

        self.data_window_size = data_window_size
        self.data_window = deque(maxlen=self.data_window_size)
        self.query2observation = {}

    @staticmethod
    def read_profiling_data(energy_file, accuracy_file):
        accuracy_df = pd.read_csv(accuracy_file)
        energy_df = pd.read_csv(energy_file)
        merged_df = pd.merge(accuracy_df, energy_df, on=["Frequency", "Filter", "Threshold", "Frame Bitrate"])
        merged_df = merged_df.drop(columns=['FPS', 'Start Time', 'End Time'])
        merged_df = merged_df[(merged_df['Frequency'] == 1.5) & (merged_df['Filter'] == 'pixel')]
        merged_df = merged_df.drop(columns=['Frequency', 'Filter'])

        # Normalize the 'Average IoU' column
        merged_df['Average IoU'] = merged_df['Average IoU'] / merged_df['Average IoU'].max()

        return merged_df
    
    def get_config_profile_energy(self, configs):
        if self.profiling_df is None:
            raise ValueError("No Profiling Data Has Been Read")

        energies = []
        for config in configs:
            threshold, bitrate = config
            energy = self.profiling_df[(self.profiling_df['Threshold'] == float(threshold)) & (self.profiling_df['Frame Bitrate'] == int(bitrate))]['Avg Energy'].values[0]
            energies.append(energy)
        return tf.convert_to_tensor(energies)

    def get_n_best_configs(self, n, query_points: tf.Tensor, observations: tf.Tensor):
        print(query_points, observations)
        
        accuracy_mask = observations[:, 1] >= self.target_accuracy
        query_points = query_points[accuracy_mask]
        observations = observations[accuracy_mask]
        print(f"Selecting best points from: {query_points, observations}")

        # non_dominated returns pareto of minimized objectives, so negate accuracy
        _, mask = non_dominated(tf.stack([observations[:,0], -observations[:,1]], axis=1))
        query_points = query_points[mask].numpy()
        observations = observations[mask].numpy()

        sorted_idcs = np.argsort(observations[:, 0])
        return tf.constant(query_points[sorted_idcs][:n]), tf.constant(observations[sorted_idcs][:n])

    def build_dataset_from_profile(self, energy_file, accuracy_file):
        self.profiling_df = self.read_profiling_data(energy_file, accuracy_file)
        query_points = tf.constant([[x, y] for x, y in zip(self.profiling_df['Threshold'], self.profiling_df['Frame Bitrate'])], dtype=tf.float64)
        observations = tf.constant([[x, y] for x, y in zip(self.profiling_df['Avg Energy'], self.profiling_df['Average IoU'])], dtype=tf.float64)

        best_configs, best_observations = self.get_n_best_configs(5, query_points, observations)
        print(f'Best Configs: {best_configs}')
        print(f'Best Observations: {best_observations}')

        for config, observation in zip(best_configs, best_observations):
            config, observation = tuple(config.numpy()), tuple(observation.numpy())
            self.data_window.append(config)
            self.query2observation[config] = observation

    def build_stacked_independent_objectives_model(self) -> TrainableModelStack:
        data = Dataset(tf.stack(self.data_window), tf.convert_to_tensor([[self.query2observation[c][0], -self.query2observation[c][1]] for c in self.data_window]))

        gprs = []
        for idx in range(self.num_objectives):
            single_obj_data = Dataset(
                data.query_points, tf.gather(data.observations, [idx], axis=1)
            )
            gpr = build_gpr(single_obj_data, self.search_space)
            gprs.append((GaussianProcessRegression(gpr), 1))

        self.model = TrainableModelStack(*gprs)
        self.ask_tell = AskTellOptimizer(self.search_space, data, self.model, acquisition_rule=self.rule, fit_model=True)

    def ask_for_suggestions(self) -> Dataset:
        print(f'Data Window: {self.data_window}')
        data = Dataset(tf.stack(self.data_window), tf.convert_to_tensor([[self.query2observation[c][0], -self.query2observation[c][1]] for c in self.data_window]))
        self.ask_tell = AskTellOptimizer(self.search_space, data, self.model, acquisition_rule=self.rule, fit_model=True)
        return self.ask_tell.ask()
    
    def tell_observations(self, configs, observations) -> None:
        for config, observation in zip(configs, observations):
            config, observation = tuple(config.numpy()), tuple(observation.numpy())
            self.data_window.append(config)
            self.query2observation[config] = observation

        # Remove duplicates
        self.data_window = deque(set(self.data_window), maxlen=self.data_window_size)
    
    def get_recommended_configuration(self, target_accuracy):
        configs = np.array(self.data_window)
        observations = np.array([self.query2observation[c] for c in self.data_window])

        valid_indices = observations[:, 1] >= target_accuracy
        configs, observations = configs[valid_indices], observations[valid_indices]

        sorted_valid_idcs = observations[:,0].argsort()
        valid_configs = configs[sorted_valid_idcs]

        if valid_configs.size == 0:
            print("No valid configurations found")
            return None
        
        return tuple(valid_configs[0])


class Simulation:
    def __init__(self, frame_dir, energy_profile, accuracy_profile, explore_time=10, exploit_time=110) -> None:
        self.search_space = []
        for thresh in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
            for br in [100, 400, 700, 1000, 1300, 1600, 1900, 2100, 2400, 2700, 3000]:
                self.search_space.append([thresh, br])

        self.batch_size = 4
        self.evaluator = Evaluator(frame_dir)
        self.energy_profile = energy_profile
        self.accuracy_profile = accuracy_profile
        self.total_frames = len(os.listdir(frame_dir))

        self.explore_time = explore_time
        self.exploit_time = exploit_time
        self.fps = 30

        self.config = (0.99, 3000)

    def run_verify_round(self, frame_range: range) -> float:
        print(f'Verify round for configuration: {self.config}')
        accuracy = self.evaluator.evaluate_configs([self.config], frame_range.start, frame_range.stop)[0]
        print(f'Accuracy: {accuracy}')
        return accuracy

    def run_explore_round(self, mbo: VideoBayesianOpt, frame_range: range, iterations: int):
        for _ in range(iterations):
            prev_dataset = mbo.ask_tell.to_result().try_get_final_dataset()
            # Since the MBO dataset has negative accuracy, we must negate to make them positive
            best_points = mbo.get_n_best_configs(5, prev_dataset.query_points, tf.stack([prev_dataset.observations[:, 0], -prev_dataset.observations[:, 1]], axis=1))[0]

            mbo_suggestions = remove_tensor_duplicates(mbo.ask_for_suggestions())
            queries = tf.concat([best_points, mbo_suggestions], axis=0)
            print(f'Best Points: {best_points}')
            print(f'Suggestions: {mbo_suggestions}')
            new_accuracies = self.evaluator.evaluate_configs(queries, frame_range.start, frame_range.stop)
            print(new_accuracies)
            est_energies = mbo.get_config_profile_energy(queries)
            print(est_energies)
            mbo.tell_observations(queries, tf.stack([est_energies, new_accuracies], axis=1))

        # For debugging
        mbo.ask_for_suggestions()
        dataset = mbo.ask_tell.to_result().try_get_final_dataset()
        plot_mobo_points_in_obj_space(dataset.observations, num_init=self.num_init_points, xlabel="Energy", ylabel="Accuracy")
        plt.show()

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
        mbo = VideoBayesianOpt(self.search_space, self.batch_size, target_accuracy=0.90, data_window_size=20)
        mbo.build_dataset_from_profile(self.energy_profile, self.accuracy_profile)

        mbo.build_stacked_independent_objectives_model()

        self.num_init_points = 0
        print(f'Number of initial points: {self.num_init_points}')

        i = 0
        explore = False
        cur_range = range(0, self.exploit_time * self.fps)

        while i < self.total_frames:
            print(f'Current range: {cur_range.start / self.fps}s to {cur_range.stop / self.fps}s')
            if not explore:
                # self.run_exploit_round(cur_range)
                cur_range = range(cur_range.stop, cur_range.stop + self.explore_time * self.fps)
                explore = True
            else: # explore
                configuration_acc = self.run_verify_round(cur_range)

                # TODO: Iron out how many explore iterations to run
                self.run_explore_round(mbo, cur_range, iterations=2)
                self.select_configuration(mbo, target_accuracy)
                cur_range = range(cur_range.stop, cur_range.stop + self.exploit_time * self.fps)
                explore = False
            i = cur_range.stop

        dataset = mbo.ask_tell.to_result().try_get_final_dataset()
        print(dataset)
        plot_mobo_points_in_obj_space(dataset.observations, num_init=self.num_init_points, xlabel="Energy", ylabel="Accuracy")
        plt.show()

if __name__ == "__main__":
    simulation = Simulation(frame_dir='../filter-images/ground-truth-JH-full',
                            energy_profile="../viz/energy-JH-1.csv",
                            accuracy_profile="../viz/accuracy-JH-1.csv",
                            explore_time=5)
    simulation.run(target_accuracy=0.90)