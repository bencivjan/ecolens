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
from itertools import islice

# Implement multi-objective Bayesian optimization for online video configuration updates
class VideoBayesianOpt:

    def __init__(self, search_space, batch_size, target_accuracy=0.90, data_window_size=20, round_size=10):

        self.num_objectives = 2
        self.search_space = DiscreteSearchSpace(tf.constant(search_space, dtype=tf.float64))
        
        fant_ehvi = Fantasizer(ExpectedHypervolumeImprovement())
        self.rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=fant_ehvi, num_query_points=batch_size)

        self.target_accuracy = target_accuracy

        self.data_window_size = data_window_size
        self.round_size = round_size
        self.data_window = deque(maxlen=self.data_window_size)
        self.query2observation = {}

    @staticmethod
    def read_profiling_data(energy_file, accuracy_file):
        accuracy_df = pd.read_csv(accuracy_file)
        energy_df = pd.read_csv(energy_file)
        merged_df = pd.merge(accuracy_df, energy_df, on=["Frequency", "Filter", "Threshold", "Frame Bitrate"])

        # Normalize the 'Average IoU' column
        # TODO: Potentially remove this
        # merged_df['Average IoU'] = merged_df['Average IoU'] / merged_df['Average IoU'].max()

        merged_df = merged_df.drop(columns=['FPS', 'Start Time', 'End Time'])
        merged_df = merged_df[(merged_df['Frequency'] == 1.5) & (merged_df['Filter'] == 'pixel')]
        merged_df = merged_df.drop(columns=['Frequency', 'Filter'])

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
        query_points, observations = query_points.numpy(), observations.numpy()

        accuracy_mask = observations[:, 1] >= self.target_accuracy

        # If no profiled points meet the target accuracy, just build dataset disregarding accuracy
        if tf.reduce_any(accuracy_mask) == True:
            query_points = query_points[accuracy_mask]
            observations = observations[accuracy_mask]
        print(f"Selecting best points from: {query_points, observations}")

        # non_dominated returns pareto of minimized objectives, so negate accuracy
        _, mask = non_dominated(tf.stack([observations[:,0], -observations[:,1]], axis=1))
        query_points_pareto = query_points[mask]
        print(f"Selected Pareto: {query_points_pareto}")
        observations_pareto = observations[mask]

        remaining_query_points = query_points[~mask]
        remaining_observations = observations[~mask]

        if len(query_points_pareto) < n:
            sorted_energy_idcs = np.argsort(remaining_observations[:, 0])
            sorted_query_points = remaining_query_points[sorted_energy_idcs]
            sorted_observations = remaining_observations[sorted_energy_idcs]

            # TODO: Make sure there aren't duplicates
            best_query_points = np.concatenate([query_points_pareto, sorted_query_points[:n - len(query_points_pareto)]], axis=0)
            best_observations = np.concatenate([observations_pareto, sorted_observations[:n - len(observations_pareto)]], axis=0)
        else:
            sorted_pareto_energy_idcs = np.argsort(observations_pareto[:, 0])
            best_query_points = query_points_pareto[sorted_pareto_energy_idcs][:n]
            best_observations = observations_pareto[sorted_pareto_energy_idcs][:n]

        return tf.convert_to_tensor(best_query_points), tf.convert_to_tensor(best_observations)

    def build_dataset_from_profile(self, energy_file, accuracy_file):
        self.profiling_df = self.read_profiling_data(energy_file, accuracy_file)
        query_points = tf.constant([[x, y] for x, y in zip(self.profiling_df['Threshold'], self.profiling_df['Frame Bitrate'])], dtype=tf.float64)
        observations = tf.constant([[x, y] for x, y in zip(self.profiling_df['Avg Energy'], self.profiling_df['Average IoU'])], dtype=tf.float64)

        best_configs, best_observations = self.get_n_best_configs(self.round_size, query_points, observations)
        # best_configs, best_observations = self.get_n_best_configs(self.data_window_size, query_points, observations)
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
    
    def tell_observations(self, configs: tf.Tensor, observations: tf.Tensor) -> None:
        for config, observation in zip(configs, observations):
            config, observation = tuple(config.numpy()), tuple(observation.numpy())
            self.data_window.append(config)
            self.query2observation[config] = observation

        # Remove duplicates in an order-preserving way
        seen = set()
        result = deque(maxlen=self.data_window_size)
        for item in reversed(self.data_window):
            if item not in seen:
                seen.add(item)
                result.appendleft(item)
        self.data_window = result

    def get_recommended_configuration(self, target_accuracy):
        # Only select from observed queries in the last round
        last_round = list(self.data_window)[-self.round_size:]
        configs = np.array(last_round)
        observations = np.array([self.query2observation[c] for c in last_round])

        valid_indices = observations[:, 1] >= target_accuracy
        configs, observations = configs[valid_indices], observations[valid_indices]

        sorted_valid_idcs = observations[:,0].argsort()
        valid_configs = configs[sorted_valid_idcs]

        if valid_configs.size == 0:
            print("No valid configurations found")
            return None
        
        return tuple(valid_configs[0])