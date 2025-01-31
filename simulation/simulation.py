import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import trieste.acquisition
from trieste.acquisition.function import ExpectedHypervolumeImprovement, Fantasizer, ExpectedConstrainedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.models import TrainableModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import DiscreteSearchSpace, Box
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.experimental.plotting import plot_mobo_points_in_obj_space
from evaluator import Evaluator
from utils import remove_tensor_duplicates

# Implement multi-objective Bayesian optimization for online video configuration updates

class VideoBayesianOpt:

    def __init__(self, search_space, batch_size):

        self.num_objectives = 2
        self.search_space = DiscreteSearchSpace(tf.constant(search_space, dtype=tf.float64))

        self.OBJECTIVE = "OBJECTIVE"
        self.CONSTRAINT = "CONSTRAINT"
        
        pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=0.75)
        fant_ehvi = Fantasizer(ExpectedConstrainedHypervolumeImprovement(self.OBJECTIVE, pof.using(self.CONSTRAINT)))
        self.rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=fant_ehvi, num_query_points=batch_size)

        self.explore_history = set()
        self.observation_history = []

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

    def build_dataset_from_profile(self, energy_file, accuracy_file):
        self.profiling_df = self.read_profiling_data(energy_file, accuracy_file)
        query_points = tf.constant([[x, y] for x, y in zip(self.profiling_df['Threshold'], self.profiling_df['Frame Bitrate'])], dtype=tf.float64)
        observations = tf.constant([[x, -y] for x, y in zip(self.profiling_df['Avg Energy'], self.profiling_df['Average IoU'])], dtype=tf.float64)
        self.dataset = Dataset(query_points, observations)

    def build_stacked_independent_objectives_model(self, data: Dataset) -> TrainableModelStack:
        gprs = []
        for idx in range(self.num_objectives):
            single_obj_data = Dataset(
                data.query_points, tf.gather(data.observations, [idx], axis=1)
            )
            gpr = build_gpr(single_obj_data, self.search_space, likelihood_variance=0.02)
            gprs.append((GaussianProcessRegression(gpr), 1))

        objective_model = TrainableModelStack(*gprs)

        print(data)

        constraint_observations = tf.expand_dims(tf.where(data.observations[:,1] <= -0.9, -data.observations[:,1] - 0.9, (0.9 - data.observations[:,1]) ** 2), axis=1)
        print(constraint_observations)
        constraint_data = Dataset(data.query_points, constraint_observations)
        print(constraint_data)
        gpflow_model = build_gpr(constraint_data, self.search_space)
        constraint_model = GaussianProcessRegression(gpflow_model)
        
        self.models = {self.OBJECTIVE: objective_model, self.CONSTRAINT: constraint_model}
        labeled_data = {self.OBJECTIVE: data, self.CONSTRAINT: constraint_data}

        self.ask_tell = AskTellOptimizer(self.search_space, labeled_data, self.models, acquisition_rule=self.rule, fit_model=True)

    def ask_for_suggestions(self) -> Dataset:
        suggestions = self.ask_tell.ask()
        final_suggestions = []
        for s in suggestions:
            suggestion_tup = (float(s[0].numpy()), int(s[1].numpy()))
            if suggestion_tup not in self.explore_history:
                final_suggestions.append(s)
                self.explore_history.add(suggestion_tup)
        return tf.stack(final_suggestions)
    
    def tell_observations(self, configs, observations) -> None:
        if configs.shape[0] == 0 or observations.shape[0] == 0:
            return
        for config, observation in zip(configs, observations):
            self.observation_history.append((config, observation))

        x_values = observations[:, 0]
        y_values = -observations[:, 1]
        observations = tf.stack([x_values, y_values], axis=1)
        objective_dataset = Dataset(configs, observations)

        constraint_observations = tf.expand_dims(tf.where(objective_dataset.observations[:,1] <= -0.9, -objective_dataset.observations[:,1] - 0.9, (0.9 - objective_dataset.observations[:,1]) ** 2), axis=1)
        constraint_dataset = Dataset(objective_dataset.query_points, constraint_observations)
        
        self.ask_tell.tell({self.OBJECTIVE: objective_dataset, self.CONSTRAINT: constraint_dataset})

    # def get_dataset(self):
    #     return self.ask_tell.to_result().try_get_final_dataset()
    
    def get_recommended_configuration(self, target_accuracy):
        dataset = np.array(self.observation_history)
        print(dataset)
        print(dataset.shape)
        dataset = dataset[dataset[:,1,1].argsort()[::-1]]
        valid_indices = dataset[:,1,1] >= target_accuracy
        valid_configs = dataset[valid_indices,0]

        if valid_configs.size == 0:
            print("No valid configurations found")
            return None
        
        return tuple(valid_configs[0])

        # dataset = self.ask_tell.to_result().try_get_final_dataset()
        # query_points = dataset.query_points.numpy()  # (N, 2) -> (thresh, bitrate)
        # observations = dataset.observations.numpy()  # (N, 2) -> (energy, accuracy)
        
        # # Filter observations where accuracy meets or exceeds the target
        # print(observations)
        # valid_indices = -observations[:, 1] >= target_accuracy
        # valid_observations = observations[valid_indices]
        # valid_query_points = query_points[valid_indices]

        # if valid_observations.size == 0:
        #     return None
        
        # # Sort by accuracy, high to low
        # sorted_indices = np.argsort(valid_observations[:, 1])[::-1]
        # valid_observations = valid_observations[sorted_indices]
        # valid_query_points = valid_query_points[sorted_indices]

        # # Find index of minimum energy within valid points
        # min_energy_idx = np.argmin(valid_observations[:, 0])
        # # min_energy = np.min(valid_observations[:, 0])
        # # min_energy_indices = np.where(valid_observations[:, 0] == min_energy)[0]

        # # Return corresponding query point (thresh, bitrate)
        # return tuple(valid_query_points[min_energy_idx])

    def reset_explore_history(self):
        self.explore_history = set()

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
            # suggestions = remove_tensor_duplicates(mbo.ask_for_suggestions())
            suggestions = mbo.ask_for_suggestions()
            print(suggestions)
            new_accuracies = self.evaluator.evaluate_configs(suggestions, frame_range.start, frame_range.stop)
            print(new_accuracies)
            est_energies = mbo.get_config_profile_energy(suggestions)
            print(est_energies)
            mbo.tell_observations(suggestions, tf.stack([est_energies, new_accuracies], axis=1))
        mbo.reset_explore_history()

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
        mbo = VideoBayesianOpt(self.search_space, self.batch_size)
        mbo.build_dataset_from_profile(self.energy_profile, self.accuracy_profile)
        mbo.build_stacked_independent_objectives_model(mbo.dataset)

        num_init_points = mbo.dataset.observations.shape[0]
        print(f'Number of initial points: {num_init_points}')

        i = 0
        explore = False
        cur_range = range(0, self.exploit_time * self.fps)
        # TODO: Iron out how many explore iterations to run
        while i < self.total_frames:
            print(f'Current range: {cur_range.start / self.fps}s to {cur_range.stop / self.fps}s')
            if not explore:
                # self.run_exploit_round(cur_range)
                cur_range = range(cur_range.stop, cur_range.stop + self.explore_time * self.fps)
                explore = True
            else: # explore
                configuration_acc = self.run_verify_round(cur_range)
                # if configuration_acc < target_accuracy:
                self.run_explore_round(mbo, cur_range, iterations=3)
                # TODO: Refine selection algorithm
                self.select_configuration(mbo, target_accuracy)
                cur_range = range(cur_range.stop, cur_range.stop + self.exploit_time * self.fps)
                explore = False
            i = cur_range.stop

        # dataset = mbo.ask_tell.to_result().try_get_final_dataset()
        dataset = mbo.ask_tell.to_result().final_result.unwrap().datasets["OBJECTIVE"]
        constraint_dataset = mbo.ask_tell.to_result().final_result.unwrap().datasets["CONSTRAINT"]
        print(dataset)
        plot_mobo_points_in_obj_space(dataset.observations, num_init=num_init_points, xlabel="Energy", ylabel="Accuracy")
        plt.show()

if __name__ == "__main__":
    simulation = Simulation(frame_dir='../filter-images/ground-truth-JH-full',
                            energy_profile="../viz/energy-JH-1.csv",
                            accuracy_profile="../viz/accuracy-JH-1.csv")
    simulation.run(target_accuracy=0.90)