import sys, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from trieste.acquisition.function import ExpectedHypervolumeImprovement, Fantasizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.models import TrainableModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import DiscreteSearchSpace
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.experimental.plotting import plot_mobo_points_in_obj_space
from evaluator import Evaluator

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from evaluate import calculate_accuracy

# Implement multi-objective Bayesian optimization for online video configuration updates

class VideoBayesianOpt:

    def __init__(self, search_space, batch_size):

        self.num_objectives = 2
        self.search_space = DiscreteSearchSpace(tf.constant(search_space, dtype=tf.float64))
        
        fant_ehvi = Fantasizer(ExpectedHypervolumeImprovement())
        self.rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=fant_ehvi, num_query_points=batch_size)

        self.query_history = set()

    @staticmethod
    def read_profiling_data(energy_file, accuracy_file):
        accuracy_df = pd.read_csv(accuracy_file)
        energy_df = pd.read_csv(energy_file)
        merged_df = pd.merge(accuracy_df, energy_df, on=["Frequency", "Filter", "Threshold", "Frame Bitrate"])
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
            gpr = build_gpr(single_obj_data, self.search_space, likelihood_variance=1e-5)
            gprs.append((GaussianProcessRegression(gpr), 1))

        self.model = TrainableModelStack(*gprs)
        self.ask_tell = AskTellOptimizer(self.search_space, data, self.model, acquisition_rule=self.rule, fit_model=True)

    def optimize(self, num_steps):
        return self.bo.optimize(num_steps, self.dataset, self.model, self.rule)

    def ask_for_suggestions(self) -> Dataset:
        suggestions = self.ask_tell.ask()
        final_suggestions = []
        for s in suggestions:
            suggestion_tup = (float(s[0].numpy()), int(s[1].numpy()))
            if suggestion_tup not in self.query_history:
                final_suggestions.append(s)
                self.query_history.add(suggestion_tup)
        print(self.query_history)
        return tf.stack(final_suggestions)
    
    def tell_observations(self, configs, observations) -> None:
        if configs.shape[0] == 0 or observations.shape[0] == 0:
            return
        x_values = observations[:, 0]
        y_values = -observations[:, 1]
        observations = tf.stack([x_values, y_values], axis=1)
        data = Dataset(configs, observations)
        self.ask_tell.tell(data)


def remove_tensor_duplicates(tensor):
    def tensor_in_list(tensor, tensor_list):
        for t in tensor_list:
            if tf.reduce_all(tf.equal(t, tensor)):
                return True
        return False

    unique = []
    for t in tensor:
        if not tensor_in_list(t, unique):
            unique.append(t)
    return tf.convert_to_tensor(unique)

class Simulation:
    def __init__(self, frame_dir, energy_profile, accuracy_profile, explore_time=10, exploit_time=50) -> None:
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

    def run(self) -> None:
        mbo = VideoBayesianOpt(self.search_space, self.batch_size)
        mbo.build_dataset_from_profile(self.energy_profile, self.accuracy_profile)
        mbo.build_stacked_independent_objectives_model(mbo.dataset)

        num_init_points = mbo.dataset.observations.shape[0]
        print(f'Number of initial points: {num_init_points}')

        i = 0
        explore = False
        config = (0.1, 3000)
        cur_range = range(0, self.exploit_time * self.fps)
        while i < self.total_frames:
            if not explore:
                accuracy = self.evaluator.evaluate_configs([config], cur_range.start, cur_range.stop)
                cur_range = range(cur_range.stop, cur_range.stop + self.explore_time * self.fps)
                # i = cur_range.stop
                explore = True
            else:
                for _ in range(4):
                    suggestions = remove_tensor_duplicates(mbo.ask_for_suggestions())
                    print(suggestions)
                    new_accuracies = self.evaluator.evaluate_configs(suggestions, cur_range.start, cur_range.stop)
                    print(new_accuracies)
                    est_energies = mbo.get_config_profile_energy(suggestions)
                    print(est_energies)
                    mbo.tell_observations(suggestions, tf.stack([est_energies, new_accuracies], axis=1))

                    cur_range = range(cur_range.stop, cur_range.stop + self.exploit_time * self.fps)
                    explore = False
            i = cur_range.stop

        dataset = mbo.ask_tell.to_result().try_get_final_dataset()
        plot_mobo_points_in_obj_space(dataset.observations, num_init=num_init_points, xlabel="Energy", ylabel="Accuracy")
        plt.show()

if __name__ == "__main__":
    simulation = Simulation(frame_dir='../filter-images/ground-truth-JH-full',
                            energy_profile="../viz/energy-JH-1.csv",
                            accuracy_profile="../viz/accuracy-JH-1.csv")
    simulation.run()