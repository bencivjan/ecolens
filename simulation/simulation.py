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

    @staticmethod
    def read_profiling_data(energy_file, accuracy_file):
        accuracy_df = pd.read_csv(accuracy_file)
        energy_df = pd.read_csv(energy_file)
        merged_df = pd.merge(accuracy_df, energy_df, on=["Frequency", "Filter", "Threshold", "Frame Bitrate"])
        merged_df = merged_df.drop(columns=['FPS', 'Start Time', 'End Time'])
        merged_df = merged_df[(merged_df['Frequency'] == 1.5) & (merged_df['Filter'] == 'pixel')]
        merged_df = merged_df.drop(columns=['Frequency', 'Filter'])
        return merged_df

    def build_dataset_from_profile(self, energy_file, accuracy_file):
        profiling_df = self.read_profiling_data(energy_file, accuracy_file)
        query_points = tf.constant([[x, y] for x, y in zip(profiling_df['Threshold'], profiling_df['Frame Bitrate'])], dtype=tf.float64)
        observations = tf.constant([[x, -y] for x, y in zip(profiling_df['Avg Energy'], profiling_df['Average IoU'])], dtype=tf.float64)
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
        return self.ask_tell.ask()
    
    def tell_observation(self, observation: Dataset) -> None:
        self.ask_tell.tell(observation)

    @staticmethod
    def evaluate_config(configuration) -> tuple[float, float]:
        # Dummy implementation for testing
        threshold, bitrate = float(configuration[0]), int(configuration[1])
        df = VideoBayesianOpt.read_profiling_data("../viz/energy-JH-1.csv", "../viz/accuracy-JH-1.csv")
        print(f'threshold: {threshold}, bitrate: {bitrate}')
        df = df[(df['Threshold'] == threshold) & (df['Frame Bitrate'] == bitrate)]
        old_acc = df['Average IoU'].values[0]
        new_acc = df['Average IoU'].values[0] + tf.random.uniform(shape=(), minval=-0.05, maxval=0.05)
        print(f'configuration: {configuration}, old accuracy: {old_acc}, new accuracy: {new_acc}')
        return float(df['Avg Energy'].values[0]), float(new_acc)

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
    def __init__(self) -> None:
        self.search_space = []
        for thresh in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
            for br in [100, 400, 700, 1000, 1300, 1600, 1900, 2100, 2400, 2700, 3000]:
                self.search_space.append([thresh, br])

        self.batch_size = 4

    def run(self) -> None:
        mbo = VideoBayesianOpt(self.search_space, self.batch_size)
        mbo.build_dataset_from_profile("../viz/energy-JH-1.csv", "../viz/accuracy-JH-1.csv")
        mbo.build_stacked_independent_objectives_model(mbo.dataset)

        num_init_points = mbo.dataset.observations.shape[0]
        print(f'Number of initial points: {num_init_points}')

        for _ in range(4):
            suggestions = remove_tensor_duplicates(mbo.ask_for_suggestions())
            print(suggestions)
            for s in suggestions:
                energy, new_acc = VideoBayesianOpt.evaluate_config(s)
                mbo.tell_observation(Dataset(tf.reshape(s, [1, 2]), tf.constant([[energy, -new_acc]], dtype=tf.float64)))

        dataset = mbo.ask_tell.to_result().try_get_final_dataset()
        plot_mobo_points_in_obj_space(dataset.observations, num_init=num_init_points, xlabel="Energy", ylabel="Accuracy")
        plt.show()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()