import tensorflow as tf
import numpy as np
from trieste.acquisition.function import ExpectedHypervolumeImprovement, Fantasizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.models import TrainableModelStack
from trieste.models.gpflow import build_gpr, GaussianProcessRegression
from trieste.space import DiscreteSearchSpace
from trieste.ask_tell_optimization import AskTellOptimizer
import matplotlib.pyplot as plt
import pandas as pd


class BayesianOpt:

    def __init__(self, search_space, batch_size):

        self.num_objectives = 2
        self.search_space = DiscreteSearchSpace(tf.constant(search_space, dtype=tf.float64))
        
        fant_ehvi = Fantasizer(ExpectedHypervolumeImprovement())
        self.rule: EfficientGlobalOptimization = EfficientGlobalOptimization(builder=fant_ehvi, num_query_points=batch_size)


    def build_stacked_independent_objectives_model(self, data: Dataset) -> TrainableModelStack:
        gprs = []
        for idx in range(self.num_objectives):
            single_obj_data = Dataset(
                data.query_points, tf.gather(data.observations, [idx], axis=1)
            )
            gpr = build_gpr(single_obj_data, self.search_space, likelihood_variance=1e-5)
            gprs.append((GaussianProcessRegression(gpr), 1))

        self.model = TrainableModelStack(*gprs)

    def ask_for_suggestions(self, dataset):
        ask_tell = AskTellOptimizer(self.search_space, dataset, self.model, acquisition_rule=self.rule, fit_model=True)
        return ask_tell.ask()
    
def build_trieste_dataset():
    profiling_df = read_profiling_data()

    query_points = tf.constant([[x, y] for x, y in zip(profiling_df['Threshold'], profiling_df['Frame Bitrate'])], dtype=tf.float64)
    # print(query_points)
    observations = tf.constant([[x, -y] for x, y in zip(profiling_df['Avg Energy'], profiling_df['Average IoU'])], dtype=tf.float64)
    # print(observations)
    return Dataset(query_points, observations)

def read_profiling_data():
    accuracy_df = pd.read_csv('../viz/accuracy-JH-1.csv')
    energy_df = pd.read_csv('../viz/energy-JH-1.csv')
    merged_df = pd.merge(accuracy_df, energy_df, on=["Frequency", "Filter", "Threshold", "Frame Bitrate"])
    merged_df = merged_df.drop(columns=['FPS', 'Start Time', 'End Time'])
    merged_df = merged_df[(merged_df['Frequency'] == 1.5) & (merged_df['Filter'] == 'pixel')]
    merged_df = merged_df.drop(columns=['Frequency', 'Filter'])
    return merged_df

if __name__ == '__main__':
    # print(build_trieste_dataset())
    # query_points = tf.constant([[0.03, 3000], [0.01, 2400]], dtype=tf.float64)
    # observations = tf.constant([[4.586802, -0.7743], [4.605219, -0.9356]], dtype=tf.float64)

    search_space = []
    for thresh in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        for br in [100, 400, 700, 1000, 1300, 1600, 1900, 2100, 2400, 2700, 3000]:
            search_space.append([thresh, br])
    batch_size = 4
    b = BayesianOpt(search_space, batch_size)
    profile_dataset = build_trieste_dataset()
    b.build_stacked_independent_objectives_model(profile_dataset)
    eval_dataset = b.ask_for_suggestions(profile_dataset)
    print(eval_dataset)

    query_points = profile_dataset.query_points
    observations = profile_dataset.observations

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(query_points[:, 0], query_points[:, 1], color='green')
    # ax1.scatter(eval_dataset[:, 0], eval_dataset[:, 1], color='purple', label='Evaluated Points')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Bitrate')
    ax1.set_title('Configuration Search Space')
    # ax1.legend()
    ax1.grid()
    # ax1.set_xlim(-0.005, 0.105)
    # ax1.set_ylim(-150, 3150)
    # ax1.set_xticks(np.arange(0.0, 0.11, 0.01))
    # ax1.set_yticks(np.arange(0, 3100, 300))

    ax2.scatter(observations[:, 0], -observations[:, 1], color='green')
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Apriori Objective Space')
    # ax2.legend()
    ax2.grid()
    # ax2.set_xlim(3.0, 5.0)
    # ax2.set_ylim(-150, 3150)
    # ax2.set_xticks(np.arange(0.0, 0.11, 0.01))
    # ax2.set_yticks(np.arange(0, 3100, 300))

    plt.savefig("search_objective_apriori.jpeg", format="jpeg")