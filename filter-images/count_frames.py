from pathlib import Path
import logging
import argparse
import pandas as pd
import numpy as np

def get_lowest_frame_count_configuration_for_accuracy(df, accuracy_threshold):
    # Filter for rows where Average IoU (accuracy) is equal to or greater than the specified threshold
    filtered_df = df[df['Average IoU'] >= accuracy_threshold]
    
    # If there are no rows that meet the accuracy threshold, return None
    if filtered_df.empty:
        return None
    
    # Find the row with the minimum energy value in the filtered DataFrame
    min_frames_row = filtered_df.loc[filtered_df['Frame Count'].idxmin()]
    
    # Return only the configuration details along with energy and accuracy
    configuration = min_frames_row[['Frequency', 'Filter', 'Threshold', 'Frame Bitrate', 'Frame Count', 'Average IoU']]
    return configuration

if __name__ == "__main__":
    # This script calculates the number of frames for each configuration, and finds the config with the lowest frames for each accuracy threshold.
    # This is the config that can be used to simulate Reducto
    # Example usage
    # python count_frames.py JH/2.4 ../viz/accuracy-JH-2-dynamic.csv

    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str)
    parser.add_argument('accuracy_profile', type=str)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    data = []

    for directory in sorted([d for d in root_dir.iterdir() if d.is_dir()]):
        freq, filter, threshold, bitrate = directory.name.split('-')
        num_files = len([f for f in directory.iterdir() if f.is_file()])
        data.append([float(freq), filter, float(threshold), int(bitrate), int(num_files)])

    framecount_df = pd.DataFrame(data, columns=['Frequency', 'Filter', 'Threshold', 'Frame Bitrate', 'Frame Count'])
    accuracy_df = pd.read_csv(args.accuracy_profile)

    merged_df = pd.merge(accuracy_df, framecount_df, on=["Frequency", "Filter", "Threshold", "Frame Bitrate"])
    merged_df = merged_df.loc[(merged_df['Filter'] == 'edge') & (merged_df['Frequency'] == 2.4)]

    config_thresh_df = pd.DataFrame(columns=['Frequency', 'Filter', 'Threshold', 'Frame Bitrate', 'Frame Count', 'Average IoU', 'Accuracy Threshold'])

    for thresh in np.arange(0.70, 0.91, 0.01):
        lowest_energy_configuration = get_lowest_frame_count_configuration_for_accuracy(merged_df, thresh)
        if lowest_energy_configuration is None:
            continue
        lowest_energy_configuration['Accuracy Threshold'] = thresh
        config_thresh_df.loc[-1] = lowest_energy_configuration
        config_thresh_df.index = config_thresh_df.index + 1

    print(config_thresh_df)