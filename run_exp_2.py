import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from run import run

from matplotlib.colors import LinearSegmentedColormap


def run_experiment2(data_path):
    '''
    Function that conducts a series of experiments with varying:
     - number of segments used for training and testing (N)
     - segment length (T)

    Parameters:
    data_path (str): path to the preprocessed dataset.

    Saves:
    - CSV files for accuracy, training time, and evaluation time grids, and 
    - a heatmap plot of model accuracy in the 'exp2_results' directory.
    '''

    segments_range = [500,350,200,100,50]
    seconds_range = [1,2,5,7,10]
    leads = [0,1]

    accuracy_grid = np.zeros((len(segments_range), len(seconds_range)))
    train_time_grid = np.zeros_like(accuracy_grid)
    eval_time_grid = np.zeros_like(accuracy_grid)


    for i, num_segments in enumerate(segments_range):
        for j, num_seconds in enumerate(seconds_range):
            print(f'Training for {num_segments} segments, {num_seconds} s long')
            accuracy, total_train_time, total_eval_time = run(NUM_SEGMENTS=num_segments, 
                                                              NUM_SECONDS=num_seconds, 
                                                              NUM_BATCH=16, 
                                                              LEADS=leads, 
                                                              NUM_EPOCHS=200, 
                                                              DATA_PATH = data_path, 
                                                              FS=128)
            accuracy_grid[i, j] = accuracy
            train_time_grid[i,j] = total_train_time
            eval_time_grid[i,j] = total_eval_time


    os.makedirs('exp2_results', exist_ok=True)
    np.savetxt('exp2_results/accuracy_grid.csv', accuracy_grid, delimiter=',', fmt='%.2f')
    np.savetxt('exp2_results/train_time_grid.csv', train_time_grid, delimiter=',', fmt='%.2f')
    np.savetxt('exp2_results/eval_time_grid.csv', eval_time_grid, delimiter=',', fmt='%.2f')

    color_list = ["#FCDDD9", "#FABEB7", "#FAA99F", "#FA968A", "#F87F71",
                "#DC7366", "#B15F56", "#884C45", "#593633", "#2E2322"]
    custom_colormap = LinearSegmentedColormap.from_list("custom_salmon", color_list)

    yticklabels = [int(w/2) for w in segments_range]
    plt.figure(figsize=(10, 8), dpi=100)
    sns.heatmap(accuracy_grid, annot=True, fmt=".2f", xticklabels=seconds_range, yticklabels=yticklabels, 
                cmap=custom_colormap, cbar_kws={'label': 'Accuracy %'})
    plt.xlabel('NUMBER OF SECONDS (T)')
    plt.ylabel('TRAIN SAMPLES PER SUBJECT (N)')
    plt.savefig('exp2_results/fig_exp2.png')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_exp_2.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    run_experiment2(data_path=data_path)
