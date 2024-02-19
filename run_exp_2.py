import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from run import run


def run_experiment2(data_path):

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
                                                              NUM_EPOCHS=10, 
                                                              DATA_PATH = data_path, 
                                                              FS=128, 
                                                              NUM_PATIENTS=3)
            accuracy_grid[i, j] = accuracy
            train_time_grid[i,j] = total_train_time
            eval_time_grid[i,j] = total_eval_time


    os.makedirs('exp2_results', exist_ok=True)
    np.savetxt('exp2_results/accuracy_grid.csv', accuracy_grid, delimiter=',', fmt='%.2f')
    np.savetxt('exp2_results/train_time_grid.csv', train_time_grid, delimiter=',', fmt='%.2f')
    np.savetxt('exp2_results/eval_time_grid.csv', eval_time_grid, delimiter=',', fmt='%.2f')

    yticklabels = [int(w/2) for w in segments_range]
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_grid, annot=True, fmt=".2f", xticklabels=seconds_range, yticklabels=yticklabels)
    plt.xlabel('NUMBER OF SECONDS')
    plt.ylabel('TRAIN SAMPLES PER SUBJECT')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_exp_2.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    run_experiment2(data_path=data_path)