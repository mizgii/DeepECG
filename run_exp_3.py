import os
import sys
from run import run_deepecg

def run_experiment3(data_path):
    '''
    Runs experiments for signal preprocessed as described in DeepECG, for leads 0, 1, 2.
    Saves the results to a text file in the 'exp3_results' directory.

    Parameters:
    - data_path (str): path to the directory containing preprocessed ECG data and patient IDs

    Saves:
    - the lead number, accuracy, training time, and evaluation time for each lead
    '''

    results = []
    os.makedirs('exp3_results', exist_ok=True)

    for lead in [0, 1, 2]:
        print(f"Running experiment for lead: {lead}")
        accuracy, train_time, eval_time = run_deepecg(NUM_BATCH=16,
                                                      LEAD=lead, 
                                                      NUM_EPOCHS=200, 
                                                      DATA_PATH=data_path, 
                                                      FS=128)
        results.append((lead, accuracy, train_time, eval_time))

    with open('exp3_results/experiment3_results.txt', 'w') as f:
        for lead, accuracy, train_time, eval_time in results:
            f.write(f"Lead: {lead}, Accuracy: {accuracy}, Training Time: {train_time}s, Evaluation Time: {eval_time}s\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_exp_3.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    run_experiment3(data_path=data_path)