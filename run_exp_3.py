import os
import sys
from run import run, run_deepecg

def run_experiment3(data_path):
    '''
    Runs experiments for each lead separately for signal:
     - preprocessed as described in DeepECG and 
     - with simplified preprocessing.

    Saves the results to a text file in the 'exp3_results' directory.

    Parameters:
    - data_path (str): path to the directory containing preprocessed ECG data and patient IDs

    Saves:
    - the lead number, accuracy, training time, and evaluation time for each lead
    '''

    results_deep = []
    results_simplified = []
    os.makedirs('exp3_results', exist_ok=True)

    leads = ['V3', 'III', 'V5']

    for i, lead in enumerate(leads):

        print(f'DeepECG prepreprocessing for lead {lead}')
        accuracy, train_time, eval_time = run_deepecg(NUM_BATCH=16,
                                                      LEAD=i, 
                                                      NUM_EPOCHS=200, 
                                                      DATA_PATH=data_path, 
                                                      FS=128)
        
        results_deep.append((lead, accuracy, train_time, eval_time))


        print(f'Simplified prepreprocessing for lead {lead}')
        accuracy, train_time, eval_time = run(NUM_SEGMENTS=250, 
                                              NUM_SECONDS=10, 
                                              NUM_BATCH=16, 
                                              LEADS=[i], 
                                              NUM_EPOCHS=200, 
                                              DATA_PATH = data_path, 
                                              FS=128)
        results_simplified.append((lead, accuracy, train_time, eval_time))

    with open('exp3_results/experiment3_deep.txt', 'w') as f:
        for lead, accuracy, train_time, eval_time in results_deep:
            f.write(f"Lead: {lead}, Accuracy: {accuracy}, Training Time: {train_time}s, Evaluation Time: {eval_time}s\n")

    with open('exp3_results/experiment3_simple.txt', 'w') as f:
        for lead, accuracy, train_time, eval_time in results_simplified:
            f.write(f"Lead: {lead}, Accuracy: {accuracy}, Training Time: {train_time}s, Evaluation Time: {eval_time}s\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_exp_3.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]

    run_experiment3(data_path=data_path)