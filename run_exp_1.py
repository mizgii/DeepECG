import run
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def run_experiment1(data_path):
    os.makedirs('exp1_results', exist_ok=True)

    accuracies1 = []
    accuracies2 = []
    accuracies3 = []
    accuracies4 = []
    accuracies5 = []

    for num_subjects in range(20, 150, 20):
        print(f"Training for {num_subjects} subjects with III+V3+V5")
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[0,1,2], 
                                                              NUM_EPOCHS=200, 
                                                              DATA_PATH = data_path, 
                                                              FS=128,
                                                              NUM_PATIENTS=num_subjects)
        accuracies1.append(accuracy)
        with open('exp1_results/result-III-V3-V5.txt', 'a') as f:
            f.write(f"With number subjects: {num_subjects}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")
    

    for num_subjects in range(20, 150, 20):
        print(f"Training for {num_subjects} subjects with III+V3")
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[0,1], 
                                                              NUM_EPOCHS=200, 
                                                              DATA_PATH = data_path, 
                                                              FS=128,
                                                              NUM_PATIENTS=num_subjects)
        accuracies2.append(accuracy)
        with open('exp1_results/result-III-V3.txt', 'a') as f:
            f.write(f"With number subjects: {num_subjects}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")

    for num_subjects in range(20, 150, 20):
        print(f"Training for {num_subjects} subjects with V5")
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[2], 
                                                              NUM_EPOCHS=200, 
                                                              DATA_PATH = data_path, 
                                                              FS=128,
                                                              NUM_PATIENTS=num_subjects)
        accuracies3.append(accuracy)
        with open('exp1_results/result-V5.txt', 'a') as f:
            f.write(f"With number subjects: {num_subjects}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")

    for num_subjects in range(20, 150, 20):
        print(f"Training for {num_subjects} subjects with V3")
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[1], 
                                                              NUM_EPOCHS=200, 
                                                              DATA_PATH = data_path, 
                                                              FS=128,
                                                              NUM_PATIENTS=num_subjects)
        accuracies4.append(accuracy)
        with open('exp1_results/result-V3.txt', 'a') as f:
            f.write(f"With number subjects: {num_subjects}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")

    for num_subjects in range(20, 150, 20):
        print(f"Training for {num_subjects} subjects with III")
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[0], 
                                                              NUM_EPOCHS=200, 
                                                              DATA_PATH = data_path, 
                                                              FS=128,
                                                              NUM_PATIENTS=num_subjects)
        accuracies5.append(accuracy)
        with open('exp1_results/result-III.txt', 'a') as f:
            f.write(f"With number subjects: {num_subjects}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")


    #plotting             
    num_subjects = list(range(20, 150, 20))
    plt.figure(figsize=(10, 8), dpi=100)
    plt.plot(num_subjects, accuracies1, '#8C2155',marker='.', label='III+V3+V5')
    plt.plot(num_subjects, accuracies2, '#F99083',marker='.', label='III+V3')
    plt.plot(num_subjects, accuracies3, '#8AA29E',marker='.', label='V5')
    plt.plot(num_subjects, accuracies4, '#98CE00',marker='.', label='V3')
    plt.plot(num_subjects, accuracies5, '#FFC857',marker='.', label='III')

    plt.xlabel('NUMBER OF SUBJECTS')
    plt.ylabel('ACCURACY %')
    plt.legend()
    plt.xlim((np.min(num_subjects),np.max(num_subjects)))
    plt.grid(True, 'both')
    plt.savefig('exp1_results/fig_exp1.png')
    plt.show()
    plt.close()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_exp_1.py <data_path>")
        sys.exit(1)
    data_path = sys.argv[1]

    run_experiment1(data_path)

    