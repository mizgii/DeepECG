import run
import sys
import os



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <data_path> ")
        sys.exit(1)
    path = sys.argv[1]

    os.makedirs('exp1_results', exist_ok=True)
    for patient in range(20, 139, 20):
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[0,1,2], 
                                                              NUM_EPOCHS=10, 
                                                              DATA_PATH = path, 
                                                              FS=128,
                                                              NUM_PAT=patient)
        with open('exp1_results/result-III-V3-V5.txt', 'a') as f:
            f.write(f"With number patients: {patient}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")
    
    for patient in range(20, 139, 20):
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[0,1], 
                                                              NUM_EPOCHS=10, 
                                                              DATA_PATH = path, 
                                                              FS=128,
                                                              NUM_PAT=patient)
        with open('exp1_results/result-III-V3.txt', 'a') as f:
            f.write(f"With number patients: {patient}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")

    for patient in range(20, 139, 20):
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[2], 
                                                              NUM_EPOCHS=10, 
                                                              DATA_PATH = path, 
                                                              FS=128,
                                                              NUM_PAT=patient)
        with open('exp1_results/result-V5.txt', 'a') as f:
            f.write(f"With number patients: {patient}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")

    for patient in range(20, 139, 20):
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[1], 
                                                              NUM_EPOCHS=10, 
                                                              DATA_PATH = path, 
                                                              FS=128,
                                                              NUM_PAT=patient)
        with open('exp1_results/result-V3.txt', 'a') as f:
            f.write(f"With number patients: {patient}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")

    for patient in range(20, 139, 20):
        accuracy, total_train_time, total_eval_time  = run.run(NUM_SEGMENTS=500, 
                                                              NUM_SECONDS=2, 
                                                              NUM_BATCH=16, 
                                                              LEADS=[0], 
                                                              NUM_EPOCHS=10, 
                                                              DATA_PATH = path, 
                                                              FS=128,
                                                              NUM_PAT=patient)
        with open('exp1_results/result-III.txt', 'a') as f:
            f.write(f"With number patients: {patient}:\n")
            f.write(f"Accuracy: {accuracy}, Time train: {total_train_time}, Time eval: {total_eval_time}\n\n")