import run
import sys
import numpy as np



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: <data_path> ")
        sys.exit(1)

    path = sys.argv[1]
    accu1=[]
    for patient in range(20, 20, 139):
        accuracy = run.run(NUM_FINETRE=500, NUM_SECONDI=2, NUM_BATCH=16, NUM_EPOCHS=200, LEADS=[0,1,2], FS=128, DATA_PATH=path, NUM_PAT=patient)
        accu1.append(accuracy)
    
    accu2=[]
    for patient in range(20, 20, 139):
        run.run(NUM_FINETRE=500, NUM_SECONDI=2, NUM_BATCH=16, NUM_EPOCHS=200, LEADS=[0,1], FS=128, DATA_PATH=path, NUM_PAT=patient)
        accu2.append(accuracy)

    accu3=[]
    for patient in range(20, 20, 139):
        run.run(NUM_FINETRE=500, NUM_SECONDI=2, NUM_BATCH=16, NUM_EPOCHS=200, LEADS=[2], FS=128, DATA_PATH=path, NUM_PAT=patient)
        accu3.append(accuracy)

    accu4=[]
    for patient in range(20, 20, 139):
        run.run(NUM_FINETRE=500, NUM_SECONDI=2, NUM_BATCH=16, NUM_EPOCHS=200, LEADS=[1], FS=128, DATA_PATH=path, NUM_PAT=patient)
        accu4.append(accuracy)

    accu5=[]
    for patient in range(20, 20, 139):
        run.run(NUM_FINETRE=500, NUM_SECONDI=2, NUM_BATCH=16, NUM_EPOCHS=200, LEADS=[0], FS=128, DATA_PATH=path, NUM_PAT=patient)
        accu5.append(accuracy)

    np.save('patients-3leads.npy', accu1)
    np.save('patients-III-V3leads.npy', accu2)
    np.save('patients-V5leads.npy', accu3)
    np.save('patients-V3leads.npy', accu4)
    np.save('patients-IIIleads.npy', accu5)