import numpy as np
import os

def load_ecgs(data_path, ids):
    ecgs=[]
    for id in ids:
        signal_path = os.path.join(data_path, f"{id}_signal.npy")
        signal = np.load(signal_path)
        signal = signal[:, (128*60*30):(128*60*60*2)] # i have to parameterized  
        ecgs.append(signal)
    return np.stack(ecgs)
