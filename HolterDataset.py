import numpy as np
import torch
from torch.utils.data import Dataset
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class ECGDataset(Dataset):
    def __init__(self, ecgs, ids, fs, n_windows, seconds):
        self.ecg_signals = ecgs
        self.ids = ids
        self.id_mapped = {int(k):v for v, k in enumerate(self.ids)} # mappa id in interi
        self.id_mapped_tensor = torch.tensor([self.id_mapped[int(x)] for x in self.ids]) #trasforma in tensore

        self.fs = fs
        self.seconds = seconds
        self.n_windows = n_windows

        self.cut_ecg, self.cut_id, self.cut_id_mapped= self.cut()
        self.cut_id_mapped_tensor = torch.tensor(self.cut_id_mapped) #trasforma in tensore
        self.classes = list(set(self.cut_id_mapped))
    
    def cut(self):
        sig = []
        id = []
        id_mapped = []
        i = 0
        window_size = int(self.seconds*self.fs)
        n_windows = self.n_windows
        N = self.ecg_signals.shape[2]
        for signal in self.ecg_signals:
            random_idx = [random.randint(0, N - window_size - 1) for i in range(n_windows)]
            random_idx.sort()
            for w in range(n_windows):
                start_point = random_idx[w]
                end_point = start_point + window_size 
                sig.append(signal[:,start_point:end_point])
                id.append(self.ids[i])
                id_mapped.append(self.id_mapped_tensor[i])
            i += 1   

        sig = np.array(sig)
        id = np.array(id)
        id_mapped = np.array(id_mapped)
        return sig, id, id_mapped

    def __len__(self):
        return len(self.cut_ecg)

    def __getitem__(self, index):
        ecg_signal = self.cut_ecg[index, :, :]
        patient_id = self.cut_id[index]
        label_class = self.cut_id_mapped_tensor[index]
        return torch.tensor(ecg_signal).type(torch.float), patient_id, label_class

