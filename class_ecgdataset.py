import os
import torch
import numpy as np
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, data_path, patient_ids, fs, n_windows, n_seconds, leads=None):
        self.data_path = data_path
        self.patient_ids = patient_ids
        self.fs = fs
        self.n_windows = n_windows
        self.n_seconds = n_seconds
        self.leads = leads
        self.signal_cache = {}
        self.id_mapped = {pid: idx for idx, pid in enumerate(patient_ids)}
        self.segment_starts = {pid: self.generate_starts(pid) for pid in patient_ids}
        
    def load_signal(self, patient_id):
        if patient_id not in self.signal_cache:
            signal_path = os.path.join(self.data_path, f"{patient_id}_signal.npy")
            signal = np.load(signal_path)
            signal = signal[:, :15*3600*self.fs]
            if self.leads is not None:
                signal = signal[self.leads, :15*3600*self.fs]
            self.signal_cache[patient_id] = signal
        return self.signal_cache[patient_id]

    def generate_starts(self, patient_id):
        signal = self.load_signal(patient_id)
        max_start = signal.shape[1] - self.n_seconds * self.fs
        # make a grid of points that are n_seconds apart
        all_starts = np.arange(0, max_start, self.n_seconds * self.fs) 
        # and choose n_windows of them as our starting pts
        chosen_starts = np.random.choice(all_starts, self.n_windows, replace=False)
        return chosen_starts
    
    def __len__(self):
        return len(self.patient_ids) * self.n_windows

    def __getitem__(self, index):
        patient_index = index // self.n_windows
        patient_id = self.patient_ids[patient_index]
        window_index = index % self.n_windows
        start_point = self.segment_starts[patient_id][window_index]
        end_point = start_point + self.n_seconds * self.fs

        signal = self.load_signal(patient_id)
        segment = signal[:, start_point:end_point]
        label = self.id_mapped[patient_id]  

        return torch.tensor(segment, dtype=torch.float), torch.tensor(label, dtype=torch.long)
    

#----------------------------------------------------
#-----------half--day--implementation----------------
#----------------------------------------------------


class ECGDataset_halfday(Dataset):
    def __init__(self, data_path, patient_ids, fs, n_windows, n_seconds, leads=None, subset='training'):
        self.data_path = data_path
        self.patient_ids = patient_ids
        self.fs = fs
        self.n_windows = n_windows
        self.n_seconds = n_seconds
        self.leads = leads
        self.subset = subset
        self.signal_cache = {}
        self.id_mapped = {pid: idx for idx, pid in enumerate(patient_ids)}
        self.segment_starts = {pid: self.generate_starts(pid) for pid in patient_ids}
        
    def load_signal(self, patient_id):
        if patient_id not in self.signal_cache:
            signal_path = os.path.join(self.data_path, f"{patient_id}_signal.npy")
            signal = np.load(signal_path)
            if self.leads is not None:
                signal = signal[self.leads, :]
            self.signal_cache[patient_id] = signal
        return self.signal_cache[patient_id]

    def generate_starts(self, patient_id):
        signal = self.load_signal(patient_id)
        midpoint = signal.shape[1] // 2
        if self.subset == 'training':
            max_start = midpoint - self.n_seconds * self.fs
            # make a grid of points that are n_seconds apart
            all_starts = np.arange(0, max_start, self.n_seconds * self.fs) 
        else:
            max_start = signal.shape[1] - self.n_seconds * self.fs
            all_starts = np.arange(0, max_start, self.n_seconds * self.fs) 
        
        # and choose n_windows of them as our starting pts
        chosen_starts = np.random.choice(all_starts, self.n_windows, replace=False)
        return chosen_starts
    
    def __len__(self):
        return len(self.patient_ids) * self.n_windows

    def __getitem__(self, index):
        patient_index = index // self.n_windows
        patient_id = self.patient_ids[patient_index]
        window_index = index % self.n_windows
        start_point = self.segment_starts[patient_id][window_index]
        end_point = start_point + self.n_seconds * self.fs

        signal = self.load_signal(patient_id)
        segment = signal[:, start_point:end_point]
        label = self.id_mapped[patient_id]

        return torch.tensor(segment, dtype=torch.float), torch.tensor(label, dtype=torch.long)
