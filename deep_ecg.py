import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, data_path, patient_ids, id_mapped, fs):
        self.data_path = data_path
        self.patient_ids = patient_ids
        self.id_mapped = id_mapped
        self.fs = fs
        self.segment_length = 10
        self.num_segments = self.find_num_of_segments()
        self.m = 8

    def find_num_of_segments(self):
        min_length = np.inf
        for patient_id in self.patient_ids:
            signal_path = os.path.join(self.data_path, f"{patient_id}_signal.npy")
            signal = np.load(signal_path)
            if signal.shape[1] < min_length:
                min_length = signal.shape[1]
        return min_length // (self.fs * self.segment_length)
    
    def __len__(self):
        return len(self.id_mapped) * self.num_segments 

    def __getitem__(self, idx):

        ### calculate the validity of a segment on the fly and dynamically adjust the indices
        ### not the most effective way, could try preprocess data to identify the indices of all valid segments??

        while True:
            patient_idx = idx // self.num_segments
            segment_idx = idx % self.num_segments
            if patient_idx >= len(self.patient_ids):
                raise IndexError("List index out of range")

            patient_id = self.patient_ids[patient_idx]
            mapped_id = self.id_mapped[patient_id]

            start = segment_idx * self.segment_length * self.fs
            end = start + self.segment_length * self.fs

            signal_path = os.path.join(self.data_path, f"{patient_id}_signal.npy")
            qrs_path = os.path.join(self.data_path, f"{patient_id}_qrs.npy")
            signal = np.load(signal_path)[0, :] #lead 0
            qrs_indices = np.load(qrs_path)

            qrs_in_segment = qrs_indices[(qrs_indices >= start) & (qrs_indices < end)] - start

            if len(qrs_in_segment) > 0: 
                segment = signal[start:end]
                vector_v = self.extract_feature_vector(segment, qrs_in_segment)
                return torch.tensor(vector_v, dtype=torch.float), torch.tensor(mapped_id, dtype=torch.long)
            else:
                idx += 1 # skip "empty" segment and try the next one


    def extract_feature_vector(self, segment, qrs_indices):
        qrs_complexes = []
        half_window = int(0.125 * self.fs / 2)
        window_length = 2 * half_window

        for idx in qrs_indices:
            start_idx = max(0, idx - half_window)
            end_idx = min(len(segment), idx + half_window)
            qrs_complex = segment[start_idx:end_idx]

            ### deep ecg doesn't mention how to handle truncated qrs when they occur at the edges of a segment
            ### I do padding, but maybe should discard those qrs?

            if len(qrs_complex) < window_length:
                padding = window_length - len(qrs_complex)
                qrs_complex = np.pad(qrs_complex, (0, padding), 'constant')

            qrs_complexes.append(qrs_complex)

        average_qrs = np.mean(qrs_complexes, axis=0) if qrs_complexes else np.zeros(window_length)

        correlations = [np.correlate(qrs_complex, average_qrs, 'valid')[0] for qrs_complex in qrs_complexes]

        top_m_indices = np.argsort(correlations)[-self.m:]

        vector_v = np.concatenate([qrs_complexes[idx] for idx in top_m_indices], axis=0)

        while len(vector_v) < self.m * window_length:
            vector_v = np.concatenate([vector_v, qrs_complexes[top_m_indices[-1]]])

        return vector_v[:self.m * window_length] 

