import os
import torch
import random
import warnings
import numpy as np
from torch.utils.data import Dataset

'''
there are 2 versions of selection of segments that feature vectors are created from
- DeepDataset that takes first 500 valid segments for feature vector extraction
- DeepDatasetV2 stores all valid segments and randomly selects 500 of them for feature vector extraction

'''

class DeepDataset(Dataset):
    def __init__(self, data_path, patient_ids, fs, lead):
        self.data_path = data_path
        self.patient_ids = patient_ids
        self.id_mapped = {pid: idx for idx, pid in enumerate(patient_ids)}
        self.fs = fs
        self.lead = lead
        self.vectors = []
        self.labels = []
        self.m = 8
        self.segment_length = 10  # In seconds
        self.num_segments = int(15*3600 / self.segment_length)  # Total segments in 15 hours
        self.max_segments_per_patient = 500
        
        for patient_id in self.patient_ids:
            valid_segments_count = 0
            signal_path = os.path.join(self.data_path, f"{patient_id}_signal.npy")
            qrs_path = os.path.join(self.data_path, f"{patient_id}_qrs.npy")
            signal = np.load(signal_path)[self.lead, :15*3600*self.fs]
            qrs_indices = np.load(qrs_path)

            for segment_idx in range(self.num_segments):
                if valid_segments_count >= self.max_segments_per_patient:
                    break  # Stop if we have enough valid segments

                start = segment_idx * self.segment_length * self.fs
                end = start + self.segment_length * self.fs
                qrs_in_segment = qrs_indices[(qrs_indices >= start) & (qrs_indices < end)] - start

                if len(qrs_in_segment) > 0:  # Check for valid segments
                    segment = signal[start:end]
                    vector_v = self.extract_feature_vector(segment, qrs_in_segment)
                    self.vectors.append(vector_v)
                    self.labels.append(self.id_mapped[patient_id])
                    valid_segments_count += 1

            if valid_segments_count < self.max_segments_per_patient:
                warnings.warn(f"Patient ID {patient_id} has less than 500 valid segments: {valid_segments_count} segments found.")
        
        self.vectors = np.array(self.vectors)
        self.labels = np.array(self.labels)


    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        vector_v = self.vectors[index]
        label = self.labels[index]
        return torch.tensor(vector_v, dtype=torch.float).unsqueeze(0), torch.tensor(label, dtype=torch.long)


    def extract_feature_vector(self, segment, qrs_indices):
        qrs_complexes = []
        half_window = int(0.125 * self.fs / 2)
        window_length = 2 * half_window

        for idx in qrs_indices:
            start_idx = max(0, idx - half_window)
            end_idx = min(len(segment), idx + half_window)
            qrs_complex = segment[start_idx:end_idx]
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



class DeepDatasetV2(Dataset):
    def __init__(self, data_path, patient_ids, fs, lead):
        self.data_path = data_path
        self.patient_ids = patient_ids
        self.id_mapped = {pid: idx for idx, pid in enumerate(patient_ids)}
        self.fs = fs
        self.lead = lead
        self.vectors = []
        self.labels = []
        self.m = 8
        self.segment_length = 10  # seconds
        self.num_segments = int(15*3600 / self.segment_length)  # total segments in 15 hours
        self.max_segments_per_patient = 500 
        
        for patient_id in self.patient_ids:
            valid_segment_indices = [] 

            signal_path = os.path.join(self.data_path, f"{patient_id}_signal.npy")
            qrs_path = os.path.join(self.data_path, f"{patient_id}_qrs.npy")
            signal = np.load(signal_path)[self.lead, :15*3600*self.fs]
            qrs_indices = np.load(qrs_path)

            for segment_idx in range(self.num_segments):
                start = segment_idx * self.segment_length * self.fs
                end = start + self.segment_length * self.fs
                qrs_in_segment = qrs_indices[(qrs_indices >= start) & (qrs_indices < end)] - start

                if len(qrs_in_segment) > 0:  # Check for valid segments
                    valid_segment_indices.append(segment_idx)

            selected_indices = random.sample(valid_segment_indices, min(len(valid_segment_indices), self.max_segments_per_patient))
            if len(valid_segment_indices) < self.max_segments_per_patient:
                warnings.warn(f"Patient ID {patient_id} has less than 500 valid segments: {len(valid_segment_indices)} segments found.")

            for segment_idx in selected_indices: #extract V vectors only from selected segments
                start = segment_idx * self.segment_length * self.fs
                end = start + self.segment_length * self.fs
                qrs_in_segment = qrs_indices[(qrs_indices >= start) & (qrs_indices < end)] - start
                segment = signal[start:end]
                vector_v = self.extract_feature_vector(segment, qrs_in_segment)
                self.vectors.append(vector_v)
                self.labels.append(self.id_mapped[patient_id])
        
        self.vectors = np.array(self.vectors)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        vector_v = self.vectors[index]
        label = self.labels[index]
        return torch.tensor(vector_v, dtype=torch.float).unsqueeze(0), torch.tensor(label, dtype=torch.long)
    
    def extract_feature_vector(self, segment, qrs_indices):
        qrs_complexes = []
        half_window = int(0.125 * self.fs / 2)
        window_length = 2 * half_window

        for idx in qrs_indices:
            start_idx = max(0, idx - half_window)
            end_idx = min(len(segment), idx + half_window)
            qrs_complex = segment[start_idx:end_idx]
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
    