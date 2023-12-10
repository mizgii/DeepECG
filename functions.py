import os
import wfdb
import numpy as np
import pandas as pd
from  scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


#--------------------------------------------
#----------------classes---------------------
#--------------------------------------------


class ECGDataset(Dataset):
    def __init__(self, metadata, data_path, filter_fn):
        self.metadata = metadata
        self.data_path = data_path
        self.filter_fn = filter_fn

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx, subset):

        record_info = self.metadata.iloc[idx]
        record_path_lr = os.path.join(self.data_path, f"{record_info['filename_lr']}")
        record = wfdb.rdrecord(record_path_lr)
        print(f"Processing record at index {idx}")

        signal = record.p_signal[:, 0].T
        signal = self.filter_fn(signal, record.fs)

        midpoint = len(signal)//2
 
        if len(signal) != 10*record.fs:
            raise ValueError(f"Invalid signal length: {len(signal)}. Expected length: 1000.")

        if subset == 'train' or 'test':
            return [torch.Tensor(signal[:midpoint].copy()) , record_info['patient_id']]
        elif subset == 'validation':
            return [torch.Tensor(signal[midpoint:].copy()) , record_info['patient_id']]
    
    


#--------------------------------------------
#----------------functions---------------------
#--------------------------------------------


def filter_signal(signal, fs=100):

    [b, a] = butter(3, (0.5, 40), btype='bandpass', fs=fs)
    signal = filtfilt(b, a, signal, axis=0)
    [bn, an] = iirnotch(50, 3, fs=fs)
    signal = filtfilt(bn, an, signal, axis=0)

    return signal


def open_data(data_path, filter_fn):
    metadata_path = os.path.join(data_path, 'ptbxl_database.csv')
    metadata = pd.read_csv(metadata_path)
    
    # we're only using healthy patients
    healthy_patients = metadata[metadata['scp_codes'].str.contains('NORM', na=False)]
    healthy_patients = healthy_patients[:10]

    print(f"Number of recordings: {len(metadata)}\nNumber of healthy recordings: {len(healthy_patients)}\nNumber of healthy patients: {len(healthy_patients.drop_duplicates(subset='patient_id', keep='first'))}")

    return ECGDataset(healthy_patients, data_path, filter_fn)


def save_data(split_data, new_data_path, split_type):
    # save metadata
    #metadata_path = os.path.join(new_data_path, 'metadata', f'{split_type}_metadata.csv')
    #split_data.to_csv(metadata_path, index=False)

    # save filtered signals
    signals_path = os.path.join(new_data_path, f'{split_type}_signals.pt')
    torch.save(ECGDataset(split_data, '', filter_signal), signals_path)

    print(f"{split_type.capitalize()} set size: {len(split_data)}")


def split_and_save(ecg_dataset, new_data_path, split_ratio=0.1, random_state=42):
    '''
    Function that splits the data and saves it to new folders

    Parameters:
    - ecg_dataset (ECGDataset): The dataset to be split.
    - new_data_path (str): Path to the directory where the new data will be saved.
    - split_ratio (float): The ratio of data to be used for testing.
    - random_state (int): Random seed for reproducibility.
    '''

    train_data, test_data = train_test_split(ecg_dataset.metadata, test_size=split_ratio, random_state=random_state)

    os.makedirs(os.path.join(new_data_path, 'metadata'), exist_ok=True)
    os.makedirs(os.path.join(new_data_path, 'filtered_signals'), exist_ok=True)

    save_data(train_data, new_data_path, 'train')

    save_data(test_data, new_data_path, 'test')



def plot_comparison(ecg_dataset, idx):
    record_info = ecg_dataset.metadata.iloc[idx]
    record_path_hr = os.path.join(ecg_dataset.data_path, f"{record_info['filename_hr']}")
    record = wfdb.rdrecord(record_path_hr)
    ecg_signal = record.p_signal[:, 0]  # Assuming the first lead (column) is used

    filtered_ecg = ecg_dataset.filter_fn(ecg_signal, record.fs)

    t = np.arange(0, len(ecg_signal)) / record.fs

    plt.figure(figsize=(12, 6))
    plt.plot(t, ecg_signal, 'k', label='Original ECG')
    plt.plot(t, filtered_ecg, 'm', label='Filtered ECG')
    plt.title(f"Original and Filtered ECG Signals fo Patient {record_info['patient_id']}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 10)
    plt.legend()
    plt.show()
