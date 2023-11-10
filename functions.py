import os
import wfdb
import numpy as np
import pandas as pd
from  scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


#--------------------------------------------
#----------------classes---------------------
#--------------------------------------------


class ECGDataset(Dataset):
    def __init__(self, metadata, data_path, filter_fn):
        self.metadata = metadata
        self.data_path = data_path
        self.filter_fn = filter_fn
        self.train_index = None
        self.test_index = None   
        self.train_set = None    
        self.test_set = None 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        record_info = self.metadata.iloc[idx]
        record_path_hr = os.path.join(self.data_path, f"{record_info['filename_hr']}")
        record = wfdb.rdrecord(record_path_hr)
        ecg_signal = record.p_signal[:, 0]  # assuming the first lead (column) is used

        # apply the filters
        filtered_ecg = self.filter_fn(ecg_signal, record.fs)

        return torch.Tensor(filtered_ecg), record_info['patient_id']
    
    def set_train_test_split(self, train_index, test_index):
        self.train_index = train_index
        self.test_index = test_index
        self.train_set = torch.utils.data.Subset(self, train_index)
        self.test_set = torch.utils.data.Subset(self, test_index)

    def plot_comparison(self, idx):
        record_info = self.metadata.iloc[idx]
        record_path_hr = os.path.join(self.data_path, f"{record_info['filename_hr']}")
        record = wfdb.rdrecord(record_path_hr)
        ecg_signal = record.p_signal[:, 0]  # Assuming the first lead (column) is used

        # Apply the notch filter
        filtered_ecg = self.filter_fn(ecg_signal, record.fs)

        t = np.arange(0,len(ecg_signal))/record.fs

        # Plot original and filtered signals
        plt.figure(figsize=(12, 6))
        plt.plot(t, ecg_signal, 'k', label='Original ECG')
        plt.plot(t, filtered_ecg, 'm', label='Filtered ECG')
        plt.title(f"Original and Filtered ECG Signals fo Patient {record_info['patient_id']}")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.ylabel('Amplitude')
        plt.xlim(0,10)
        plt.legend()
        plt.show()



#--------------------------------------------
#----------------functions---------------------
#--------------------------------------------


def filter_signal(signal, fs=500, plot_example = False):

    lowcut = 0.5  # Lower cutoff frequency in Hz

    [b, a] = butter(3, lowcut, btype='highpass', fs=fs)
    signal = filtfilt(b, a, signal, axis=0)
    [bn,an] = iirnotch(50, 3, fs=fs)
    signal = filtfilt(bn, an, signal, axis=0)

    return signal


def open_and_filter_data(data_path, filter_fn):
    metadata_path = os.path.join(data_path, 'ptbxl_database.csv')
    metadata = pd.read_csv(metadata_path)
    
    # we're only using healthy patients
    healthy_patients = metadata[metadata['scp_codes'].str.contains('NORM', na=False)]

    print(f"Number of recordings: {len(metadata)}\nNumber of healthy recordings: {len(healthy_patients)}\nNumber of healthy patients: {len(healthy_patients.drop_duplicates(subset='patient_id', keep='first'))}")

    ecg_dataset = ECGDataset(healthy_patients, data_path, filter_fn)

    return ecg_dataset



def save_data(ecg_dataset, new_data_path, fold):
    # Save metadata
    train_metadata_path = os.path.join(new_data_path, 'metadata', f'train_metadata_fold_{fold}.csv')
    test_metadata_path = os.path.join(new_data_path, 'metadata', f'test_metadata_fold_{fold}.csv')

    train_metadata = ecg_dataset.metadata.iloc[ecg_dataset.train_index]
    test_metadata = ecg_dataset.metadata.iloc[ecg_dataset.test_index]

    train_metadata.to_csv(train_metadata_path, index=False)
    test_metadata.to_csv(test_metadata_path, index=False)

    # Save filtered signals
    train_signals_path = os.path.join(new_data_path, 'filtered_signals', f'train_signals_fold_{fold}.pt')
    test_signals_path = os.path.join(new_data_path, 'filtered_signals', f'test_signals_fold_{fold}.pt')

    torch.save(ecg_dataset.train_set, train_signals_path)
    torch.save(ecg_dataset.test_set, test_signals_path)

    print(f"Fold {fold + 1}: Train set size: {len(ecg_dataset.train_set)}, Test set size: {len(ecg_dataset.test_set)}")


