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
    
    def cut_signal(self, signal):
        return [signal[:5*500], signal[5*500:]]

    def __getitem__(self, idx):

        record_info = self.metadata.iloc[idx]
        record_path_hr = os.path.join(self.data_path, f"{record_info['filename_hr']}")
        record = wfdb.rdrecord(record_path_hr)

        # assuming the first 3 leads (column) are used
        leads = record.p_signal[:, [0, 1, 2]].T  
        filtered_sigals = [self.filter_fn(lead, record.fs) for lead in leads]
        cut_signals = [self.cut_signal(signal) for signal in filtered_sigals]
        cut_filtered_signals = [item for sublist in cut_signals for item in sublist]
        
        return [torch.Tensor(signal.copy()) for signal in cut_filtered_signals], record_info['patient_id']
    


#--------------------------------------------
#----------------functions---------------------
#--------------------------------------------


def filter_signal(signal, fs=500):

    lowcut = 0.5  # Lower cutoff frequency in Hz

    [b, a] = butter(3, lowcut, btype='highpass', fs=fs)
    signal = filtfilt(b, a, signal, axis=0)
    [bn, an] = iirnotch(50, 3, fs=fs)
    signal = filtfilt(bn, an, signal, axis=0)

    return signal


def open_data(data_path, filter_fn):
    metadata_path = os.path.join(data_path, 'ptbxl_database.csv')
    metadata = pd.read_csv(metadata_path)
    
    # we're only using healthy patients
    healthy_patients = metadata[metadata['scp_codes'].str.contains('NORM', na=False)]
    #healthy_patients = healthy_patients[:10]

    print(f"Number of recordings: {len(metadata)}\nNumber of healthy recordings: {len(healthy_patients)}\nNumber of healthy patients: {len(healthy_patients.drop_duplicates(subset='patient_id', keep='first'))}")

    return ECGDataset(healthy_patients, data_path, filter_fn)


def save_data(split_data, new_data_path, split_type):
    # save metadata
    metadata_path = os.path.join(new_data_path, 'metadata', f'{split_type}_metadata.csv')
    split_data.to_csv(metadata_path, index=False)

    # save filtered signals
    signals_path = os.path.join(new_data_path, 'filtered_signals', f'{split_type}_signals.pt')
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
