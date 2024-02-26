import os
import re
import wfdb
import numpy as np
from wfdb import processing
import sys

from tqdm import tqdm
from scipy.signal import butter, iirnotch, filtfilt
from sklearn.decomposition import PCA

from ecgdetectors import Detectors #https://pypi.org/project/py-ecg-detectors/


def interpolate_nans(record):
    '''
    interpolate over NaN values in each lead of a given signal

    Parameters:
    - record (numpy.ndarray): 2D numpy array representing the ECG signal with leads in rows
                                shape (channels x samples)

    Returns:
    -numpy.ndarray: the signal with NaNs interpolated, leads with all NaNs remain unchanged.
    '''

    for i in range(record.shape[0]):  ###for future: instead of looping through each lead, applying interpolation in a vectorized manner
        lead_signal = record[i, :]

        if np.isnan(lead_signal).all():
            print(f"Warning: Lead {i} contains only NaNs.")
            continue

        if np.isnan(lead_signal).any():
            valid_mask = ~np.isnan(lead_signal)
            record[i, :] = np.interp(
                np.arange(len(lead_signal)),
                valid_mask.nonzero()[0],
                lead_signal[valid_mask]
            )
    return record

def filter_signal(signal, fs):
    """
    apply a high-pass and notch filter to an ECG signal

    Parameters:
    signal (array): the ECG signal to filter with shape (channels x samples)
                    can be 1D (single lead) or 2D (multichannel)
    fs (int): sampling frequency of the signal

    Returns:
    array or None: The filtered signal or None if an error occurs.
    """
    try:
        if np.isnan(signal).any():  # if the interpolation didn't work this will catch it
            print("NaN values detected during filtering, interpolating...")
            signal = np.nan_to_num(signal)  # Replace NaNs with zeros

        [b, a] = butter(3, (0.5, 40), btype='bandpass', fs=fs)
        signal = filtfilt(b, a, signal, axis=1)
        [bn, an] = iirnotch(50, 3, fs=fs)
        signal = filtfilt(bn, an, signal, axis=1)

        if signal.size == 0:
            print("Warning: Filtered signal is empty.")

        return signal
    except Exception as e:
        print(f"An error occurred during filtering: {e}")
        return None

#I'm using regular expression for this one coz info.txt is messy
def extract_patient_ids(info_path):
    '''
    extracts patient IDs from info.txt file
    
    returns:
    - list of IDs
    '''
    patient_ids = []
    id_pattern = re.compile(r'^\d{4}\b')

    with open(info_path, 'r') as file:
        for line in file:
            match = id_pattern.match(line)
            if match:
                patient_id = match.group()
                patient_ids.append(patient_id)

    return patient_ids


def preprocess_and_save(data_path, save_path):
    '''
    Preprocesses ECG signal data and saves the filtered signals and QRS complex indices.

    steps:
    - itaration over ECG records specified in 'info.txt'
    - NaN interpolation and filtering of the signals
    - QRS detection in a two-step process: 
        - an initial detection using the Pan-Tompkins algorithm
        - refinement with WFDB's `correct_peaks`

    parameters:
    - data_path (str): path to database
    - save_path (str): where to store preprocessed signals

    signals are saved in separate files with shape (number of channels x number of samples)
    '''

    os.makedirs(save_path, exist_ok=True)

    info_path = os.path.join(data_path, 'info.txt')
    patient_ids = extract_patient_ids(info_path)

    patient_ids_path = os.path.join(save_path, 'patient_ids.txt')
    with open(patient_ids_path, 'w') as f:
        for id in patient_ids:
            f.write(f"{id}\n")

    for patient_id in tqdm(patient_ids, desc='Processing Patients'):
        record_path = os.path.join(data_path, f"0{patient_id}") #I'm adding a zero before id coz thats how the files are saved
        save_signals_file = os.path.join(save_path, f"{patient_id}_signal.npy")
        save_qrs_file = os.path.join(save_path, f"{patient_id}_qrs.npy")

        record = wfdb.rdrecord(record_path)
        fs = record.fs
        signals = record.p_signal[(fs*60*5):, :].T #getting the shape (number of channels x number of samples)
        interpolated_signals = interpolate_nans(signals)

        filtered_signals = filter_signal(interpolated_signals, record.fs)
        if filtered_signals is None:
            print(f"Filtering failed for patient ID {patient_id}. Skipping QRS detection.")
            continue

        ### for future: fs is constant so filter coefficients don't have to be recalculated for every call
        ### move filtering from seperate function to preprocess_and_save

        #its way faster to find approximate peaks and correct them with wfdb rather than doing the whole search with wfdb
        detectors = Detectors(record.fs) 
        qrs_inds = detectors.pan_tompkins_detector(interpolated_signals[0])
        corrected_peak_inds = processing.correct_peaks(filtered_signals[0],
                                                    peak_inds=qrs_inds,
                                                    search_radius=int(record.fs*0.2),
                                                    smooth_window_size=int(record.fs*0.1))
        
        np.save(save_signals_file, filtered_signals)
        np.save(save_qrs_file, corrected_peak_inds)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dataset.py <data_path> <save_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    save_path = sys.argv[2]

    preprocess_and_save(data_path, save_path)