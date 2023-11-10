import os
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt

from preprocessing import filter_signal

# path to the directory where your data is stored 
data_path = r'c:\Users\mizgi\Desktop\gowno\studia\erasmus\a_lab_bisp\ptb-xl'
# path to directory where you want to store limited data after preprocessing
new_data_path = r'c:\Users\mizgi\Desktop\gowno\studia\erasmus\a_lab_bisp\DeepECG\ptb-xl'


metadata_path = os.path.join(data_path, 'ptbxl_database.csv')
metadata = pd.read_csv(metadata_path)

# we're only using healthy patients
healthy_patients = metadata[metadata['scp_codes'].str.contains('NORM', na=False)]
#droping duplicates - only one record from one patient
healthy_patients_unique = healthy_patients.drop_duplicates(subset='patient_id', keep='first')

print(f'Number of recordings: {len(metadata)}\nNumber of healthy recordings: {len(healthy_patients)}\nNumber of healthy patients: {len(healthy_patients_unique)}')



#---------example for  filtering------------

first_healthy_patient = healthy_patients_unique.iloc[0]
record_path = os.path.join(data_path, f"{first_healthy_patient['filename_hr']}")
record = wfdb.rdrecord(record_path)

ecg_signal = record.p_signal[:, 0]  # Assuming the first lead (column) is used
print(type(ecg_signal))

fs = record.fs
t = np.arange(0,len(ecg_signal))/fs

filtered = filter_signal(ecg_signal, fs)


plt.figure(figsize=(12, 6))
plt.plot(t, ecg_signal, 'k', label='Original ECG')
plt.plot(t, filtered, 'm', label='Filtered ECG')
plt.title('Original and Filtered ECG Signals')
plt.xlim(0,10)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.show()
