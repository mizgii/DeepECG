import os
import pandas as pd
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


# # saving the new data
#os.makedirs(new_data_path, exist_ok=True)
#new_metadata_path = os.path.join(new_data_path, 'ptbxl_database_healthy.csv')
#healthy_patients.to_csv(new_metadata_path, index=False)


#---------example for  filtering------------

# SOMETHING DOES NOT WORK HERE YET, FILTERING IS WRONG

first_healthy_patient = healthy_patients_unique.iloc[0]
record_path = os.path.join(data_path, f"{first_healthy_patient['filename_hr']}")
record = wfdb.rdrecord(record_path)

ecg_signal = record.p_signal[:, 0]  # Assuming the first lead (column) is used

fs = record.fs

filtered = filter_signal(ecg_signal, fs)

plt.figure(figsize=(12, 6))
plt.plot(ecg_signal, label='Original ECG')
plt.plot(filtered, label='Filtered ECG - STH IS WRONG IM GONNA FIX IT')
plt.title('Original and Filtered ECG Signals - FILTERING IS WRONG FOR NOW')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
