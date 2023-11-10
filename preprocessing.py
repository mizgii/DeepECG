import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from functions import filter_signal, open_and_filter_data, save_data


def preprocessing(data_path, new_data_path, plot_filtering = True, id = 0):
    '''
    Function that
    - opens the data
    - chooses only the healhy patients
    - filters the signals
    - divides data into test and train
    - saves preprocessed data to new folders
    '''

    ecg_dataset = open_and_filter_data(data_path, filter_signal)

    if plot_filtering == True:
        ecg_dataset.plot_comparison(id)
        #plt.show()


    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # path to directory for train/test data
    os.makedirs(os.path.join(new_data_path, 'metadata'), exist_ok=True)
    os.makedirs(os.path.join(new_data_path, 'filtered_signals'), exist_ok=True)

    # iterate over folds for train/test split and save the metadata+signals
    for fold, (train_index, test_index) in enumerate(kf.split(ecg_dataset.metadata)):
        ecg_dataset.set_train_test_split(train_index, test_index)
        save_data(ecg_dataset, new_data_path, fold)
