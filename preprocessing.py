from functions import filter_signal, open_data, split_and_save, plot_comparison


def preprocessing(data_path, new_data_path, plot_filtering = True, id = 0):
    
    ecg_dataset = open_data(data_path, filter_signal)

    if plot_filtering:
        plot_comparison(ecg_dataset, id)

    split_and_save(ecg_dataset, new_data_path)