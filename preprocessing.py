from functions import filter_signal, open_data, split_and_save, plot_comparison


def preprocessing(data_path, new_data_path, plot_filtering = False, id = 0):
    
    ecg_dataset = open_data(data_path, filter_signal)

    if plot_filtering:
        plot_comparison(ecg_dataset, id)

    split_and_save(ecg_dataset, new_data_path)
    


if __name__ == "__main__":

    # path to the directory where your data is stored 
    data_path = r'c:\Users\mizgi\Desktop\gowno\studia\erasmus\a_lab_bisp\ptb-xl'
    
    # path to directory where you want to store limited data after preprocessing
    new_data_path = r'c:\Users\mizgi\Desktop\gowno\studia\erasmus\a_lab_bisp\DeepECG\ptb-xl-new'

    preprocessing(data_path, new_data_path, plot_filtering=False)
