from preprocessing import preprocessing
import time

if __name__ == "__main__":

    # path to the directory where your data is stored 
    data_path = r'c:\Users\mizgi\Desktop\gowno\studia\erasmus\a_lab_bisp\ptb-xl'
    
    # path to directory where you want to store limited data after preprocessing
    new_data_path = r'c:\Users\mizgi\Desktop\gowno\studia\erasmus\a_lab_bisp\DeepECG\ptb-xl-new'

    preprocessing(data_path, new_data_path, plot_filtering=True, id=10)