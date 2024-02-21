
import os
import time
import torch
import random
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import neural
from class_ecgdataset import ECGDataset
from class_deepdataset import DeepDatasetV2
from train_and_eval import train_model, evaluate_model


def run(NUM_SEGMENTS, NUM_SECONDS, NUM_BATCH, LEADS, NUM_EPOCHS, DATA_PATH, FS, NUM_PATIENTS=None):
    '''
    Function that:
    - loads ECG data, prepares the dataset, performs a stratified train-test split
    - trains a DeepECG model and evaluates its performance

    It operates on a specified number of subjects (if provided), segments, and leads.
   
    Returns:
    - accuracy (float): the accuracy of the model on the test set
    - total_train_time (int): the total training time in seconds.
    - total_eval_time (int): the total evaluation time in seconds.
    '''

    torch.manual_seed(44)
    random.seed(44)

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    print(f'Used device: {device}')
    print('Loading signals...')

    patient_ids_path = os.path.join(DATA_PATH, 'patient_ids.txt')
    with open(patient_ids_path, 'r') as f:
        PATIENT_IDS = [line.strip() for line in f.readlines()]

    if NUM_PATIENTS is not None:
        PATIENT_IDS = PATIENT_IDS[:NUM_PATIENTS]

    full_dataset = ECGDataset(DATA_PATH, PATIENT_IDS, FS, NUM_SEGMENTS, NUM_SECONDS, LEADS)

    # stratified split to maintain equal distribution of each patient's data
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.5, stratify=full_dataset.labels)
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=NUM_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NUM_BATCH, shuffle=False)

    single_batch_segments, _ = next(iter(train_loader))
    output_shape = len(PATIENT_IDS) 
    hidden_units = 32 
    
    dummy_network = neural.DeepECG_DUMMY(len(LEADS), hidden_units, output_shape).to(device)
    single_batch_segments = single_batch_segments.to(device)
    final_features = dummy_network(single_batch_segments)

    model = neural.DeepECG(len(LEADS), hidden_units, output_shape, final_features).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


    total_train_time = train_model(model=model,
                                   data_loader=train_loader, 
                                   loss_fn=loss_fn,
                                   optimizer=optimizer,
                                   device=device, 
                                   num_epochs=NUM_EPOCHS,
                                   output_shape = output_shape
                                   )
                    
            
    accuracy, total_eval_time = evaluate_model(model=model, 
                                               test_loader=test_loader,
                                               loss_fn=loss_fn,
                                               device=device,
                                               output_shape = output_shape
                                               )
    return accuracy, total_train_time, total_eval_time




def run_deepecg(NUM_BATCH, LEAD, NUM_EPOCHS, DATA_PATH, FS):
    '''
    Function that:
    - loads ECG data, prepares the dataset using DeepECG methodology, performs a stratified train-test split
    - trains a DeepECG model and evaluates its performance

    It operates on a specified lead of ECG recordings.
   
    Returns:
    - accuracy (float): the accuracy of the model on the test set
    - total_train_time (int): the total training time in seconds.
    - total_eval_time (int): the total evaluation time in seconds.
    '''

    torch.manual_seed(44)
    random.seed(44)

    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    print(f'Used device: {device}')
    print('Loading signals...')
    start_time = time.time()

    patient_ids_path = os.path.join(DATA_PATH, 'patient_ids.txt')
    with open(patient_ids_path, 'r') as f:
        PATIENT_IDS = [line.strip() for line in f.readlines()]

    full_dataset = DeepDatasetV2(DATA_PATH, PATIENT_IDS, FS, LEAD)

    # stratified split to maintain equal distribution of each patient's data
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.5, stratify=full_dataset.labels)
      
    total_time = int((time.time() - start_time))
    print(f"Total loading time: {total_time//60} minutes and {total_time%60} seconds\n")

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=NUM_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NUM_BATCH, shuffle=False)

    single_batch_segments, _ = next(iter(train_loader))
    num_leads = 1
    output_shape = len(PATIENT_IDS) 
    hidden_units = 32 
    
    dummy_network = neural.DeepECG_DUMMY(num_leads, hidden_units, output_shape).to(device)
    single_batch_segments = single_batch_segments.to(device)
    final_features = dummy_network(single_batch_segments)

    model = neural.DeepECG(num_leads, hidden_units, output_shape, final_features).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


    total_train_time = train_model(model=model,
                                   data_loader=train_loader, 
                                   loss_fn=loss_fn,
                                   optimizer=optimizer,
                                   device=device, 
                                   num_epochs=NUM_EPOCHS,
                                   output_shape = output_shape
                                   )
                    
            
    accuracy, total_eval_time = evaluate_model(model=model, 
                                               test_loader=test_loader,
                                               loss_fn=loss_fn,
                                               device=device,
                                               output_shape = output_shape
                                               )

    return accuracy, total_train_time, total_eval_time