
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import neural
from class_ecgdataset import ECGDataset_halfday, ECGDataset
from training import train_model, evaluate_model


def run(NUM_FINETRE, NUM_SECONDI, NUM_BATCH, LEADS, NUM_EPOCHS, DATA_PATH, FS):

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    patient_ids_path = os.path.join(DATA_PATH, 'patient_ids.txt')
    with open(patient_ids_path, 'r') as f:
        PATIENT_IDS = [line.strip() for line in f.readlines()]

    full_dataset = ECGDataset(DATA_PATH, PATIENT_IDS, FS, NUM_FINETRE, NUM_SECONDI, LEADS)

    labels = []
    for patient_id in PATIENT_IDS:
        labels.extend([patient_id] * NUM_FINETRE)

    # stratified split to maintain equal distribution of each patient's data
    train_indices, test_indices = train_test_split(range(len(full_dataset)), test_size=0.5, stratify=labels)

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of test samples: {len(test_dataset)}')

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


    train_model(model=model,
                data_loader=train_loader, 
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device, 
                num_epochs=NUM_EPOCHS,
                output_shape = output_shape
                )
        
            
    accuracy = evaluate_model(model=model, 
                              test_loader=test_loader,
                              loss_fn=loss_fn,
                              device=device,
                              output_shape = output_shape
                              )
    return accuracy


def run_halfday(NUM_FINETRE, NUM_SECONDI, NUM_BATCH, LEADS, NUM_EPOCHS, DATA_PATH, FS):

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    patient_ids_path = os.path.join(DATA_PATH, 'patient_ids.txt')
    with open(patient_ids_path, 'r') as f:
        PATIENT_IDS = [line.strip() for line in f.readlines()]

    train_dataset = ECGDataset_halfday(DATA_PATH, PATIENT_IDS, FS, NUM_FINETRE, NUM_SECONDI, LEADS, 'training')
    test_dataset = ECGDataset_halfday(DATA_PATH, PATIENT_IDS, FS, NUM_FINETRE, NUM_SECONDI, LEADS, 'test')
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of test samples: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=NUM_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NUM_BATCH, shuffle=False)

    single_batch_segments, _ = next(iter(train_loader))

    output_shape = len(PATIENT_IDS) 
    hidden_units = 32 

    # Initialize the dummy network and pass the single batch through
    dummy_network = neural.DeepECG_DUMMY(len(LEADS), hidden_units, output_shape).to(device)
    single_batch_segments = single_batch_segments.to(device)
    final_features = dummy_network(single_batch_segments)

    # Initialize the main model with the inferred number of features
    model = neural.DeepECG(len(LEADS), hidden_units, output_shape, final_features).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


    train_model(model=model,
                data_loader=train_loader, 
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device, 
                num_epochs=NUM_EPOCHS,
                output_shape = output_shape
                )
        
            
    accuracy = evaluate_model(model=model, 
                              test_loader=test_loader,
                              loss_fn=loss_fn,
                              device=device,
                              output_shape = output_shape
                              )
    return accuracy

