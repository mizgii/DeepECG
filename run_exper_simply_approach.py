import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split 
import HolterDataset as HD
import neural
from torch import nn
import torch
import load

from training import train_model, evaluate_model

device = "cuda" if torch.cuda.is_available() else "cpu" #should be in a place reachable by run function or in the function
ids = ['1911', '2012']#dataset.extract_patient_ids("files/info.txt")
#ids_mapped = {id: i for i, id in enumerate(ids)}
#ecg = np.load("cache/1911_signal.npy")

data_path = "cache/"

ecgs = load.load_ecgs(data_path, ids)

def run(NUM_FINETRE, NUM_SECONDI, NUM_BATCH, NUM_LEADS, NUM_EPOCHS, NUM_SOGGETTI):
    '''
    what I did (14.02.24):
    - switched to torchmetrics accuracy 
    - put iteration over epochs inside training function (thats not so important and its a personal preference)
    - added time calculation for each epoch and the whole training process
    - wrapped the model evaluation in a function and transported it to training.py file
    small stuff: 
    - adding loss.item() for memory efficency
    - changed calculation of test_size
    '''

    dataset= HD.ECGDataset(ecgs=ecgs[:NUM_SOGGETTI,:NUM_LEADS,:], ids=ids, fs=128, n_windows=NUM_FINETRE, seconds=NUM_SECONDI)
    train_ratio = 0.5
    #test_ratio = 0.5
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size # more robust than int(test_ratio * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=NUM_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NUM_BATCH, shuffle=False)

    input_shape = NUM_LEADS
    output_shape = len(dataset.classes) 
    hidden_units = 32 

    model = neural.DeepECG_DUMMY(input_shape, hidden_units, output_shape)
    dummy_sig, _, _ = train_dataset[:]
    final = model(dummy_sig)

    model=neural.DeepECG(input_shape, hidden_units, output_shape, final).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


    train_model(model=model,
                data_loader=train_loader, 
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device, 
                num_epochs=NUM_EPOCHS
                )
        
            
    accuracy = evaluate_model(model=model, 
                              test_loader=test_loader,
                              loss_fn=loss_fn,
                              device=device,
                              )
    return accuracy

accuracy= run(NUM_FINETRE=50, NUM_SECONDI=2, NUM_BATCH=16, NUM_LEADS=3, NUM_EPOCHS=10, NUM_SOGGETTI=70)


print(accuracy)
