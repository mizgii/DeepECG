from torch.utils.data import DataLoader
from class_ecgdataset import ECGDataset
import neural
from torch import nn
import torch

from training import train_model, evaluate_model


ids = ['1911', '2012'] #dataset.extract_patient_ids("files/info.txt")
#ids_mapped = {id: i for i, id in enumerate(ids)}
data_path = 'path/to/data'


def run(NUM_FINETRE, NUM_SECONDI, NUM_BATCH, LEADS, NUM_EPOCHS, DATA_PATH, PATIENT_IDS, FS):

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    train_dataset = ECGDataset(DATA_PATH, PATIENT_IDS, FS, NUM_FINETRE, NUM_SECONDI, LEADS, 'training')
    test_dataset = ECGDataset(DATA_PATH, PATIENT_IDS, FS, NUM_FINETRE, NUM_SECONDI, LEADS, 'test')

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

accuracy= run(NUM_FINETRE=50, NUM_SECONDI=2, NUM_BATCH=16, LEADS=[0,1,2], NUM_EPOCHS=10, DATA_PATH = data_path, PATIENT_IDS =ids, FS=128)


print(accuracy)

