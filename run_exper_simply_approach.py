import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split 
import HolterDataset as HD
import neural
from torch import nn
import torch
import training
import metric
import load

device = "cuda" if torch.cuda.is_available() else "cpu"
ids = ['1911', '2012']#dataset.extract_patient_ids("files/info.txt")
#ids_mapped = {id: i for i, id in enumerate(ids)}
#ecg = np.load("cache/1911_signal.npy")

data_path = "cache/"

ecgs = load.load_ecgs(data_path, ids)

def run(NUM_FINETRE, NUM_SECONDI, NUM_BATCH, NUM_LEADS, NUM_EPOCHS, NUM_SOGGETTI):

    dataset= HD.ECGDataset(ecgs=ecgs[:NUM_SOGGETTI,:NUM_LEADS,:], ids=ids, fs=128, n_windows=NUM_FINETRE, seconds=NUM_SECONDI)
    train_ratio = 0.5
    test_ratio = 0.5
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
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

    epochs = NUM_EPOCHS


    for epoch in range(epochs):
        if epoch % 10 == 0:
            print(f"-------\n Epoch: {epoch}")

        training.train_step(data_loader=train_loader, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=metric.accuracy_fn,
            device=device, 
            epoch=epoch
        )
        
            

    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode(): 
        for signal, _, class_label in test_loader:
            signal, class_label = signal.to(device), class_label.to(device)
            test_pred = model(signal)
            test_loss += loss_fn(test_pred, class_label)

            p = torch.softmax(test_pred, dim=1)
            class_est = torch.argmax(p, dim=1)
            test_acc += metric.accuracy_fn(y_true=class_label, y_pred=class_est)
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return test_acc

accuracy= run(NUM_FINETRE=50, NUM_SECONDI=2, NUM_BATCH=16, NUM_LEADS=3, NUM_EPOCHS=10, NUM_SOGGETTI=70)


print(accuracy)
