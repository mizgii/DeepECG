import time
import torch
from torchmetrics import Accuracy


def train_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, #criterion
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               num_epochs,
               output_shape):
    
    start_time = time.time()
    
    accuracy_metric = Accuracy(num_classes= output_shape, task='multiclass').to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        model.train() #making sure model is in training mode
        train_loss = 0
        for signal, class_label in data_loader: 
            signal, class_label = signal.to(device), class_label.to(device) #
            train_pred = model(signal)
            loss = loss_fn(train_pred, class_label)
            train_loss += loss.item() #converts a tensor to a Python scalar, more memory efficent

            accuracy_metric(train_pred, class_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = accuracy_metric.compute() * 100  
        print(f"Train loss: {train_loss / len(data_loader):.5f} | Train accuracy: {train_acc:.2f}%")
        accuracy_metric.reset()

        epoch_time = int((time.time() - start_time) // 60)  
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time} minutes")

    total_time = int((time.time() - start_time)//60)
    print(f"Total training time: {total_time} minutes")


def evaluate_model(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader, 
                   loss_fn: torch.nn.Module, #criterion
                   device: torch.device,
                   output_shape):

    test_loss = 0
    accuracy_metric = Accuracy(num_classes=output_shape, task='multiclass').to(device)

    model.eval()#
    with torch.inference_mode(): 
        for signal, class_label in test_loader:
            signal, class_label = signal.to(device), class_label.to(device)
            test_pred = model(signal)
            loss = loss_fn(test_pred, class_label)
            test_loss +=loss.item()

            accuracy_metric(test_pred, class_label)

    test_acc = accuracy_metric.compute() * 100  
    print(f"Test loss: {test_loss/len(test_loader):.5f} | Test accuracy: {test_acc:.2f}%\n")
    accuracy_metric.reset() 

    return test_acc.item()