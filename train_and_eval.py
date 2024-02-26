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

        model.train() 
        train_loss = 0
        for signal, class_label in data_loader: 
            signal, class_label = signal.to(device), class_label.to(device) #
            train_pred = model(signal)
            loss = loss_fn(train_pred, class_label)
            train_loss += loss.item() 

            accuracy_metric(train_pred, class_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = accuracy_metric.compute() * 100  
        print(f"Train loss: {train_loss / len(data_loader):.5f} | Train accuracy: {train_acc:.2f}%")
        accuracy_metric.reset()

    total_time = (time.time() - start_time)
    print(f"\nTotal training time: {total_time} seconds")
    return total_time 


def evaluate_model(model: torch.nn.Module,
                   test_loader: torch.utils.data.DataLoader, 
                   loss_fn: torch.nn.Module, #criterion
                   device: torch.device,
                   output_shape):
    
    start_time = time.time()

    test_loss = 0
    accuracy_metric = Accuracy(num_classes=output_shape, task='multiclass').to(device)

    model.eval()
    with torch.inference_mode(): 
        for signal, class_label in test_loader:
            signal, class_label = signal.to(device), class_label.to(device)
            test_pred = model(signal)
            loss = loss_fn(test_pred, class_label)
            test_loss +=loss.item()

            accuracy_metric(test_pred, class_label)

    test_acc = accuracy_metric.compute() * 100  
    print(f"\nTest loss: {test_loss/len(test_loader):.5f} | Test accuracy: {test_acc:.2f}%")
    accuracy_metric.reset() 

    total_time = (time.time() - start_time)
    print(f"Total evaluation time: {total_time} seconds\n")

    return test_acc.item(), total_time 