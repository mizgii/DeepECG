import torch

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device,
               epoch):
    train_loss, train_acc = 0, 0
    model.to(device)
    for signal,  _, class_label in data_loader:
        signal, class_label = signal.to(device), class_label.to(device)
        train_pred = model(signal)
        loss = loss_fn(train_pred, class_label)
        train_loss += loss

        p = torch.softmax(train_pred, dim=1)
        class_est = torch.argmax(p, dim=1)
        train_acc += accuracy_fn(y_true=class_label, y_pred=class_est)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    if epoch % 10 == 0:
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
