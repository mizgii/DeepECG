import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch.nn as nn

class ECGNet(nn.Module):
    def __init__(self, input_channels, hidden_units, num_classes):
        super(ECGNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, hidden_units, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(hidden_units, hidden_units * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(hidden_units * 2, hidden_units * 4, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(hidden_units * 4, hidden_units * 8, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(hidden_units * 8 * 4, hidden_units * 16)  
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units * 16, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        #softmax?


    def forward(self, x):
        # Input shape: (batch_size, channels, length)
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.lrn1(self.relu2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.conv3(x)))
        x = self.maxpool4(self.lrn2(self.relu4(self.conv4(x))))

        x = x.view(-1, self.hidden_units * 8 * 4)  #flattening

        x = self.dropout(self.relu5(self.fc1(x)))
        x = self.fc2(x)

        return x
    
#try on data


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



def load_data(data_path, split_type):
    pass


train_data = None
test_data = None

train_loader = None #DataLoader(ECGDataset(train_data, data_path, filter_signal), batch_size=32, shuffle=True)
test_loader = None #DataLoader(ECGDataset(test_data, data_path, filter_signal), batch_size=32, shuffle=False)


num_classes = len(train_data['patient_id'].unique())
input_channels = None
hidden_units = None
cnn_model = ECGNet(input_channels, hidden_units, num_classes)

criterion = nn.CrossEntropyLoss() #if it aplies softmax, if not add it in model
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001) #defining optimizer

train_model(cnn_model, train_loader, criterion, optimizer, num_epochs=10)

evaluate_model(cnn_model, test_loader)

#perform the training