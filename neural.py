import torch 
from torch import nn

class DeepECG(nn.Module):
    
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, final):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75, k=1.0),
            nn.MaxPool1d(kernel_size=1,
                         stride=2)
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,
                         stride=2)
        )
        self.block_4 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3, 
                      stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75, k=1.0),
        )
        self.block_5 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU()
        )
        self.block_6 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75, k=1.0),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=final, out_features=output_shape),
        )
        
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.classifier(x)
        return x
    

class DeepECG_DUMMY(nn.Module):
    
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75, k=1.0),
            nn.MaxPool1d(kernel_size=1,
                         stride=2)
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,
                         stride=2)
        )
        self.block_4 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3, 
                      stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75, k=1.0),
        )
        self.block_5 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU()
        )
        self.block_6 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0002, beta=0.75, k=1.0),
            nn.Dropout(0.5)
        )

        
        
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        a = x.shape[1]
        b = x.shape[2]
        return a*b
