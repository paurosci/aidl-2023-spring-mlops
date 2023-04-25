import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
            self, 
            num_inp_channels: int, 
            num_out_fmaps: int,
            kernel_size: int, 
            pool_size: int=2,
            padding: int = None) -> None:
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels=num_inp_channels, 
            out_channels=num_out_fmaps, 
            kernel_size=kernel_size,
            padding=padding)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.conv(x)))

class LinearBlock(nn.Module):

    def __init__(
            self, 
            input_size: int, 
            output_size: int) -> None:

        super().__init__()

        self.linear = nn.Linear(input_size, output_size)  
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.relu(x)
        return x

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(num_inp_channels=1, num_out_fmaps=6, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=6, kernel_size=5)
        self.conv3 = ConvBlock(num_inp_channels=6, num_out_fmaps=6, kernel_size=5)

        self.mlp = nn.Sequential(
            LinearBlock(384, 84),
            nn.Linear(84, 15),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # flatten the output for the MLP
        x = self.mlp(x)
        return x
