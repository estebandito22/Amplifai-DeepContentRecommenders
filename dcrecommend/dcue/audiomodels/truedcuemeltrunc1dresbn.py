"""PyTorch classes for the audio model component in DCUE."""

import torch
from torch import nn


class TrueDcueNetMelTrunc1DResBn(nn.Module):

    """ConvNet used on data prepared with melspectogram transform."""

    def __init__(self, dict_args):
        """
        Initialize ConvNetMel1D.

        Args
            dict_args: dictionary containing the following keys:
                hidden_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(TrueDcueNetMelTrunc1DResBn, self).__init__()
        self.hidden_size = dict_args["hidden_size"]
        # input_size = batch size x 128 x 131
        self.bn0 = nn.BatchNorm1d(128)
        self.layer1 = nn.Conv1d(
            in_channels=128, out_channels=self.hidden_size, kernel_size=4,
            stride=1, padding=2, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.timepool1 = nn.AvgPool1d(kernel_size=33)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        # batch size x 128 x 33

        self.layer2 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.hidden_size,
            kernel_size=4, stride=1, padding=2, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.timepool2 = nn.AvgPool1d(kernel_size=8)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        # batch size x 128 x 8

        self.layer3 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.hidden_size,
            kernel_size=4, stride=1, padding=2, bias=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.timepool3 = nn.AvgPool1d(kernel_size=2)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        # batch size x 128 x 2

        self.layer4 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.hidden_size,
            kernel_size=2, stride=1, padding=1, bias=True)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.timepool4 = nn.AvgPool1d(kernel_size=1)
        self.bn4 = nn.BatchNorm1d(self.hidden_size)
        # batch size x 128 x 1

        self.outsize = [self.hidden_size, 4]

        # initizlize weights
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer4.weight, nonlinearity='relu')

    def forward(self, x):
        """Execute forward pass."""
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        tp1 = self.timepool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        tp2 = self.timepool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        tp3 = self.timepool3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        tp4 = self.timepool4(x)

        return torch.cat([tp1, tp2, tp3, tp4], dim=2)  # batch_size x context_dim (self.hidden_size) x context_size (time dim)
