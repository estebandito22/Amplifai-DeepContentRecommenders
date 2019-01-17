"""PyTorch classes for the audio model component in DCUE."""

from torch import nn


class TrueDcueNetMel1D(nn.Module):

    """ConvNet used on data prepared with melspectogram transform."""

    def __init__(self, dict_args):
        """
        Initialize ConvNetMel1D.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(TrueDcueNetMel1D, self).__init__()
        self.output_size = dict_args["output_size"]
        self.hidden_size = dict_args["hidden_size"]
        # input_size = batch size x 128 x 131
        self.layer1 = nn.Conv1d(
            in_channels=128, out_channels=self.hidden_size,
            kernel_size=4, stride=1, padding=2, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        # batch size x 128 x 33

        self.layer2 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.hidden_size,
            kernel_size=4, stride=1, padding=2, bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        # batch size x 64 x 8

        self.layer3 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.hidden_size,
            kernel_size=4, stride=1, padding=2, bias=True)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        # batch size x 128 x 2

        self.layer4 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.hidden_size,
            kernel_size=2, stride=1, padding=1, bias=True)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        # batch size x 128 x 1

        self.layer5 = nn.Conv1d(
            in_channels=self.hidden_size, out_channels=self.output_size,
            kernel_size=1, stride=1, bias=True)
        self.relu5 = nn.ReLU()
        # self.pool5 = nn.AvgPool1d(kernel_size=4)
        # batch size x self.hidden_size x 1

        self.fc = nn.Linear(self.output_size, self.output_size)

        # initizlize weights
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer3.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer4.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer5.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """Execute forward pass."""
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        x = self.relu4(x)
        x = self.layer5(x)
        # x = self.pool5(x)
        x = self.relu5(x)

        return self.fc(x.permute(0, 2, 1)).squeeze()
