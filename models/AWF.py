import torch.nn as nn
import torch
import torch.nn.functional as F


class AWF_CNN(nn.Module):
    def __init__(self, len, nb_class):
        super(AWF_CNN, self).__init__()
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.feature_size = self._get_conv_output(len)
        self.fc = nn.Linear(self.feature_size, nb_class)

    def _get_conv_output(self, len):
        with torch.no_grad():
            input = torch.rand(1, 1, len)
            input = self.dropout1(input)
            input = self.pool1(F.relu(self.conv1(input)))
            input = self.dropout2(self.pool2(F.relu(self.conv2(input))))
            input = self.flatten(input)
            return input.shape[1]

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


