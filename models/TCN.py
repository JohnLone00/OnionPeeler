import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_normalization(x):
    # Implement channel normalization in PyTorch
    max_values = torch.max(torch.abs(x), dim=2, keepdim=True)[0] + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    # Implement the WaveNet activation in PyTorch
    tanh_out = torch.tanh(x)
    sigm_out = torch.sigmoid(x)
    return tanh_out * sigm_out


class ResidualBlock(nn.Module):
    def __init__(self, nb_filters, kernel_size, dilation_rate, activation):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(nb_filters, nb_filters, kernel_size, dilation=dilation_rate, padding='same')
        self.spatial_dropout = nn.Dropout(0.05)
        self.conv_1x1 = nn.Conv1d(nb_filters, nb_filters, 1)
        self.activation = activation

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        if self.activation == 'norm_relu':
            x = F.relu(x)
            x = channel_normalization(x)
        elif self.activation == 'wavenet':
            x = wave_net_activation(x)
        else:
            x = F.relu(x)  # Default to ReLU if unspecified
        x = self.spatial_dropout(x)
        x = self.conv_1x1(x)
        res_x = original_x + x
        return res_x, x


class DilatedTCN(nn.Module):
    def __init__(self, kernel_size, dilatations, nb_stacks, nb_filters, input_channels, num_classes,
                 activation='wavenet', use_skip_connections=True):
        super(DilatedTCN, self).__init__()
        self.initial_conv = nn.Conv1d(input_channels, nb_filters, kernel_size, padding='same')
        self.stacks = nn.ModuleList([
            ResidualBlock(nb_filters, kernel_size, 2 ** i, activation) for _ in range(nb_stacks) for i in dilatations
        ])
        self.use_skip_connections = use_skip_connections
        self.final_relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(nb_filters * 2000, num_classes)  # Assuming nb_filters is the same as the output channels of the last Conv1D
        self.dropout_1 = nn.Dropout(0.1)
        self.dense_2 = nn.Linear(num_classes, num_classes)
        self.dropout_2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.initial_conv(x)
        skip_connections = []
        for block in self.stacks:
            x, skip_out = block(x)
            skip_connections.append(skip_out)

        if self.use_skip_connections:
            x = sum(skip_connections)
        x = self.final_relu(x)
        x = self.flatten(x)
        x = self.dropout_1(x)
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.dense_2(x)

        return x


# tcn_model = DilatedTCN(kernel_size=8, dilatations=[1, 2, 4, 8], nb_stacks=16,nb_filters=24,  num_classes=1024,input_channels=2,activation='norm_relu')
# x = torch.rand((1,2,2000))
# tcn_model(x)