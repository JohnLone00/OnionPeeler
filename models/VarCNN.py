import torch
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 1)):
        super(BasicBlock1D, self).__init__()

        # 计算为了 'same' 填充需要的填充量
        padding1 = ((kernel_size - 1) // 2) * dilations[0]  # 对于第一层卷积
        padding2 = ((kernel_size - 1) // 2) * dilations[1]  # 对于第二层卷积

        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding1,
                               dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 第二层卷积
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding2,
                               dilation=dilations[1], bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, block, layers, dilations):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Adjusting dilation rates as per your TensorFlow/Keras settings
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,dilations=dilations)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilations=dilations)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilations=dilations)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilations=dilations)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, out_channels, blocks, stride=1, dilations=(1, 1)):
        layers = []
        # 添加第一个残差块，膨胀率为 (1, 2)
        if dilations==(1,1):
            layers.append(block(self.in_channels, out_channels, stride=stride, dilations=dilations))
            self.in_channels = out_channels

            # 添加第二个残差块，膨胀率为 (4, 8)
            if blocks > 1:
                layers.append(block(out_channels, out_channels, stride=stride, dilations=dilations))

            return nn.Sequential(*layers)
        else:
            layers.append(block(self.in_channels, out_channels, stride=stride, dilations=(1, 2)))
            self.in_channels = out_channels

            # 添加第二个残差块，膨胀率为 (4, 8)
            if blocks > 1:
                layers.append(block(out_channels, out_channels, stride=stride, dilations=(4, 8)))

            return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class MetadataMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MetadataMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)

class VarResNet(nn.Module):
    def __init__(self, use_dir=True, use_time=False, use_metadata=False,
                 dir_dilations=True, time_dilations=False, num_classes=539,
                 metadata_input_size=7):
        super(VarResNet, self).__init__()
        self.use_dir = use_dir
        self.use_time = use_time
        self.use_metadata = use_metadata


        # Define the ResNet components for directory and time inputs
        if use_dir:
            dilations = (1, 2) if dir_dilations else (1, 1)
            self.dir_resnet = ResNet1D(BasicBlock1D, [2, 2, 2, 2], dilations=dilations)

        if use_time:
            dilations = (1, 2) if time_dilations else (1, 1)
            self.time_resnet = ResNet1D(BasicBlock1D, [2, 2, 2, 2], dilations=dilations)

        # Define the MLP for metadata
        if use_metadata:
            self.metadata_mlp = MetadataMLP(metadata_input_size, 32)

        # Define the final fully connected layers
        self.fc_concat = nn.Linear((32 * use_metadata) + (512 * use_dir) + (512 * use_time), 1024)
        self.bn_concat = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc_final = nn.Linear(1024, num_classes)

    def forward(self, dir_input=None, time_input=None, metadata_input=None):
        concat_tensors = []

        # Process the directory input if it is used
        if self.use_dir and dir_input is not None:
            dir_output = self.dir_resnet(dir_input)
            concat_tensors.append(dir_output)

        # Process the time input if it is used
        if self.use_time and time_input is not None:
            time_output = self.time_resnet(time_input)
            concat_tensors.append(time_output)

        # Process the metadata input if it is used
        if self.use_metadata and metadata_input is not None:
            metadata_output = self.metadata_mlp(metadata_input)
            concat_tensors.append(metadata_output)

        # Concatenate the outputs from the different streams
        combined = torch.cat(concat_tensors, dim=1)
        combined = F.relu(self.bn_concat(self.fc_concat(combined)))
        combined = self.dropout(combined)
        out = self.fc_final(combined)

        return out

#
# config = {
#     'num_classes': 1000,  # adjust as per your requirements
#     'metadata_input_size': 7,
#     'use_dir': True,
#     'use_time': False,
#     'use_metadata': False,
#     'dir_dilations': False,
#     'time_dilations': False
# }
#
# model = VarResNet(**config)
#
# # Optimizer and loss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.CrossEntropyLoss()
#
# # Learning rate scheduler
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=np.sqrt(0.1), patience=5, verbose=True)
#
# x = torch.randn(8,1,5000)
#
# y = model(dir_input=x)
# print(y.shape)