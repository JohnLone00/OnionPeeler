import torch.nn as nn
import torch
import torch.nn.functional as F
class DF(nn.Module):
    def __init__(self):
        super(DF, self).__init__()
        self.filter_num = [0, 32, 64, 128, 256]
        self.kernel_size = [0, 4, 4, 4, 4]
        self.conv_stride_size = [0, 1, 1, 1, 1]
        self.pool_stride_size = [0, 4, 4, 4, 4]
        self.pool_size = [0, 4, 4, 4, 4]
        self.CNN_Block_1 = nn.Sequential(
            nn.Conv1d(in_channels=3,
                      out_channels=self.filter_num[1],
                      kernel_size=self.kernel_size[1],
                      stride=self.conv_stride_size[1],
                      padding='same'),
            nn.ELU(),
            # nn.ReLU(),
            nn.Conv1d(in_channels=self.filter_num[1],
                      out_channels=self.filter_num[1],
                      kernel_size=self.kernel_size[1],
                      stride=self.conv_stride_size[1],
                      padding='same'),
            nn.ELU(),
            # nn.ReLU(),
            nn.MaxPool1d(stride=self.pool_stride_size[1],
                         kernel_size=self.pool_size[1]),
            nn.Dropout(0.1)
        )
        self.CNN_Block_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_num[1],
                      out_channels=self.filter_num[2],
                      kernel_size=self.kernel_size[2],
                      stride=self.conv_stride_size[2],
                      padding='same'),
            # nn.ReLU(),
            nn.ELU(),
            nn.Conv1d(in_channels=self.filter_num[2],
                      out_channels=self.filter_num[2],
                      kernel_size=self.kernel_size[2],
                      stride=self.conv_stride_size[2],
                      padding='same'),
            # nn.ReLU(),
            # nn.ReLU(),
            nn.ELU(),
            nn.MaxPool1d(stride=self.pool_stride_size[2],
                         kernel_size=self.pool_size[2]),
            nn.Dropout(0.1)
        )
        self.CNN_Block_3 = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_num[2],
                      out_channels=self.filter_num[3],
                      kernel_size=self.kernel_size[3],
                      stride=self.conv_stride_size[3],
                      padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filter_num[3],
                      out_channels=self.filter_num[3],
                      kernel_size=self.kernel_size[3],
                      stride=self.conv_stride_size[3],
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(stride=self.pool_stride_size[3],
                         kernel_size=self.pool_size[3]),
            nn.Dropout(0.1)
        )
        self.CNN_Block_4 = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_num[3],
                      out_channels=self.filter_num[4],
                      kernel_size=self.kernel_size[4],
                      stride=self.conv_stride_size[4],
                      padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filter_num[4],
                      out_channels=self.filter_num[4],
                      kernel_size=self.kernel_size[4],
                      stride=self.conv_stride_size[4],
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(stride=self.pool_stride_size[3],
                         kernel_size=self.pool_size[3]),
            nn.Dropout(0.1)
        )

        self.liner = nn.Sequential(
            nn.Linear(4608, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 539),
        )


    def forward(self, X):
        X = self.CNN_Block_1(X)
        X = self.CNN_Block_2(X)
        X = self.CNN_Block_3(X)
        X = self.CNN_Block_4(X)
        X = X.view(X.shape[0], -1)
        # print(X.shape)
        X = self.liner(X)
        return X
