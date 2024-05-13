from sklearn import preprocessing
from torch.utils.data import Dataset,random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import torch
from collections import Counter


class Load_Dataset(Dataset):
    def __init__(self, length):
        super(Load_Dataset, self).__init__()

        X_time = np.load('./data/time_x_Onion.npy', allow_pickle=True).astype('float32')
        y_time = np.load('./data/time_y_Onion.npy')
        X_dir = np.load('./data/dir_x_Onion.npy', allow_pickle=True)
        y_dir = np.load('./data/dir_y_Onion.npy')
        X_length = np.load('./data/length_x_Onion.npy', allow_pickle=True)
        y_length = np.load('./data/length_y_Onion.npy')
        #
        # X_time = np.load('./data/time_x_WL.npy',allow_pickle=True).astype('float32')
        # y_time = np.load('./data/time_y_WL.npy')
        # X_dir = np.load('./data/dir_x_WL.npy',allow_pickle=True)
        # y_dir = np.load('./data/dir_y_WL.npy')
        # X_length = np.load('./data/length_x_WL.npy',allow_pickle=True)
        # y_length = np.load('./data/length_y_WL.npy')

        X_length = np.abs(X_length)
        scaler = StandardScaler()
        X_length = scaler.fit_transform(X_length)

        # X_length = np.fft.fft(X_length, axis=-1).real
        # np.save("./data/length_x_WL.npy",X_length)

        # X_TAM = np.load('./data/TAM_x_WL.npy',allow_pickle=True)
        # y_TAM = np.load('./data/TAM_y_WL.npy', allow_pickle=True)


        # X_length = np.fft.fft(X_length,axis=-1)
        # X = X_dir

        # X = X_length  * X_time
        X = np.stack((X_dir, X_time), axis=1)

        print(X.shape)
        y = y_length



        X = X.reshape((-1,2, 5000))[:,:,:length]

        labels = y
        self.le = preprocessing.LabelEncoder()
        self.le.fit(labels)
        labels = self.le.transform(labels)
        y = labels

        print(np.unique(y).shape)
        # shuffle
        data = list(zip(X, y))
        np.random.shuffle(data)
        X, y = zip(*data)
        X = np.array(list(X))
        y = np.array(list(y))

        if isinstance(X, np.ndarray):
            self.x_data = torch.from_numpy(X)
            self.y_data = torch.from_numpy(y).long()
        else:
            self.x_data = X
            self.y_data = y

        self.len = self.x_data.shape[0]


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Load_Dataset_muti_view(Dataset):
    def __init__(self):
        super(Load_Dataset_muti_view, self).__init__()
        X_time = np.load('./data/time_x_onion.npy',allow_pickle=True).astype('float32')
        y_time = np.load('./data/time_y_onion.npy')

        X_dir = np.load('./data/dir_x_onion.npy',allow_pickle=True)
        y_dir = np.load('./data/dir_y_onion.npy')

        X_length = np.load('./data/length_x_onion.npy',allow_pickle=True)
        y_length = np.load('./data/length_y_onion.npy')
        X_length = np.abs(X_length)

        scaler = StandardScaler()
        X_length = scaler.fit_transform(X_length)



        labels = y_length
        self.le = preprocessing.LabelEncoder()
        self.le.fit(labels)
        labels = self.le.transform(labels)
        y = labels

        X_time = X_time.reshape((-1, 1, 5000))
        X_length = X_length.reshape((-1, 1, 5000))
        X_dir = X_dir.reshape((-1, 1, 5000))
        X_time = np.array(list(X_time))
        X_length = np.array(list(X_length))
        X_dir = np.array(list(X_dir))
        y = np.array(list(y))

        self.X_time = torch.from_numpy(X_time)
        self.X_length = torch.from_numpy(X_length)
        self.X_dir = torch.from_numpy(X_dir)
        self.y = torch.from_numpy(y)


        self.len = self.y.shape[0]

    def __getitem__(self, index):
        return self.X_length[index],self.X_time[index], self.y [index]

    def __len__(self):
        return self.len