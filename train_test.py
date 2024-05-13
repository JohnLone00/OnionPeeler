import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import classification_report
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import Load_Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.TCN import DilatedTCN
from models.VarCNN import VarResNet
from models.OnionPeeler import InceptionTime
from models.AWF import AWF_CNN

device = torch.device("cuda")
torch.cuda.empty_cache()
dataset = Load_Dataset(length=2000)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)



# model = DF().to(device)
# model = VarResNet().to(device)
# model = getRF_CAM(399).to(device)
# model = InceptionTime(1,539,seq_len=2000,nf=32,depth=12).to(device)
# model = AWF_CNN(5000,399).to(device)
model = DilatedTCN(kernel_size=8, dilatations=[1, 2, 4, 8], nb_stacks=16,nb_filters=24,  num_classes=539,input_channels=2,activation='norm_relu').to(device)



num_epochs = 400

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):

    epoch_loss_train = 0.0  # 记录每个epoch的总损失
    epoch_acc_train = 0.0  # 记录每个epoch的总准确率
    epoch_acc_test = 0.0  # 记录每个epoch的总准确率

    model.train()
    for X, y in train_loader:
        X = X.float().to(device)
        y = y.long().to(device)

        y_hat = model(X)
        loss = criterion(y_hat,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = y.eq(y_hat.detach().argmax(dim=1)).float().mean()

        epoch_loss_train += loss.item()
        epoch_acc_train += acc.item()

    model.eval()
    val_loss = 0
    val_corrects = 0
    total = 0
    for X, y in val_loader:
        X = X.float().to(device)
        y = y.long().to(device)

        y_hat = model(X)
        loss = criterion(y_hat, y)
        acc = y.eq(y_hat.detach().argmax(dim=1)).float().mean()
        epoch_acc_test += acc.item()
        val_loss += loss.item()
        val_corrects += acc.item() * X.size(0)
        total += X.size(0)
    # scheduler.step(val_loss / len(val_loader))
    print(f"Epoch {epoch + 1}: Loss_train = {epoch_loss_train / len(train_loader):.4f}, Loss_val = {val_loss / len(val_loader):.4f}, Acc_train = {epoch_acc_train / len(train_loader):.4f}, Acc_val = {epoch_acc_test / len(val_loader):.4f}")

model.eval()
y_hat = []
y_true = []
outputs_list = []

for X, y in val_loader:
    output = model(X.float().to(device))  # Feed Network
    outputs_list.append(output.data.cpu().numpy())
    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_hat.extend(output)  # Save Prediction

    labels = y.data.cpu().numpy()
    y_true.extend(labels)

all_outputs = np.concatenate(outputs_list, axis=0)
result = classification_report(y_true, y_hat,target_names=dataset.le.classes_,digits=2)
result = classification_report(y_true, y_hat, target_names=dataset.le.classes_, output_dict=True)