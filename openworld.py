import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split, ConcatDataset
from sklearn.metrics import classification_report
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import Load_Dataset,Load_Dataset_ow
import configs
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.TCN import DilatedTCN
from models.VarCNN import VarResNet
# from tsai.models import InceptionTime
from models.OnionPeeler import InceptionTime
from models.AWF import AWF_CNN
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from models.OnionPeeler import InceptionTime
from models.DF import DF
from models.VarCNN import VarResNet
from models.RF import getRF_CAM
import csv
import torch.nn.functional as F
from models.AWF import AWF_CNN

def score_func_precision_recall(result_file, website_res, unmon_label):
    eps = 1e-6
    file = open(result_file, 'w+', encoding='utf-8', newline='')
    csvwirter = csv.writer(file)
    upper_bound = 1.0
    thresholds = upper_bound - upper_bound / np.logspace(0.05, 2, num=15, endpoint=True)
    csvwirter.writerow(['TH  ', 'TP   ', 'TN   ', 'FP   ', 'FN   ', 'Pre. ', 'Rec. '])
    fmt_str = '{:.2f}:\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}'

    # evaluate list performance at different thresholds
    # high threshold will yield higher precision, but reduced recall
    for TH in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0

        # Test with Monitored testing instances
        for i in range(len(website_res)):
            ground_truths = website_res[i][0]
            sm_vector = np.array(website_res[i][1:])
            predicted_class = np.argmax(sm_vector)
            max_prob = max(sm_vector)
            if ground_truths != unmon_label:
                if predicted_class == ground_truths:  # predicted as Monitored
                    if max_prob >= TH:  # predicted as Monitored and actual site is Monitored
                        TP = TP + 1
                    else:  # predicted as Unmonitored and actual site is Monitored
                        FN = FN + 1
                else:  # predicted as Unmonitored and actual site is Monitored
                    FN = FN + 1
            else:
                if predicted_class != unmon_label:  # predicted as Monitored
                    if max_prob >= TH:  # predicted as Monitored and actual site is Unmonitored
                        FP = FP + 1
                    else:  # predicted as Unmonitored and actual site is Unmonitored
                        TN = TN + 1
                else:  # predicted as Unmonitored and actual site is Unmonitored
                    TN = TN + 1
        res = [TH, TP, TN, FP, FN, float(TP) / (TP + FP + eps), float(TP) / (TP + FN + eps)]
        print(fmt_str.format(*res))
        csvwirter.writerow(res)

    file.close()
    return 'finish'


device = torch.device("cuda")
torch.manual_seed(44)
monitored_ds = Load_Dataset(length=5000)
train_size = int(0.8 * len(monitored_ds))
test_size = len(monitored_ds) - train_size
monitored_tr, monitored_te = random_split(monitored_ds, [train_size, test_size])

unmonitored_ds = Load_Dataset_ow(length=5000,classID=monitored_ds.getClassNum())
train_size = 1000
test_size = len(unmonitored_ds) - train_size
unmonitored_tr, unmonitered_te = random_split(unmonitored_ds, [train_size, test_size])

tr = ConcatDataset([monitored_tr, unmonitored_tr])
te = ConcatDataset([monitored_te, unmonitered_te])

print(len(tr))
train_loader = DataLoader(tr, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(te, batch_size=1, shuffle=True)



# model = InceptionTime(3,monitored_ds.getClassNum()+1,seq_len=2000,nf=32,depth=6).to(device)
# model = DF(class_nb=monitored_ds.getClassNum()+1).to(device)
model = AWF_CNN(len=5000,nb_class=monitored_ds.getClassNum()+1).to(device)
# model = VarResNet(num_classes=monitored_ds.getClassNum()+1).to(device)
# model = DilatedTCN(kernel_size=8, dilatations=[1, 2, 4, 8], nb_stacks=16,nb_filters=24,  num_classes=monitored_ds.getClassNum()+1,input_channels=2,activation='norm_relu').to(device)
# model = getRF_CAM(monitored_ds.getClassNum()+1).to(device)

# model_save_path = './model_saved/OnionPeeler_ow.pt'
# model_state_dict = torch.load(model_save_path)
# model.load_state_dict(model_state_dict)

num_epochs = 30
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss()
#
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

    print(f"Epoch {epoch + 1}: Loss_train = {epoch_loss_train / len(train_loader):.4f},Acc_train = {epoch_acc_train / len(train_loader):.4f}")
#
# torch.save(model.state_dict(),'./model_saved/AWF_ow_exp1_osv2.pt')

model.eval()

website_res = []
for x, y in test_loader:
    x = x.float().to(device)
    website_output = F.softmax(model(x), dim=--1).cpu().squeeze().detach().numpy()
    cur = [y.item()]

    cur.extend(website_output.tolist())
    website_res.append(cur)

score_func_precision_recall('openworld_AWF_OSv3_exp1.csv', website_res, monitored_ds.getClassNum())

