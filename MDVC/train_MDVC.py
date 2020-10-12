import MDVC_model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import os
dirname = os.path.dirname(__file__)

train_data = os.path.dirname(dirname) + '/data/train.pickle'
dev_data = os.path.dirname(dirname) + '/data/dev.pickle'
feature_list = ['txt_fea', 'img_fea']

dataset_train = MDVC_model.load_data(train_data, feature_list)
dataset_validation = MDVC_model.load_data(dev_data, feature_list)

loader_train = data_utils.DataLoader(
    dataset_train,
    batch_size=128, shuffle=True)

loader_validation = data_utils.DataLoader(
    dataset_validation,
    batch_size=500)

d_encoder = 1024
d_decoder = 768

d_model = 50
H = 2
d_ff = 100
N = 1
dout_p = 0.0
n_label = 2
N_Epoch = 300


model = MDVC_model.Model(d_encoder=d_encoder, d_decoder=d_decoder,
                            d_model=d_model, dout_p=dout_p, H=H,
                            d_ff=d_ff, n_label=n_label, N=N)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

writer = SummaryWriter(log_dir='C:/Users/milad/PycharmProjects/Hateful_meme/MDVC/runs/1')
for epoch in range(N_Epoch):
    model.train()
    model.train_manual(criterion, optimizer, loader_train)
    model.eval()
    model.evalute(criterion, loader_validation)

    writer.add_scalar("Loss/train", model.loss[-1], epoch)
    writer.add_scalar("AUC/train", model.auc[-1], epoch)
    writer.add_scalar("Loss/val", model.loss_val[-1], epoch)
    writer.add_scalar("AUC/val", model.auc_val[-1], epoch)

    if model.early_stop(patience=20):
        break

    print('[%d] loss: %.3f aus: %.3f  val loss: %.3f  val auc: %.3f' %
          (epoch + 1, model.loss[-1], model.auc[-1], model.loss_val[-1], model.auc_val[-1]))

writer.close()
print(max(model.auc_val))