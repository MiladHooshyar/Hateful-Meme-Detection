import MDVC_model_V1
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import pickle

train = np.load('D:/meme_data/features/train.npz')
dev = np.load('D:/meme_data/features/dev.npz')

feature_list = ['txt_fea', 'txt_mod', 'img_fea',
                'img_mod']

dataset_train = MDVC_model_V1.load_data(train, feature_list)
dataset_validation = MDVC_model_V1.load_data(dev, feature_list)

loader_train = data_utils.DataLoader(
    dataset_train,
    batch_size=128, shuffle=True)

loader_validation = data_utils.DataLoader(
    dataset_validation,
    batch_size=500)

d_encoder = 1024
d_decoder = 768

H = 2
N = 1
n_label = 2
N_Epoch = 300
result = []

for dout_p in [0.0, 0.1, 0.2, 0.3]:
    for d_model in [50, 100, 150, 200, 250, 500, 750]:
        for d_ff in [50, 100, 150, 200, 250, 500]:

            model = MDVC_model_V1.Model(d_encoder=d_encoder, d_decoder=d_decoder,
                                        d_model=d_model, dout_p=dout_p, H=H,
                                        d_ff=d_ff, n_label=n_label, N=N)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters())

            name = 'dm' + str(d_model) + '_df' + str(d_ff) + '_do' + str(int(dout_p * 10))
            writer = SummaryWriter(log_dir='C:/Users/milad/PycharmProjects/Hateful_meme/MDVC/runs/' + name)
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
            writer.close()
            print(d_model, d_ff, dout_p, max(model.auc_val))
            result.append([d_model, d_ff, dout_p, max(model.auc_val)])
with open('C:/Users/milad/PycharmProjects/Hateful_meme/MDVC/result/exp_1.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
