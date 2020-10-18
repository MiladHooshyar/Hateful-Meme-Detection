from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import torch.utils.data as data_utils
import pickle
from collections import defaultdict


class ClClassifier(nn.Module):
    def __init__(self, d_cl, d_vb, d_model):
        super(ClClassifier, self).__init__()
        self.d_model = d_model
        self.d_vb = d_vb
        self.d_cl = d_cl

        self.norm1_cl = nn.LayerNorm(self.d_cl)
        self.norm2_cl = nn.LayerNorm(2 * self.d_model)
        self.linear1_cl = nn.Linear(self.d_cl, 2 * self.d_model)
        self.linear2_cl = nn.Linear(2 * self.d_model, self.d_model)

        self.norm_vb = nn.LayerNorm(self.d_vb)
        self.linear1_vb = nn.Linear(self.d_vb, d_model)

        self.linear1_class = nn.Linear(2 * self.d_model, self.d_model)
        self.linear2_class = nn.Linear(self.d_model, 2)

        self.loss = []
        self.auc = []
        self.loss_val = []
        self.auc_val = []

    def forward(self, x1, x2, x3, x4, x5, x_vb, p):
        x_cl = torch.cat((x1, x2, x3, x4, x5), 1)
        x_cl = self.norm1_cl(x_cl)
        x_cl = self.linear1_cl(x_cl)
        x_cl = F.relu(x_cl)

        x_cl = self.norm2_cl(x_cl)
        x_cl = self.linear2_cl(x_cl)
        x_cl = F.relu(x_cl)

        x_vb = self.norm_vb(x_vb)
        x_vb = self.linear1_vb(x_vb)
        x_vb = F.relu(x_vb)

        if p < 0.1:
            x = torch.cat((x_cl, x_vb), 1)
        else:
            x = torch.cat((x_cl, x_vb * 0.), 1)

        x = self.linear1_class(x)
        x = F.relu(x)
        x = self.linear2_class(x)
        x = torch.sigmoid(x)
        return x

    def train_manual(self, criterion, optimizer, loader):
        running_loss, auc = 0.0, 0.0
        pred_lb, lb = [], []
        for i, data in enumerate(loader, 0):
            x1, x2, x3, x4, x5, x_vb, ID, label = data

            optimizer.zero_grad()
            outputs = self.forward(x1, x2, x3, x4, x5, x_vb, np.random.rand())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred_lb += list(outputs[:, 1].detach().numpy())
            lb += list(label.detach().numpy()[:, 1])
        auc = roc_auc_score(lb, pred_lb)
        self.loss.append(running_loss / (i + 1))
        self.auc.append(auc)

    def evalute(self, criterion, loader, option):
        running_loss, auc = 0, 0
        pred_lb, lb, pred_output = [], [], defaultdict()
        pred_output['id'] = []
        pred_output['proba'] = []
        pred_output['label'] = []
        for i, data in enumerate(loader, 0):
            x1, x2, x3, x4, x5, x_vb, ID, label = data
            outputs = self.forward(x1, x2, x3, x4, x5, x_vb, 0.)

            loss = criterion(outputs, label)
            running_loss += loss.item()
            pred_lb += list(outputs[:, 1].detach().numpy())
            lb += list(label.detach().numpy()[:, 1])

            if option == "TEST":
                id_list = list(ID.detach().numpy())
                pred_list = list(outputs[:, 1].detach().numpy())
                class_list = [0 if p < 0.5 else 1 for p in pred_list]
                for i, p, c in zip(id_list, pred_list, class_list):
                    pred_output['id'].append(i)
                    pred_output['proba'].append(p)
                    pred_output['label'].append(c)
        if option == 'VAL':
            auc = roc_auc_score(lb, pred_lb)
            self.loss_val.append(running_loss / (i + 1))
            self.auc_val.append(auc)

        return pred_output

    def early_stop(self, patience=10):
        if max(self.auc_val[-patience:]) >= max(self.auc_val[:patience]):
            return False
        else:
            return True

def load_data(data_file, feature_list):
    with open(data_file, 'rb') as handle:
        data = pickle.load(handle)
    data_tensor = defaultdict()
    for fe in feature_list:
        temp = np.asarray(data[fe])
        temp = temp.reshape(temp.shape[0], temp.shape[1])
        data_tensor[fe] = torch.tensor(temp).float()

    if 'label' in data.keys():
        label = np.zeros((data['label'].shape[0], 2))
        label[:, 0] = 1 - data['label']
        label[:, 1] = data['label']
        data_tensor['label'] = torch.tensor(label).float()
    else:
        data_tensor['label'] = torch.zeros(data_tensor[fe].size()[0])

    data_tensor['ID'] = torch.tensor(data['id']).int()

    print(temp.shape)

    return data_utils.TensorDataset(data_tensor['img_fea'],
                                    data_tensor['txt_fea'],
                                    data_tensor['txt_mod'],
                                    data_tensor['img_mod'],
                                    data_tensor['txt_img_fea'],
                                    data_tensor['feature_vb'],
                                    data_tensor['ID'],
                                    data_tensor['label'])

