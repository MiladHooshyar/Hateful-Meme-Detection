from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import torch.utils.data as data_utils
import pickle


def clone(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def attention(Q, K, V, mask):
    d_k = Q.size(-1)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)

    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)

    return out


class MultiheadedAttention(nn.Module):
    def __init__(self, d_model, H):
        super(MultiheadedAttention, self).__init__()
        assert d_model % H == 0
        self.d_model = d_model
        self.H = H
        self.d_k = d_model // H
        self.linears = clone(nn.Linear(d_model, d_model), 4)

    def forward(self, Q, K, V, mask):
        B, d_model = Q.shape

        Q = self.linears[0](Q)
        K = self.linears[1](K)
        V = self.linears[2](V)

        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        att = attention(Q, K, V, mask)
        att = att.transpose(-3, -2).contiguous().view(B, d_model)
        att = self.linears[3](att)

        return att


class ResidualConnection(nn.Module):

    def __init__(self, size, dout_p):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dout_p)

    def forward(self, x, sublayer):
        res = self.norm(x)
        res = sublayer(res)
        res = self.dropout(res)

        return x + res


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)


        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_encoder, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(d_encoder, d_model)
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.feed_forward_0 = PositionwiseFeedForward(d_model, d_model)
        self.feed_forward_1 = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, src_mask):
        sublayer0 = self.feed_forward_0
        sublayer1 = self.feed_forward_1

        x = self.linear(x)
        x = F.relu(x)
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)

        return x


class Encoder(nn.Module):

    def __init__(self, d_encoder, d_model, dout_p, H, d_ff, N):
        super(Encoder, self).__init__()
        self.enc_layers = clone(EncoderLayer(d_encoder, d_model, dout_p, H, d_ff), N)

    def forward(self, x, src_mask):
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_decoder, d_model, dout_p, H, d_ff):
        super(DecoderLayer, self).__init__()
        self.linear = nn.Linear(d_decoder, d_model)
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 3)
        self.enc_att = MultiheadedAttention(d_model, H)
        self.feed_forward_0 = PositionwiseFeedForward(d_model, d_model)
        self.feed_forward_1 = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, memory, src_mask,
                trg_mask):
        sublayer0 = self.feed_forward_0
        sublayer1 = lambda x: self.enc_att(x, memory, memory, src_mask)
        sublayer2 = self.feed_forward_1

        x = self.linear(x)
        x = F.relu(x)
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
        x = F.relu(x)
        x = self.res_layers[2](x, sublayer2)

        return x


class Decoder(nn.Module):

    def __init__(self, d_decoder, d_model, dout_p, H, d_ff, N):
        super(Decoder, self).__init__()
        self.dec_layers = clone(DecoderLayer(d_decoder, d_model, dout_p, H, d_ff), N)

    def forward(self, x, memory, src_mask, trg_mask):
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, trg_mask)
        return x


class Classifier(nn.Module):

    def __init__(self, d_model, n_label, dout_p):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(d_model, n_label)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.linear(self.dropout(x))
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


class Model(nn.Module):
    def __init__(self, d_encoder, d_decoder, d_model, dout_p, H, d_ff, n_label, N=1):
        super(Model, self).__init__()
        self.encoder = Encoder(d_encoder, d_model, dout_p, H, d_ff, N)
        self.decoder = Decoder(d_decoder, d_model, dout_p, H, d_ff, N)
        self.classifier = Classifier(d_model, n_label, dout_p)
        self.loss = []
        self.auc = []
        self.loss_val = []
        self.auc_val = []

    def forward(self, x_encode, x_decode, src_mask, trg_mask):
        x_encode = self.encoder(x_encode, src_mask=src_mask)
        x = self.decoder(x_decode, memory=x_encode, src_mask=src_mask, trg_mask=trg_mask)
        x = self.classifier(x, x_encode)
        return x

    def train_manual(self, criterion, optimizer, loader):
        running_loss, auc = 0.0, 0.0
        pred_lb, lb = [], []
        for i, data in enumerate(loader, 0):
            x_encoder, x_decoder, label = data
            optimizer.zero_grad()
            outputs = self.forward(x_encode=x_encoder, x_decode=x_decoder,
                                   src_mask=None, trg_mask=None)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred_lb += list(outputs[:, 1].detach().numpy())
            lb += list(label.detach().numpy()[:, 1])
        auc = roc_auc_score(lb, pred_lb)
        self.loss.append(running_loss / (i + 1))
        self.auc.append(auc)

    def evalute(self, criterion, loader):
        running_loss, auc = 0, 0
        pred_lb, lb = [], []
        for i, data in enumerate(loader, 0):
            x_encoder, x_decoder, label = data
            outputs = self.forward(x_encode=x_encoder, x_decode=x_decoder,
                                   src_mask=None, trg_mask=None)

            loss = criterion(outputs, label)
            running_loss += loss.item()
            pred_lb += list(outputs[:, 1].detach().numpy())
            lb += list(label.detach().numpy()[:, 1])

        auc = roc_auc_score(lb, pred_lb)
        self.loss_val.append(running_loss / (i + 1))
        self.auc_val.append(auc)

    def early_stop(self, patience=10):
        if max(self.auc_val[-patience:]) >= max(self.auc_val[:patience]):
            return False
        else:
            return True

    def save_model(self, path):
        if len(self.auc_val) >= 2:
            if self.auc_val[-1] >= max(self.auc_val[:-1]):
                torch.save(self.state_dict(), path)


def load_data(data_file, feature_list):
    with open(data_file, 'rb') as handle:
        data = pickle.load(handle)
    data_tensor = defaultdict()
    for fe in feature_list:
        temp = data[fe]
        temp = temp.reshape(temp.shape[0], temp.shape[1])
        data_tensor[fe] = torch.tensor(temp).float()
    label = np.zeros((data['label'].shape[0], 2))
    label[:, 0] = 1 - data['label']
    label[:, 1] = data['label']
    data_tensor['label'] = torch.tensor(label).float()
    return data_utils.TensorDataset(data_tensor['img_fea'], data_tensor['txt_fea'],
                                    data_tensor['label'])
