from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


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
        B, seq_len, d_model = Q.shape

        Q = self.linears[0](Q)
        K = self.linears[1](K)
        V = self.linears[2](V)

        Q = Q.view(B, -1, self.H, self.d_k).transpose(-3, -2)  # (-4, -3*, -2*, -1)
        K = K.view(B, -1, self.H, self.d_k).transpose(-3, -2)
        V = V.view(B, -1, self.H, self.d_k).transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        att = attention(Q, K, V, mask)  # (B, H, seq_len, d_k)
        att = att.transpose(-3, -2).contiguous().view(B, seq_len, d_model)
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

        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_encoder, d_model, dout_p, H, d_ff):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(d_encoder, d_model)
        self.res_layers = clone(ResidualConnection(d_model, dout_p), 2)
        self.self_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, src_mask):
        sublayer0 = lambda x: self.self_att(x, x, x, src_mask)
        sublayer1 = self.feed_forward

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
        self.self_att = MultiheadedAttention(d_model, H)
        self.enc_att = MultiheadedAttention(d_model, H)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, x, memory, src_mask,
                trg_mask):
        sublayer0 = lambda x: self.self_att(x, x, x, trg_mask)
        sublayer1 = lambda x: self.enc_att(x, memory, memory, src_mask)
        sublayer2 = self.feed_forward

        x = self.linear(x)
        x = F.relu(x)
        x = self.res_layers[0](x, sublayer0)
        x = self.res_layers[1](x, sublayer1)
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
        self.linear = nn.Linear(d_model, n_label)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(n_label, n_label)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(self.dropout(torch.sigmoid(x)))
        return x


class Model(nn.Module):
    def __init__(self, d_encoder, d_decoder, d_model, dout_p, H, d_ff, n_label, N=1):
        super(Model, self).__init__()
        self.encoder = Encoder(d_encoder, d_model, dout_p, H, d_ff, N)
        self.decoder = Decoder(d_decoder, d_model, dout_p, H, d_ff, N)
        self.classifier = Classifier(d_model, n_label, dout_p)

    def forward(self, x_encode, x_decode, src_mask, trg_mask):
        x_encode = self.encoder(x_encode, src_mask=src_mask)
        x = self.decoder(x_decode, memory=x_encode, src_mask=src_mask, trg_mask=trg_mask)
        x = self.classifier(x)
        return x

    def train_manual(self, criterion, optimizer, loader):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            x_encoder, x_decoder, label = data
            optimizer.zero_grad()
            outputs = self.forward(x_encode=x_encoder, x_decode=x_decoder,
                                   src_mask=None, trg_mask=None)
            loss = criterion(outputs[:, 0, :], label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss/(i + 1)

    def evalute(self, criterion, loader):
        running_loss, auc = 0, 0
        for i, data in enumerate(loader, 0):
            x_encoder, x_decoder, label = data
            outputs = self.forward(x_encode=x_encoder, x_decode=x_decoder,
                                   src_mask=None, trg_mask=None)
            loss = criterion(outputs[:, 0, :], label)
            running_loss += loss.item()
            auc += roc_auc_score(label.detach().numpy(), outputs[:, 0, 1].detach().numpy())

        return  running_loss/(i + 1), auc/(i + 1)
