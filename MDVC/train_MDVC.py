import MDVC_model_V1
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from collections import  defaultdict

train = np.load('D:/meme_data/features/train.npz')
feature_list = ['txt_fea', 'txt_mod', 'img_fea',
                'img_mod']
train_tensor = defaultdict()
for fe in feature_list:
    temp = train[fe]
    temp = temp.reshape(temp.shape[0], 1, temp.shape[1])
    train_tensor[fe] = torch.tensor(temp).float()
train_tensor['label'] = torch.tensor(train['label']).long()

dataset_train = data_utils.TensorDataset(train_tensor['img_fea'], train_tensor['txt_fea'],
                                         train_tensor['label'])
loader_train = data_utils.DataLoader(
    dataset_train,
    batch_size=1, shuffle=True)

dev = np.load('D:/meme_data/features/dev.npz')
validation_tensor = defaultdict()
for fe in feature_list:
    temp = dev[fe]
    temp = temp.reshape(temp.shape[0], 1, temp.shape[1])
    validation_tensor[fe] = torch.tensor(temp).float()
validation_tensor['label'] = torch.tensor(dev['label']).long()

dataset_validation = data_utils.TensorDataset(validation_tensor['img_fea'], validation_tensor['txt_fea'],
                                              validation_tensor['label'])
loader_validation = data_utils.DataLoader(
    dataset_validation,
    batch_size=128)

d_encoder = validation_tensor['img_fea'].size(-1)
d_decoder = validation_tensor['txt_fea'].size(-1)

d_model = 200
H = 2

d_ff = 100
N = 1
dout_p = 0.1
n_label = 2

N_Epoch = 300

model = MDVC_model_V1.Model(d_encoder=d_encoder, d_decoder=d_decoder,
           d_model=d_model, dout_p=dout_p, H=H,
           d_ff=d_ff, n_label=n_label, N=N)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(N_Epoch):
    model.train()
    loss = model.train_manual(criterion, optimizer, loader_train)
    model.eval()
    loss_val, auc_val = model.evalute(criterion, loader_validation)

    print('[%d] loss: %.3f  val loss: %.3f  val auc: %.3f' %
          (epoch + 1, loss, loss_val, auc_val))



# d_model = 300
# H = 3
# d_ff = 100
#
# l = MDVC_model_V1.DecoderLayer(d_decoder, d_model, dout_p, H , d_ff)
#
# x = l(torch.zeros(1, 1, d_decoder), torch.zeros(1, 1, d_model), None, None)
# print(x.size())