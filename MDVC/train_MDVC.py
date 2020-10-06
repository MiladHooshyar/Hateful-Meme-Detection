import MDVC_model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

train = np.load('D:/meme_data/features/train.npz')

txt_fea_train = train['txt_fea']
txt_mod_train = train['txt_mod']
img_fea_train = train['img_fea']
img_mod_train = train['img_mod']
label_train = train['label']


dev = np.load('D:/meme_data/features/dev.npz')
txt_fea_dev = dev['txt_fea']
txt_mod_dev = dev['txt_mod']
img_fea_dev = dev['img_fea']
img_mod_dev = dev['img_mod']
label_dev = dev['label']


batch_size = 256
X_encoder, X_decoder, Label = [], [], []
for i in range(0, img_fea_train.shape[0], batch_size):
    temp_batch = min(img_mod_train.shape[0], i + batch_size) - i
    temp = img_fea_train[i:i + temp_batch, :]
    temp = torch.tensor(temp.reshape(temp_batch, 1, temp.shape[1]))
    X_encoder.append(temp.float())

    temp = txt_fea_train[i:i + temp_batch, :]
    temp = torch.tensor(temp.reshape(temp_batch, 1, temp.shape[1]))
    X_decoder.append(temp.float())

    temp = label_train[i:i+temp_batch]
    Label.append(torch.tensor(temp).long())

temp = img_fea_dev.reshape(img_fea_dev.shape[0], 1, img_fea_dev.shape[1])
X_encoder_v = torch.tensor(temp).float()

temp = txt_fea_dev.reshape(txt_fea_dev.shape[0], 1, txt_fea_dev.shape[1])
X_decoder_v = torch.tensor(temp).float()

Label_v = torch.tensor(label_dev).long()


d_encoder = X_encoder[0].size(-1)
d_decoder = X_decoder[0].size(-1)
d_model = 500
H = 5
d_ff = 100
N = 1
dout_p = 0.1
n_label = 2

N_Epoch = 200

model = MDVC_model.Model(d_encoder=d_encoder, d_decoder=d_decoder,
           d_model=d_model, dout_p=dout_p, H=H,
           d_ff=d_ff, n_label=n_label, N=N)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train_manual(criterion, optimizer, N_Epoch,
              X_encoder, X_decoder, Label,
              X_encoder_v, X_decoder_v, Label_v)

# pred = model(X_encoder_v, X_decoder_v, src_mask=None, trg_mask=None)
# print(pred.size())
# print(criterion(pred[:, 0, :], Label_v))


# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# output = loss(input, target)
# print(target.size())