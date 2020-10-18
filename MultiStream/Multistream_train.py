import os
import MultiStream_model as model
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


dirname = os.path.dirname(__file__)

train_data = os.path.dirname(dirname) + '/data/train.pickle'
dev_data = os.path.dirname(dirname) + '/data/dev.pickle'
test_data = os.path.dirname(dirname) + '/data/test.pickle'

feature_list = ['img_fea', 'txt_fea', 'txt_mod', 'img_mod', 'txt_img_fea', 'feature_vb']

dataset_train = model.load_data(train_data, feature_list)
dataset_validation = model.load_data(dev_data, feature_list)
dataset_test = model.load_data(test_data, feature_list)

loader_train = data_utils.DataLoader(
    dataset_train,
    batch_size=256, shuffle=True)

loader_validation = data_utils.DataLoader(
    dataset_validation,
    batch_size=500)

loader_test = data_utils.DataLoader(
    dataset_test,
    batch_size=2000)

d_cl = 3333
d_vb = 768
d_model = 20

N_Epoch = 100

model_cl = model.ClClassifier(d_cl=d_cl, d_vb=d_vb, d_model=d_model)

criterion = nn.BCELoss()
optimizer = optim.Adam(model_cl.parameters())

writer = SummaryWriter(log_dir=dirname + '/runs/Base')
for epoch in range(N_Epoch):
    model_cl.train()
    model_cl.train_manual(criterion, optimizer, loader_train)
    model_cl.eval()
    _ = model_cl.evalute(criterion, loader_validation, 'VAL')

    writer.add_scalar("Loss/train", model_cl.loss[-1], epoch)
    writer.add_scalar("AUC/train", model_cl.auc[-1], epoch)
    writer.add_scalar("Loss/val", model_cl.loss_val[-1], epoch)
    writer.add_scalar("AUC/val", model_cl.auc_val[-1], epoch)

    if epoch > 1 and model_cl.auc_val[-1] >= max(model_cl.auc_val[:-1]):
        print(model_cl.auc_val[-1])
        pred_test = model_cl.evalute(criterion, loader_test, 'TEST')
        pred_test = pd.DataFrame.from_dict(pred_test)
        pred_test.to_csv(dirname + '/test_result/MH_submission.csv',
                         index=False)


    if model_cl.early_stop(patience=20):
        break
    print('[%d] loss: %.3f aus: %.3f  val loss: %.3f  val auc: %.3f' %
          (epoch + 1, model_cl.loss[-1], model_cl.auc[-1], model_cl.loss_val[-1], model_cl.auc_val[-1]))
writer.close()
print(max(model_cl.auc_val))
