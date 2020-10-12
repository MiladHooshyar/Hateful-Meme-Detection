import LogisticRegression as LR
import numpy as np
import os
import pickle
dirname = os.path.dirname(__file__)


checkpoint_filepath = dirname + '/model/'
train_path = os.path.dirname(dirname) + '/data/train.pickle'
dev_path = os.path.dirname(dirname) + '/data/dev.pickle'

with open(train_path, 'rb') as handle:
    train_data = pickle.load(handle)
with open(dev_path, 'rb') as handle:
    validation_data = pickle.load(handle)

feature_name = ['txt_fea', 'img_fea']
feature_len = [train_data[fe].shape[1] for fe in feature_name]

X = np.concatenate([train_data[fe] for fe in feature_name], axis = 1)
y = train_data['label']

X_val = np.concatenate([validation_data[fe] for fe in feature_name], axis = 1)
y_val = validation_data['label']

auc, auc_val, _ = LR.logistic_regression(X, y, X_val, y_val)

print('Train auc: ', auc, ', Validation auc: ', auc_val)