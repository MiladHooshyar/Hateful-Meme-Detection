import Concat_model
import os
import pickle
import sys
dirname = sys.path[0]


checkpoint_filepath = dirname + '/model/'
train_path = os.path.dirname(dirname) + '/data/train.pickle'
dev_path = os.path.dirname(dirname) + '/data/dev.pickle'

with open(train_path, 'rb') as handle:
    train_data = pickle.load(handle)
with open(dev_path, 'rb') as handle:
    validation_data = pickle.load(handle)

feature_name = ['img_fea', 'txt_fea', 'txt_mod', 'img_mod', 'txt_img_fea']
feature_len = [train_data[fe].shape[1] for fe in feature_name]

dout = 0.3
cm = Concat_model.concat_model(feature_len= feature_len, feature_name = feature_name, dout = dout)
cm.make_model()
cm.train_model(train_data, validation_data, checkpoint_filepath, Epochs=500,
                    batch_size=256, weighted_sample=True)

