import Concat_model
import numpy as np


checkpoint_filepath = '/model/'
train_data = np.load('D:/meme_data/features/train.npz')
validation_data = np.load('D:/meme_data/features/dev.npz')
feature_name = ['txt_fea', 'txt_mod', 'txt_img_fea', 'img_fea', 'img_mod']
feature_len = [train_data[fe].shape[1] for fe in feature_name]

cm = Concat_model.concat_model(feature_len= feature_len, feature_name = feature_name)
cm.make_model()
cm.train_model(train_data, validation_data, checkpoint_filepath, Epochs=500,
                    batch_size=256, weighted_sample=True)

