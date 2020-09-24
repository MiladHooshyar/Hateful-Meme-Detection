import numpy as np
from sklearn.manifold import TSNE

train = np.load('D:/meme_data/features/train.npz')
img_feature = train['img_fea']
text_feature = train['txt_fea']
label = train['label']

dev = np.load('D:/meme_data/features/dev.npz')
img_feature_dev = dev['img_fea']
text_feature_dev = dev['txt_fea']
label_dev = dev['label']

########
X_non = img_feature[label == 0, :]
X_embedded_non = TSNE(n_components=2).fit_transform(X_non)
X_hat = img_feature[label == 1, :]
X_embedded_hat = TSNE(n_components=2).fit_transform(X_hat)
np.savez('D:/meme_data/features/TSNE_img', Hatefull=X_embedded_hat,
         Non_hateful=X_embedded_non)
#######
X_non = text_feature[label == 0, :]
X_embedded_non = TSNE(n_components=2).fit_transform(X_non)
X_hat = text_feature[label == 1, :]
X_embedded_hat = TSNE(n_components=2).fit_transform(X_hat)
np.savez('D:/meme_data/features/TSNE_txt', Hatefull=X_embedded_hat,
         Non_hateful=X_embedded_non)
#######
X_non = np.concatenate((img_feature[label == 0, :], text_feature[label == 0, :]), axis=1)
X_embedded_non = TSNE(n_components=2).fit_transform(X_non)
X_hat = np.concatenate((img_feature[label == 1, :], text_feature[label == 1, :]), axis=1)
X_embedded_hat = TSNE(n_components=2).fit_transform(X_hat)
np.savez('D:/meme_data/features/TSNE_img_txt', Hatefull=X_embedded_hat,
         Non_hateful=X_embedded_non)
