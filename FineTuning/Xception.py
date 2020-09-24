import tensorflow as tf
from tensorflow import keras


def make_Xception_network():
    base_model = keras.applications.Xception(
        weights='imagenet',
        input_shape=(150, 150, 3),
        include_top=False)
    base_model.trainable = False

    input_img = tf.keras.layers.Input(shape=(150, 150, 3), name="input_img")
    x = base_model(input_img, training=False)
    Xce_st = keras.layers.GlobalAveragePooling2D()(x)

    return Xce_st, input_img
