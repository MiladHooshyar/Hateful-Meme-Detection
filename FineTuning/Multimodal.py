import tensorflow as tf
from tensorflow.keras import layers
from FineTuning import Xception as XC, Bert as BT


def make_mm_network(MAX_SEQ_LEN, NUM_CLASS=2):
    bert_st, input_word_ids, input_mask, segment_ids, tokenizer = BT.make_bert_network(MAX_SEQ_LEN)
    Xce_st, input_img = XC.make_Xception_network()

    concatenated = layers.concatenate([bert_st, Xce_st])

    classifier = tf.keras.layers.Dropout(0.2)(concatenated)
    classifier = tf.keras.layers.Dense(NUM_CLASS, activation="sigmoid", name="dense_output")(classifier)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_mask, segment_ids, input_img], outputs=classifier)

    return model, tokenizer
