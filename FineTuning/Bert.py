import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import bert


def get_masks(tokens, max_seq_length):
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def create_single_input(sentence, tokenizer, MAX_SEQ_LEN):
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:MAX_SEQ_LEN]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)
    return ids, masks, segments


def create_input_array(sentences, tokenizer, MAX_SEQ_LEN):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences, position=0, leave=True):
        ids, masks, segments = create_single_input(sentence, tokenizer, MAX_SEQ_LEN - 2)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens, )
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def make_bert_network(MAX_SEQ_LEN):
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)

    FullTokenizer = bert.bert_tokenization.FullTokenizer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                        name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    bert_st = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)

    return bert_st, input_word_ids, input_mask, segment_ids, tokenizer
