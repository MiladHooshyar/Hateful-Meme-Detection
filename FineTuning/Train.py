from FineTuning import Multimodal as MM, Bert as BT
import numpy as np

model, tokenizer = MM.make_mm_network(MAX_SEQ_LEN=128, NUM_CLASS=2)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_sentences = ['David is a donkey']
txt_inputs = BT.create_input_array(train_sentences, MAX_SEQ_LEN=124, tokenizer=tokenizer)
img_input = np.random.randn(1, 150, 150, 3)

train_y = np.array([0])

train_x = txt_inputs + [img_input]

model.fit(train_x, train_y, epochs=1)
