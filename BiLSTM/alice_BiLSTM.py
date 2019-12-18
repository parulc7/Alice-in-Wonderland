# Importing the required modules
from __future__ import print_function
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense,Bidirectional
from keras.utils import vis_utils
import numpy as np

# Import the Dataset and preprocess
alice = open('alice.txt', 'rb')
lines = []
for line in alice:
    line = line.strip().lower()
    line = line.decode('ascii', 'ignore')
    if len(line)==0:
        continue
    lines.append(line)
alice.close()
text = " ".join(lines)

# Creating sets for comparison in RNN
## Identifying the characters and creating lookup dictionaries for them
chars = set([c for c in text])
len_chars = len(chars)
chars2index = dict([(c, i) for i, c in enumerate(chars)])
index2chars = dict([(i, c) for i, c in enumerate(chars)])

# Creating input and output labels
SEQ_LEN = 10
STEP = 1
input_chars = []
label_chars = []
for i in range(0, len(text)-SEQ_LEN, STEP):
    input_chars.append(text[i:i+SEQ_LEN])
    label_chars.append(text[i+SEQ_LEN])


# Vectorizing Labels
x = np.zeros((len(input_chars), SEQ_LEN, len_chars), dtype=np.bool)
y = np.zeros((len(input_chars), len_chars), dtype=np.bool)

for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        x[i, j, chars2index[ch]] = 1
    y[i, chars2index[label_chars[i]]] = 1

# HyperParameter Definition

HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100


#Creating the Model
model = Sequential()
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), input_shape=(SEQ_LEN, len_chars)))
model.add(Bidirectional(LSTM(HIDDEN_SIZE)))
model.add(Dense(len_chars, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])



# Predicting and testing the model
for iteration in range(NUM_ITERATIONS):
    print('='*50)
    print("Iteration #: %d"%(iteration))
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating text from the seed : %s \n"%(test_chars))
    print(test_chars, end='')
    for i in range(NUM_PREDS_PER_EPOCH):
        X_test = np.zeros((1, SEQ_LEN, len_chars))
        for i, ch in enumerate(test_chars):
            X_test[0, i, chars2index[ch]] = 1
        pred = model.predict(X_test, verbose=2)[0]
        y_pred = index2chars[np.argmax(pred)]
        print(y_pred, end='')
        test_chars=test_chars[1:] + y_pred
print()





