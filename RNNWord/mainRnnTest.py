import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

raw_text = open("./study.txt", encoding="utf8").read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

step = 100
x = []
y = []

for i in range(0, len(raw_text) - step):
    given = raw_text[i:i + step]
    predict = raw_text[i + step]
    x.append([char_to_int[char] for char in given])
    y.append(char_to_int[predict])

n_patten = len(x)
n_vocab = len(chars)

x = numpy.reshape(x, (n_patten, step, 1))
x = x / float(n_vocab)

y = np_utils.to_categorical(y)

model = Sequential()
model.add(LSTM(128, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(x, y, nb_epoch=20, batch_size=8192)
model.save("./trainedModel.tf")


def predict_next(input_array):
    x = numpy.reshape(input_array, (1, step, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y


def string_to_index(raw_input):
    res = []
    for c in raw_input[(len(raw_input) - step):]:
        res.append(char_to_int[c])
    return res


def y_to_char(y):
    largest_index = y.argmax()
    c = int_to_char[largest_index]
    return c


def generate_article(init, rounds=500):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string
