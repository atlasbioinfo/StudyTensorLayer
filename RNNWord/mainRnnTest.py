# import numpy
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM
# from keras.callbacks import ModelCheckpoint
# from keras.utils import np_utils

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
    x.append(char_to_int[predict])

n_patten=len(x)