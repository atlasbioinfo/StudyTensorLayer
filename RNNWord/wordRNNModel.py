import os
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from gensim.models.word2vec import Word2Vec

# raw_text = open("./study.txt", encoding="utf8").read()
raw_text = ''
print(1)
for file in os.listdir("./"):
    if file.endswith(".txt"):
        raw_text += open("./"+file, errors='ignore').read()
print(1)
raw_text = raw_text.lower()
sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
#分成小单词，中文对应另一个分词器
sents = sentensor.tokenize(raw_text)
corpus = []
print(1)
for sen in sents:
    corpus.append(nltk.word_tokenize(sen))

# print(len(corpus))
# print(corpus[:3])
# print(1)
#W2V把单词变成数字坐标
# w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
# w2v_model.save("w2vmodel.w2v")

w2v_model = Word2Vec.load("w2vmodel.w2v")
# print(w2v_model['office'])
print(1)
raw_input = [item for sublist in corpus for item in sublist]

# print(raw_input[2333])
print(1)
text_stream = []
vocab = w2v_model.wv.vocab

for word in raw_input:
    if word in vocab:
        text_stream.append(word)
print(1)
# print(text_stream[233])

#
seq_length = 10
x = []
y = []
print(1)
for i in range(0, len(text_stream) - seq_length):
    given = text_stream[i:i+seq_length]
    predict = text_stream[i+seq_length]
    x.append(np.array([w2v_model[word] for word in given]))
    y.append(w2v_model[predict])

# print(len(x))
# print(len(y))

x = np.reshape(x, (-1, seq_length, 128))
y = np.reshape(y, (-1, 128))

model = Sequential()
model.add(LSTM(512, dropout_W=0.2, dropout_U=0.2, input_shape=(seq_length, 128), return_sequences=True))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, nb_epoch=100, batch_size=4096)

model.save("./trainedModelWord3.tf")


def predict_next(input_array):
    x = np.reshape(input_array, (-1, seq_length, 128))
    y = model.predict(x)
    return y

def string_to_index(raw_input):
    raw_input = raw_input.lower()
    input_stream = nltk.word_tokenize(raw_input)
    res = []
    for word in input_stream[(len(input_stream)-seq_length):]:
        res.append(w2v_model[word])
    return res


def y_to_word(y):
    word = w2v_model.most_similar(positive=y, topn=1)
    return word

def generate_artical(init, rounds=30):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_word(predict_next(string_to_index(in_string)))
        in_string += ' '+ n[0][0]
    return in_string

init = "Nothing is difficult if you put your heart into it"
article = generate_artical(init)
print(article)