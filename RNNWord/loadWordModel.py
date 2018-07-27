import numpy as np
from keras.models import load_model
from gensim.models.word2vec import Word2Vec
import nltk

seq_length = 10

model = load_model("trainedModelWord1.tf")
#
# raw_text = open("./study.txt", encoding="utf8").read()
# raw_text = raw_text.lower()
# sentensor = nltk.data.load('tokenizers/punkt/english.pickle')
# sents = sentensor.tokenize(raw_text)
#
# corpus = []
#
# for sen in sents:
#     corpus.append(nltk.word_tokenize(sen))

# w2v_model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
w2v_model = Word2Vec.load("w2vmodel.w2v")
# print(w2v_model['office'])

# raw_input = [item for sublist in corpus for item in sublist]

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