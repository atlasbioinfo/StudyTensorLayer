import numpy
from keras.models import load_model

model = load_model("./trainedModel.tf")

raw_text = open("./study.txt", encoding="utf8").read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

step = 100
n_vocab = len(chars)


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


init = "Your mind is like this water, my friend , when it is agitated ,it becomes difficult to see ,but if you allow it to settle , the answer becomes"
artical = generate_article(init)

print(artical)