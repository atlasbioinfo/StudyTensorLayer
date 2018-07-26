import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = np.loadtxt("./iris.txt", delimiter=',')
np.random.shuffle(data)
np.random.shuffle(data)
np.random.shuffle(data)
x = []
y = []
for i in range(0, len(data)):
    x.append(data[i][0:4])
    y.append(data[i][4])

x = np.array(x)
y = np.array(y)
y_ = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

for i in range(0, len(data)):
    np.append(x[i], y_[i])


