from time import time
import numpy as np
import pickle
from random import sample

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from pystruct.models import GraphCRF
from pystruct.learners import OneSlackSSVM, NSlackSSVM, SubgradientSSVM

iris = load_iris()
# X, y = iris.data, iris.target

idx = sample(range(1200),400)
X = pickle.load(open("../data/X_train.dat","rb"))[idx,:20]
y = pickle.load(open("../data/y_train.dat","rb")).astype(int)[idx]
print(np.unique(y).shape)
ydict = {}
nval = 0
for (i,val) in enumerate(y):
	if val in ydict:
		y[i] = ydict[val]
	else:
		ydict[val] = nval
		y[i] = nval
		nval += 1
print(np.unique(y))
print(y.shape, flush=True)
# 
# print(idx)
# X = X[idx,:10]
# y = y[idx]
# print(X.shape)
# print(y.shape)


# make each example into a tuple of a single feature vector and an empty edge
# list
X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]
Y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_, Y)

pbl = GraphCRF(inference_method='unary')
svm = NSlackSSVM(pbl, C=100)


start = time()
svm.fit(X_train, y_train)
time_svm = time() - start
y_pred = np.vstack(svm.predict(X_test))
print("Score with pystruct crf svm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_svm))