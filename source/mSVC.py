from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.metrics.pairwise import additive_chi2_kernel, rbf_kernel

import numpy as np
from time import time

# import multiprocessing


# def fit_svm_c(svm,X,y,c):
# 	y_c = (y==c)
# 	svm.fit(X,y_c)

class mSVC(BaseEstimator):
	def __init__(self,
				cache_size=1500, tol=0.001, kernel="rbf", C=100,
				use_SGD=False,alpha=0.01,n_iter=30): # last line is only for SGD
		self.cache_size = cache_size
		self.tol = tol
		self.kernel = kernel
		self.C = C
		self.use_SGD = use_SGD
		self.alpha = alpha
		self.n_iter = n_iter
		self.use_SGD = use_SGD
		self.SVMs = []
			

	def fit(self, X_train,y_train):
		# pool = multiprocessing.Pool(num_jobs)
		# c_svms = zip(self.classes,self.SVMs)
		# pool.map(lambda c, svm: fit_svm_c(svm,X_train,y_train,c), c_svms)
		classes = np.unique(y_train)
		if (self.use_SGD):
			self.SVMs = [SGD(alpha=self.alpha,epsilon=0.1,n_iter=self.n_iter,n_jobs=2) for _ in classes]
		else:
			self.SVMs = [SVC(cache_size=self.cache_size,tol=self.tol,kernel=self.kernel,C=self.C) for _ in classes]
		
		for c in classes:
			y_c = (y_train==c)
			self.SVMs[c].fit(X_train,y_c)

	def decision_function(self, X_test):
		decision_funs = np.array([svm.decision_function(X_test) for svm in self.SVMs])
		return np.max(decision_funs,axis=0)

	def predict(self, X_test):
		decision_funs = np.array([svm.decision_function(X_test) for svm in self.SVMs])
		return np.argmax(decision_funs,axis=0)

	def score(self, X, y):
		y_pred = predict(X)
		return (y_pred==y)