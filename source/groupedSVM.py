from sklearn.svm import SVC
# from mSVC import mSVC
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler, SkewedChi2Sampler
from sklearn.base import BaseEstimator, ClassifierMixin
import class_groups as grouping
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.preprocessing import normalize
import numpy as np
import os

def sigmoid(X):
	return 1/(1+np.exp(-1*X))


class groupedSVM(BaseEstimator, ClassifierMixin):
	num_groups = 20

	def __init__(self,
				cache_size=500, tol=0.01, kernel="rbf",
				skewedness=0.0005, gamma=1/40,
				use_SGD=False, n_iter=25, alpha_g=0.001, alpha_all=0.005): # last line is only for SGD
		self.groups = list(grouping.get_group2class().keys())

		self.cache_size = cache_size
		self.tol = tol
		self.kernel = kernel
		self.use_SGD = use_SGD
		self.n_iter = n_iter
		self.alpha_g = alpha_g
		self.alpha_all = alpha_all
		self.skewedness = skewedness
		self.gamma = gamma

		self.rbf_feature = RBFSampler(gamma=gamma, n_components=500)
		self.chi2_feature_0 = SkewedChi2Sampler(skewedness=skewedness, n_components=300)
		self.chi2_feature_1 = SkewedChi2Sampler(skewedness=skewedness, n_components=300)
		self.chi2_feature_2 = SkewedChi2Sampler(skewedness=skewedness, n_components=300)

		if (use_SGD):
			self.SVMs = [SGD(alpha=alpha_g, epsilon=0.1, n_iter=n_iter*4//5, n_jobs=4) for _ in self.groups]
			self.SVM_all = SGD(alpha=alpha_all, epsilon=0.1, n_iter=n_iter, n_jobs=4)
		else:
			self.SVMs = [SVC(cache_size=cache_size,tol=tol,kernel=kernel,C=500) for _ in self.groups]
			self.SVM_all = SVC(cache_size=cache_size,tol=tol,kernel=kernel,C=1000)

	def get_params(self, deep=True):
		return {
			'cache_size': self.cache_size,
			'tol': self.tol,
			'kernel': self.kernel,
			'use_SGD': self.use_SGD,
			'n_iter': self.n_iter,
			'alpha_g': self.alpha_g,
			'alpha_all': self.alpha_all,
			'skewedness': self.skewedness,
			'gamma': self.gamma,
			}

	def set_params(self, **parameters):
		for (param, value) in parameters.items():
			setattr(self, param, value)
		return self

	def fit(self, X, y):
		##
			# pool = multiprocessing.Pool(num_jobs)
			# c_svms = zip(self.groups,self.SVMs)
			# pool.map(lambda c, svm: fit_svm_c(svm,X_train,y_train,c), c_svms)
		self.fit_kernelApproximation(X)
		X = self.transform_kernelApproximation(X)
		n = len(y)
		y_all = []
		X_all = []
		class2group = grouping.get_class2group()
		for i in range(n):
			for g in class2group[y[i]]:
				y_all += [g]
				X_all += [X[i]]
		y_all = np.array(y_all)
		X_all = np.array(X_all)
		self.SVM_all.fit(X_all,y_all)

		group2class = grouping.get_group2class()
		for g in self.groups:
			y_g = []
			X_g = []
			for i in range(n):
				c = y[i]
				if c in group2class[g]:
					y_g += [c]
					X_g += [X[i]]
			
			self.SVMs[g].fit(X_g,y_g)

	def predict(self, X):
		# y_all = self.SVM_all.predict(X)
		# n = len(y_all)
		# y = np.zeros(n)
		# for g in self.groups:
		# 	idx_g = (y_all == g)
		# 	y[idx_g] = self.SVMs[g].predict(X[idx_g])
		return np.argmax(self.predict_proba(X), axis=1)

	def predict_proba(self, X):
		X = self.transform_kernelApproximation(X)
		group2class = grouping.get_group2class()
		k = 121

		
		dfun_all = self.SVM_all.decision_function(X)
		(n, num_groups) = dfun_all.shape
		probs_all = sigmoid(dfun_all*10)
		probs_all = normalize(probs_all,axis=1, norm='l1')

		probs = np.zeros((n, k))
		for g in self.groups:
			print(g)

			idx_classes = np.zeros(k).astype(bool)
			idx_samples = np.zeros(n).astype(bool)
			for i in range(k):
				idx_classes[i] = (i in group2class[g])
			for i in range(n):
				idx_samples[i] = (probs_all[i,g] > 0.1 or g == np.argmax(probs_all[i,:]))

			probs_in_g = sigmoid(self.SVMs[g].decision_function(X[idx_samples,:])*5)
			if len(probs_in_g.shape) == 1:
				probs_in_g = np.array([probs_in_g, 1-probs_in_g]).T
			probs_in_g = normalize(probs_in_g,axis=1, norm='l1')
			
			prob_of_g = np.tile(probs_all[idx_samples,g], (probs_in_g.shape[1],1)).T
			
			prob = prob_of_g * probs_in_g
			i = 0
			for i_c in np.where(idx_classes)[0]:
				probs[idx_samples,i_c] += prob[:,i]
				i += 1

		return normalize(probs+0.003, axis=1, norm='l1')




		# return np.argmax(decision_funs,axis=0)

	# def score(self, X, y):
		# y_pred = self.predict(X)
		# return np.mean(y_pred==y)

	def score(self, X, y):
		P_pred = self.predict_proba(X)
		res = 0
		for i, c in enumerate(y):
			res += np.log(P_pred[i,c])
		return res/len(y)

	def log_score(self, X, y):
		P_pred = self.predict_proba(X)
		res = 0
		for i, c in enumerate(y):
			res -= np.log(P_pred[i,c])
		return res/len(y)

	def fit_kernelApproximation(self, X):
		self.chi2_feature_0.fit(X[:,:25])
		self.chi2_feature_1.fit(X[:,25:33])
		self.chi2_feature_2.fit(X[:,33:41])
		self.rbf_feature.fit(X[:,41:])

	def transform_kernelApproximation(self, X):
		X_chi0 = self.chi2_feature_0.transform(X[:,:25])
		X_chi1 = self.chi2_feature_1.transform(X[:,25:33])
		X_chi2 = self.chi2_feature_2.transform(X[:,33:41])
		X_rbf = self.rbf_feature.transform(X[:,41:])
		return np.concatenate([X_chi0, X_chi1, X_chi2, X_rbf], 1)