from sklearn import cross_validation, grid_search
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import feature_extraction as FE
import class_groups as grouping
from sklearn.preprocessing import normalize

import os
import pickle
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

## Classifiers
from sklearn.ensemble import RandomForestClassifier as RF
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC as SVC
from mSVC import mSVC
from groupedSVM import groupedSVM
from sklearn.metrics.pairwise import chi2_kernel, rbf_kernel
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler, SkewedChi2Sampler
from sklearn.linear_model import SGDClassifier as SGD

from time import time

import warnings
warnings.filterwarnings("ignore")


## Get training data
	# path_train_dict_codes = "../data/train_feature_extraction/dict_codes"
	# X = []
	# label = 0
	# path_class_names = "../data/train_feature_extraction/class_names.dat"
	# classes = pickle.load(open(path_classes, "rb") )
	# # class_names = pickle.load(open(path_class_names, "rb") )
	# for c in classes:
	# 	for img_id in classes[c]:
	# 		x = pickle.load(open(path_train_dict_codes + "/" + img_id + ".jpg","rb") )
	# 		X += [x]
	# X_dict = np.array(X)	
	# pickle.dump(X_dict,open("../data/train_feature_extraction/train_dict_codes2.dat","wb"))

# (X_train, y_train) = FE.getTrainingData()
# pickle.dump(y_train,open("../data/y_train.dat","wb"))
# pickle.dump(X_train,open("../data/X_train.dat","wb"))

path_classes = "../data/train_feature_extraction/classes.dat"
X_dict = pickle.load(open("../data/train_feature_extraction/train_dict_codes.dat","rb"))
X_train = pickle.load(open("../data/X_train.dat","rb")).astype(float)
X_train = np.concatenate([X_dict, X_train],axis=1)
y_train = pickle.load(open("../data/y_train.dat","rb")).astype(int)


(n,m) = X_train.shape
# n = 12000
perm = np.random.permutation(n)
X_train = X_train[perm,:]
y_train = y_train[perm].astype(int)

path_class_names = "../data/train_feature_extraction/class_names.dat"
class_names = pickle.load(open(path_class_names, "rb") )
path_classes = "../data/train_feature_extraction/classes.dat"
classes = pickle.load(open(path_classes, "rb") )

k = len(classes)
print((n,m,k),flush=True)

def myKernel(x, y):
	gamma_hist = 1

	now = time()
	hist_kernel = 0
	hist_kernel = chi2_kernel(x[:, :25], y[:, :25], gamma_hist)
	hist_kernel += chi2_kernel(x[:, 25:33], y[:, 25:33], gamma_hist)
	hist_kernel += chi2_kernel(x[:, 33:41], y[:, 33:41], gamma_hist)
	print(time()-now,flush=True)

	now = time()
	rbf_kern = rbf_kernel(x[:,41:],
							y[:,41:],1/40)
	print(time()-now,flush=True)

	return hist_kernel + rbf_kern

svm = groupedSVM(use_SGD=True)
parameters = {
	'alpha_g': [0.3, 0.1, 0.03, 0.01, 0.008, 0.005],
	# 'alpha_all': [0.03, 0.01, 0.003,0.001, 0.0003],
	# 'skewedness': [0.001, 0.0003, 0.0001],
	# 'gamma': [1/20, 1/30, 1/40, 1/50, 1/60],
	}

grid = grid_search.GridSearchCV(svm, parameters, n_jobs=3)


start = time()
from scipy import stats
import class_groups as grouping
class2group = grouping.get_class2group()
mode = 0
if mode == 0:					#### Cross Validation ####
	grid.fit(X_train,y_train)
	best = grid.best_estimator_
	print(best.get_params(), flush=True)
	print(" °_°", flush=True)
	print(best.get_params(deep=False), flush=True)
	classifier = best

	neg_log_scores = cross_validation.cross_val_score(classifier, X_train, y_train, cv=5, n_jobs=1)
	print("CV negative log-score: " + str(np.mean(neg_log_scores)))

	## old
		# y = y_train
		# X = X_train
		# num_folds = 2
		# kf = KFold(y, n_folds=num_folds)
		# y_pred = np.zeros(n)
		# it = 0
		# misclassified = [[] for _ in range(k)]
		# for train, test in kf:
		# 	# it += 1
		# 	X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
			
		# 	# svc = mSVC(k,use_SGD=False)
		# 	svc.fit(X_train,y_train)
		# 	# y_pred[test] = svc.predict(X_test)
		# 	logscore = svc.log_score(X_test,y_test)
		# 	print("Log-score in fold: " + str(logscore), flush=True)

			## CC
				# cc = combinedClassifier()
				# cc.fit_candidates(X_train, y_train)
				# print("--- Fitted candidates ---", flush=True)
				# candidates = cc.predict_candidates(X_test)
				# cand_score = 0
				# for i in range(len(y_test)):
				# 	cand_score += (y_test[i] in candidates[i])
				# cand_score /= len(y_test)
				# print(cand_score)
				# print(len(set(candidates)))
				# print("--- Predicted candidates ---", flush=True)
				# cc.fit(X_kernel[train,:],y_train)
				# print("--- Fitted SVMs ---", flush=True)
				# y_pred[test] =  cc.predict(X_kernel[test,:])
				# print(np.unique(y_pred[test]), flush=True)
			
			## RF
				# rf = RF(n_estimators=100, n_jobs=3, max_depth=20)
				# rf.fit(X_train,y_train)
				# y_pred[test] = rf.predict(X_test)

			# print("Score in fold " + str(it) + ": " + str(np.mean(y_pred[test]==y_test)),flush=True)
		# scores = (y_pred==y)


		# for i in range(n):
		# 	if y_pred[i] != y[i]:
		# 		misclassified[y[i]] += [class2group[y_pred[i]][0]]
		# for i in range(k):
		# 	if len(misclassified[i]) > 0.9*len(classes[i]) + 20:
		# 		print(class_names[i],flush=True)
		# 		print(stats.itemfreq(misclassified[i]).T,flush=True)

elif mode == 1:					   #### Prediction ####
	## Get test data
	# (X_test, file_names) = FE.getTestData()

	## In this case: Load the dictionary codes of the test data
		# path_test_dict_codes = "../data/test_dict_codes"
		# X = []
		# for file_name in os.listdir(path_test_dict_codes):
		# 	x = pickle.load(open(path_test_dict_codes + "/" + file_name,"rb"))
		# 	if x.shape != (25,):
		# 		# print(x.shape)
		# 		print(file_name)
		# 	X += [x]
		# print(len(X))
		# X_1 = np.array(X)
		# pickle.dump(X_1,open("../data/test_dict_codes.dat","wb"))

		# (X_2, names) = FE.getTestData()
		# print("computed features", flush=True)
		# pickle.dump(X_2,open("../data/test_feature_extraction/X_test.dat","wb"))

	X_1 = pickle.load(open("../data/test_dict_codes.dat","rb"))
	X_2 = pickle.load(open("../data/test_feature_extraction/X_test.dat","rb"))
	
	print(X_1.shape)
	print(X_2.shape, flush=True)
	X_test = np.concatenate([X_1,X_2], axis=1).astype(float)
	print("... merged", flush=True)

	rbf_feature = RBFSampler(gamma=1.5/m, n_components=600)
	chi2_feature = SkewedChi2Sampler(skewedness=0.0005, n_components=400)
	X_chi0 = chi2_feature.fit_transform(X_test[:,:25])
	X_chi1 = chi2_feature.fit_transform(X_test[:,25:33])
	X_chi2 = chi2_feature.fit_transform(X_test[:,33:41])
	X_rbf = rbf_feature.fit_transform(X_test[:,41:])

	X_kernel = np.concatenate([X_chi0, X_chi1, X_chi2, X_rbf], 1)
	X_test = X_kernel.astype(float)
	print("... computed kernel approximation", flush=True)

	## Write header (class names)
	f_submission = open("../submission.csv","w+")
	path_train = "../data/train"
	class_names = os.listdir(path_train)
	f_submission.write("image,")
	f_submission.write(",".join(class_names) + "\n")

	## Make prediction
	path_test = "../data/test"
	classifier.fit(X_train,y_train)
	print(X_test.shape, flush=True)
	class_probabilites = classifier.predict_proba(X_test)
	i = 0
	img_names = os.listdir("../data/test_feature_extraction/cropped")
	for img_name in img_names:
		# print(class_probabilites[i,:])
		class_prob_string = ",".join(map(str,class_probabilites[i,:]))
		line = img_name + "," + class_prob_string + "\n"
		f_submission.write(line)
		i += 1

	f_submission.close() 