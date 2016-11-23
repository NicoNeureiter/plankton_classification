import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning, DictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
from pylab import cm

from random import sample

import pickle
import os

import warnings
warnings.filterwarnings("ignore")


## PARAMETERS #################################################
#
## Path to data
path_train_cropped = "../data/train_feature_extraction/cropped"
path_test = "../data/test"
path_dict = "../data/dictionary.dat"
path_classes = "../data/train_feature_extraction/classes.dat"
path_class_names = "../data/train_feature_extraction/class_names.dat"
#
## Patch size
patch_dim = (7,7)
patch_size = patch_dim[0]*patch_dim[1]
#
## Image size
img_dim = (40,40)
img_size = img_dim[0]*img_dim[1]
#
#
###############################################################

def padToSquare(img):
	(height,width) = img.shape
	if height < width:
		diff = width - height
		img_square = np.ones((width, width))
		img_square[0:height, :] = img
		img_square = np.roll(img_square, diff//2, axis=0)
	else:
		diff = height - width
		img_square = np.ones((height, height))
		img_square[:, 0:width] = img
		img_square = np.roll(img_square, diff//2)
	return img_square

def getClassDictionaries():
	i = 0
	label = 0
	class_dict = []
	for class_name in os.listdir(path_train):
		class_dict += [MiniBatchDictionaryLearning(n_components=15, alpha=10, n_iter=200)]
		X = []
		for file_name in os.listdir(path_train + "/" + class_name):
			(img, _, lbl) = pickle.load(open(path_train + "/" + class_name + "/" + file_name,"rb") )
			if lbl != label:
				print("AAAAAAAAAAAAAAH")
			## Consider taking the originals (i.e. different size)
			# img = padToSquare(img)
			# img = resize(img,img_dim)
			
			(h,w) = img.shape
			if h < patch_dim[0]:
				temp = np.ones((patch_dim[0],w))
				temp[:h,:] = img
				img = temp
				h = patch_dim[0]
			if w < patch_dim[1]:
				temp = np.ones((h,patch_dim[1]))
				temp[:,:w] = img
				img = temp

			print(img.shape)


			img_patches = extract_patches_2d(img,patch_dim)
			if img_patches.shape[0] == 0:
				print(file_name, flush=True)
				print(img.shape, flush=True)
				plt.imshow(img, cmap=cm.gray, interpolation='none')
				plt.show()
			img_patches = img_patches.reshape(img_patches.shape[0],-1)
			X += [img_patches]
			i += 1

		X = np.concatenate(X,axis=0)
		class_dict[label].fit(np.array(X))
		label += 1
		print(label, flush=True)
	return class_dict

def getOverallDict ():
	classes = pickle.load(open(path_classes,"rb"))
	class_names = pickle.load(open(path_class_names,"rb"))
	i = 0
	X = np.zeros((0,patch_size))
	dictionary = MiniBatchDictionaryLearning(n_components=35, alpha=20, n_iter=700, n_jobs=3)
	for label in classes:
		class_name = class_names[label]
		X_class = []
		if class_name in ["artifacts","artifacts_edge"]:
			continue
		for img_id in classes[label]:
			img = imread(path_train_cropped + "/" + img_id + ".jpg")/255

			(h,w) = img.shape
			if h < patch_dim[0]:
				temp = np.ones((patch_dim[0],w))
				temp[:h,:] = img
				img = temp
				h = patch_dim[0]
			if w < patch_dim[1]:
				temp = np.ones((h,patch_dim[1]))
				temp[:,:w] = img
				img = temp

			# print(img.shape)

			img_patches = extract_patches_2d(img,patch_dim)
			# if img_patches.shape[0] < 10:
			# 	print(class_name+ "/" + file_name)
			# 	print(img.shape)
			# 	plt.imshow(img, cmap=cm.gray, interpolation='none')
			# 	plt.show()
			img_patches = img_patches.reshape(img_patches.shape[0],-1)
			old_shape = img_patches.shape
			img_patches = img_patches[~np.all(img_patches > 0.98, axis=1)]
			# if old_shape != img_patches.shape:
			# 	print((img_patches.shape,old_shape))
			# 	print("--------------------------")
			X_class += list(img_patches)
			i += 1
		label += 1
		if (len(X_class) > 33000):
			X_class = sample(X_class, 30000)
		# X_class = np.concatenate(X_class)
		print(str(label) + " --- " + str(len(X_class)), flush=True)
		X_class = np.array(X_class)
		X = np.concatenate([X,X_class],axis=0)
	print(X.shape)
	pickle.dump(X, open("../data/temp.dat","wb"))
	dictionary.fit(X)
	return dictionary

if __name__ == '__main__':
	dictionary = getOverallDict()
	pickle.dump(dictionary, open(path_dict,"wb"))