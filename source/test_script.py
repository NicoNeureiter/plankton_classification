from preprocess_images import getLargestRegion
import feature_extraction as FE

import skimage
from skimage.filters.rank import noise_filter
from skimage.morphology import disk, opening, square, skeletonize
from skimage.io import imread
from skimage import segmentation, measure, morphology
from scipy.ndimage.filters import laplace
from skimage.draw import line, set_color

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# from sklearn.cluster import Kmeans

from matplotlib import pyplot as plt
from pylab import cm

from time import time
import random
import math

import numpy as np
import pickle
import os
import multiprocessing


# ######## Compute Dictionaries

def compute_dict_features(files):
	i = 0
	n = len(files)
	start = time()
	for file_name in files:
		img = imread(path_src + "/" + file_name)/255
		x = FE.computeDictCodes(img)
		pickle.dump(x,open(path_code + "/" + file_name,"wb"))
		os.remove(path_src + "/" + file_name)
		if (i%50 == 0):
			print(str((10000*i//n)/100) + " %    " + str(int(time() - start)) + " s", flush=True)
		i += 1



# mode = "test"
# if mode == "test":
# 	path_code = "../data/test_dict_codes"
# 	path_src = path_test_pp_copy
# 	files_all = os.listdir(path_test_pp_copy)
# 	n = len(files_all)
# 	# files_part = [files_all[i::cores] for i in range(cores)]

# 	# start = time()
# 	compute_dict_features(files_all)
# 	# pool.map(compute_dict_features,files_part)

if __name__=="__main__":
	path_test_cropped_copy = "../data/test_preprocessed_copy"
	path_train_cropped_copy = "../data/train_feature_extraction/cropped2"

	path_code = "../data/train_feature_extraction/dict_codes/"
	path_src = path_train_cropped_copy

	cores = 2
	pool = multiprocessing.Pool(cores)

	files_all = list(os.listdir(path_train_cropped_copy))
	n = len(files_all)
	files_part = [files_all[i::cores] for i in range(cores)]
	# paths_part = [paths[i::cores] for i in range(cores)]

	start = time()
	# compute_dict_features(files_all)
	pool.map(compute_dict_features,files_part)



############## Show reinforceEdges filter

	# from swig.reinforceEdges import reinforceEdges as swig


	# rows = 2
	# cols = 4
	# n = rows*cols
	# class_name = "acantharia_protist"
	# path_train = "../data/train/" + class_name
	# file_names = random.sample(os.listdir(path_train),n)
	# for (i,file_name) in enumerate(file_names):
	# 	img = imread(path_train + "/" + file_name,as_grey=True)/255
	# 	img = np.minimum(np.ones(img.shape), img + noise_filter(img, disk(1))/255)
	# 	swig.reinforceEdges(img)
	# 	img = FE.padToSquare(img)
	# 	img = FE.resize(img, (25,25))
	# 	sub = plt.subplot(rows,cols,i)
	# 	plt.imshow(img,cmap=cm.gray, interpolation='none')
	# plt.show()


############## Plot region extraction
	# num_subs = 5

	# path_train = "../data/train/"

	# # img = imread("../data/train/copepod_cyclopoid_oithona/22532.jpg", as_grey=True)
	# # img = imread("../data/train/copepod_calanoid_small_longantennae/135901.jpg", as_grey=True)
	# # img = imread("../data/train/copepod_cyclopoid_oithona/43411.jpg", as_grey=True)
	# # img = imread("../data/train/siphonophore_physonect/95245.jpg", as_grey=True)
	# # img = imread("../data/train/hydromedusae_shapeA/55683.jpg", as_grey=True)
	# # img = imread("../data/train/copepod_cyclopoid_oithona/30585.jpg", as_grey=True)

	# img_class = random.choice(os.listdir(path_train))
	# img_name = random.choice(os.listdir(path_train + img_class))
	# print(img_class + "/" + img_name)
	# img = imread(path_train + img_class + "/" + img_name)

	# (rows,cols) = img.shape
	# sub0 = plt.subplot(1,num_subs,1)
	# plt.imshow(img, cmap=cm.gray, interpolation='none')

	# img = img/255
	# img_filtered = img
	# ones = np.ones(img.shape)

	# from swig.reinforceEdges import reinforceEdges as swig

	# img_filtered = np.minimum(ones, img_filtered + noise_filter(img_filtered, disk(1))/255)
	# swig.reinforceEdges(img_filtered)

	# sub1 = plt.subplot(1,num_subs,2)
	# plt.imshow(img_filtered, cmap=cm.gray, interpolation='none')

	# img_diff = img - img_filtered
	# sub2 = plt.subplot(1,num_subs,3)
	# plt.imshow(img_diff, cmap=cm.gray, interpolation='none')


	# # img_thresh = np.where(img_filtered > np.mean(img_filtered), 0.,1.0)
	# # img_thresh = np.where(img_filtered > (1+39*np.mean(img_filtered[np.nonzero(img_filtered)]) )/40, 0.,1.0)
	# img_thresh = np.where(img_filtered > 0.6 + 0.4*np.mean(img_filtered[img_filtered>0.2])**12, 0.,1.0)

	# print(np.mean(img_filtered[img_filtered>0.2])**12)
	# sub2 = plt.subplot(1,num_subs,3)
	# plt.imshow(img_thresh, cmap=cm.gray_r, interpolation='none')


	# img_dilated = morphology.dilation(img_thresh, disk(2))#np.ones((3,3)))
	# sub3 = plt.subplot(1,num_subs,4)
	# plt.imshow(img_dilated, cmap=cm.gray_r, interpolation='none')


	# region = getLargestRegion(img_dilated,img_thresh)
	# if region == None:
	# 	img_cropped = img
	# 	print(region.bbox)
	# else:
	# 	img_cropped = region.convex_image
	# 	(minx,miny,maxx,maxy) = region.bbox
	# 	img_cropped = np.maximum(np.ones(img_cropped.shape) - img_cropped, img[minx:maxx, miny:maxy])

	# sub4 = plt.subplot(1,num_subs,5)
	# plt.imshow(img_cropped, cmap=cm.gray, interpolation='none')
	# plt.show()