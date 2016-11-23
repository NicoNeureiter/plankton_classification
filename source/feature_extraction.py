from skimage.io import imread
from skimage.transform import resize, rotate, AffineTransform
from scipy.fftpack import fft2

import os
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation, measure
from skimage.morphology import disk
from skimage.filters.rank import noise_filter
from scipy.ndimage.filters import laplace
from skimage.exposure import adjust_gamma
import sys
sys.path.append('/usr/local/lib/python3.4/dist-packages/skimage/feature')
from skimage.feature import local_binary_pattern

import numpy as np
import pickle
import math
import random
from sklearn.preprocessing import normalize

import multiprocessing

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

import warnings
warnings.filterwarnings("ignore")



## PARAMETERS #################################################
#
## To wich size should the cropped images be scaled ?
crop_dim = 25
crop_size = crop_dim*crop_dim
fft_n = 36
num_bins = 8
#
## How many features will we include for every image ?
# num_features = crop_size + fft_n + 5*num_bins + 20
# dict_size = 25
# num_features = dict_size
#
## How many augmentation (transformations) of the same image will we add ?
num_augmentations = 1
#
## Patch size
patch_dim = (7,7)
patch_size = patch_dim[0]*patch_dim[1]
#
## Path to data
path_train = "../data/train"
path_train_pp = "../data/train_preprocessed"
path_test = "../data/test"
path_test_pp = "../data/test_preprocessed"
path_classes = "../data/train_feature_extraction/classes.dat"
path_class_names = "../data/train_feature_extraction/class_names.dat"
path_train_cropped = "../data/train_feature_extraction/cropped"
path_test_cropped = "../data/test_feature_extraction/cropped"
path_train_regions = "../data/train_feature_extraction/regions"
path_test_regions = "../data/test_feature_extraction/regions"
#
###############################################################


from swig.SOM import test_kernel_swig as swig_som
from swig.distToTree import distToTree as swig_distToTree

def swigSOM(img,k):
	res = swig_som.som(img, 2*k)
	res = np.reshape(res,(k,2))
	return res

def swigDistToTree(cents,dims):
	res = np.ones(dims)
	swig_distToTree.distToTree(cents.flatten(),res);
	return res

def invertImage(img):
	return np.ones(img.shape) - img

def normalize_hist(lst):
    s = sum(lst)
    if s == 0:
    	return [1/len(lst)]*len(lst)
    	print("Zero-hist !!!")
    else:
    	return map(lambda x: x/s, lst)

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


path_dict = "../data/dictionary.dat"
U = pickle.load(open(path_dict,"rb"))
kwargs = {'transform_n_nonzero_coefs': 4, 'n_jobs': 2}
U.set_params(transform_algorithm="omp", **kwargs)
def computeDictCodes(img):
	## Load dictionary (has to be precomputed -> build_dictionary.py)

	## Extract batches 
	(h,w) = img.shape
	if h < patch_dim[0]:
		temp = np.ones((patch_dim[0],w+1))
		temp[:h,:-1] = img
		img = temp
		h = patch_dim[0]
		w += 1
	if w <= patch_dim[1]:
		temp = np.ones((h+1,patch_dim[1]))
		temp[:-1,:w] = img
		img = temp
		h += 1
	patches = extract_patches_2d(img,patch_dim)
	patches = patches.reshape(patches.shape[0],-1)
	patches_tmp = patches[~np.all(patches > 0.98, axis=1)]
	if patches_tmp.shape[0]>0:
		patches = patches_tmp
	
	code = U.transform(patches)
	# print(code)

	x = sum(code != 0)

	return x

def angle(p0,p1,p2):
	v0 = p1 - p0
	l0 = np.linalg.norm(v0)
	if l0 > 0:
		v0 /= l0
	else:
		return 0
	v1 = p2 - p1
	l1 = np.linalg.norm(v1)
	if l1 > 0:
		v1 /= l1
	else:
		return 0
	res = np.arccos( np.dot(v0,v1) )
	return min(res, 2*np.pi - res)


from swig.reinforceEdges import reinforceEdges as swig
def computeFeatures(img, region):
	x = []
	x_hist = []
	# img = np.minimum(np.ones(img.shape), img + noise_filter(img, disk(1))/255)
	# swig.reinforceEdges(img)
	(rows,cols) = img.shape
	diag = math.sqrt(rows*rows + cols*cols)
	# if not region is None:
	# 	img = rotate(img,region.orientation)

	## Add rescaled image
		# img_sq = padToSquare(img)
		# img_sq = resize(img_sq, (crop_dim,crop_dim))
		# x += list(img_sq.flatten())
		
	## Augmentation
		# if random.randrange(50)==0:
		# 	plt.imshow(img,interpolation='none',cmap=cm.gray)
		# 	plt.show()
		# for j in range(4):
		# 	X[i, 0:crop_size] = im2col(rotate(img_sq, j*90), crop_dim)

	for k in [1,3,5]:
		if k > 1:
			centroids = swigSOM(img,k)
			som_len = 0
			for c in range(k-1):
				som_len += np.linalg.norm(centroids[c+1]-centroids[c])
			x += [som_len/((k-1)*diag)]

			som_angles = 0
			for c in range(1,k-1):
				som_angles += angle(centroids[c-1], centroids[c], centroids[c+1])
			x += [som_angles/(k-2)]
		else:
			moments = measure.moments(img, order=1)
			centroid_x = moments[0,1] / moments[0,0]
			centroid_y = moments[1,0] / moments[0,0]
			centroids = np.ones((2,2))
			centroids[0,0] = centroid_x
			centroids[1,0] = centroid_x
			centroids[0,1] = centroid_y
			centroids[1,1] = centroid_y
			x += [centroid_x,centroid_y]

		img_dist = swigDistToTree(centroids,img.shape)
		# def histIndex(val):
		# 	return min(math.floor(2 * math.sqrt(k) * num_bins * val), num_bins-1 )
		# vHistIndex = np.vectorize(histIndex)
		# binAreas = vHistIndex(img_dist)
		# hist = [0]*num_bins
		# for b in range(num_bins):
		# 	bin_area = (binAreas==b)
		# 	hist[b] += np.sum(bin_area*invertImage(img))
		# x_hist += normalize_hist(hist)
		values = np.repeat(img_dist.flatten(), (img.flatten()*51).astype(int) )
		# print(values.shape,flush=True)
		x += [np.percentile(values[~np.isnan(values)],25)]
		x += [np.percentile(values[~np.isnan(values)],50)]
		x += [np.percentile(values[~np.isnan(values)],75)]
		x += [np.percentile(values[~np.isnan(values)],90)]



	hist_img = np.histogram(img,num_bins)
	hist_lst = list(hist_img[0])
	x_hist += normalize_hist(hist_lst)

	lbp = local_binary_pattern(invertImage(img) ,16 ,4, method='uniform')
	hist_lbp = np.histogram(lbp, num_bins)
	hist_lst = list(hist_lbp[0])
	x_hist += normalize_hist(hist_lst)

	# # img_laplace = laplace(img)
	# # hist_laplace = np.histogram(img_laplace,num_bins)
	# # hist_lst = list(hist_laplace[0])
	# # x_hist += normalize_hist(hist_lst)

	# # fft_mat = fft2(img,(6,6))
	# # fft_lst = list(fft_mat.flatten())
	# # x += normalize_hist(fft_lst)

	x += [rows/cols]
	if not region is None:
		x += [region.extent]
		x += [region.orientation]
	# 	# x += list(region.moments.flatten())
		x += [region.eccentricity]
	else:
		x += [-1]
		x += [10]
		x += [-1]

	# if sum(np.array(x_hist) < 0) > 0:
	# 	print(np.array(x_hist)<0)
	return (x_hist + x)

def getImageData(img_id):
	img  = imread(path_test_cropped + "/" + img_id + ".jpg", as_grey=True)/255
	region = pickle.load(open(path_test_regions + "/" + img_id + ".dat","rb") )
	return computeFeatures(img,region)

	
def getTrainingData():
	classes = pickle.load(open(path_classes, "rb") )

	num_images = 0
	for label in classes:
		num_images += len(classes[label])

	## Get feature Matrix X
	y = np.zeros((num_augmentations * num_images))
	X = []
	i = 0
	for label in classes:
		for img_id in classes[label]:
			img = imread(path_train_cropped + "/" + img_id + ".jpg", as_grey=True)/255
			region = pickle.load(open(path_train_regions + "/" + img_id + ".dat","rb") )
			X += [computeFeatures(img,region)]
			y[i] = label

			if np.sum(np.isnan(np.array(X[i]).astype(float))):
				print("NANANANAN ----- img_id:" + str(img_id),flush=True)
				print(np.where(np.isnan(np.array(X[i]))))

			i += 1
		print(label, flush=True)
	return (np.array(X).astype(float), y.astype(int)) ## If needed: Compute and return label_map

def getTestData():
	pool = multiprocessing.Pool(3)
	img_names = os.listdir(path_test_cropped)
	img_ids = map(lambda s: s[:-4], img_names)
	X = pool.map(getImageData, img_ids)

	return (np.array(X).astype(float), img_ids)
		

