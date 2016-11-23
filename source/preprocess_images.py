import numpy as np
import math
import pickle
import os

from matplotlib import pyplot as plt
from pylab import cm

from skimage.io import imread, imsave
from skimage import segmentation, measure
from skimage.filter.rank import median, noise_filter
from skimage.morphology import disk, dilation

from swig.reinforceEdges import reinforceEdges as swig

## PARAMETERS #################################################
#
## Path to data
path_train = "../data/train"
path_test = "../data/test"
path_classes = "../data/train_feature_extraction/classes.dat"
path_class_names = "../data/train_feature_extraction/class_names.dat"
path_train_cropped = "../data/train_feature_extraction/cropped"
path_test_cropped = "../data/test_feature_extraction/cropped"
path_train_regions = "../data/train_feature_extraction/regions"
path_test_regions = "../data/test_feature_extraction/regions"
#
#
###############################################################

def reinforceEdges(I):
	res = 0
	for i in range(4):
		top = np.mean(I[0:2, i:i+2])
		bottom = np.mean(I[3:5, (3-i):(5-i)])
		res = max(res, top*bottom)
	
	top = np.mean(I[1:3, 0:2])
	bottom = np.mean(I[2:4, 3:5])
	res = max(res, top*bottom)

	top = np.mean(I[1:3, 3:5])
	bottom = np.mean(I[2:4, 0:2])
	res = max(res, top*bottom)

	res = math.sqrt(res) + I[2,2] - 0.5*np.mean(I)
	res = max(0,min(1,res))

	return res

def getLargestRegion(img_dilated, img_thresh):
	region_max = None
	labels = measure.label(img_dilated)
	labels = img_thresh*(labels + np.ones(labels.shape)) # all elements (+ 1) to handle 0 labels
	labels = labels.astype(int)
	regions = measure.regionprops(labels)
	bad_max = 0

	for region in regions:
		# skip if > 50% of the region was initially 0
		# if sum(img_thresh[labels == region.label])*1.0/region.filled_area < 0.15:
		# 	bad_max = max(region.filled_area,bad_max)
		# 	plt.imshow(labels==region.label, cmap=cm.gray)
		# 	plt.show()
		# 	continue
		if region_max is None:
			region_max = region
		elif region_max.filled_area < region.filled_area:
			region_max = region

	# if (region_max == None) or (bad_max > region_max.filled_area):
	# 	print("Baaaad max = " + str(bad_max))
	# 	print("Thresh.-area: " + str(sum(img_thresh[labels == region.label])))
	# 	sub1 = plt.subplot(1,2,1)
	# 	plt.imshow(img_dilated, cmap = cm.gray_r)
	# 	sub2 = plt.subplot(1,2,2)
	# 	plt.imshow(img_thresh, cmap = cm.gray_r)
	# 	plt.show()
	return region_max

def preprocessTraining():
	## Count images and create directories
	n = 0
	m = 0
	class_names = {}
	classes = {}
	for class_name in os.listdir(path_train):
		class_names[m] = class_name
		classes[m] = []
		for image_name in os.listdir(path_train + "/" + class_name):
			if (image_name[-4:] == ".jpg"):
				classes[m] += [image_name[:-4]]
			n += 1
		m += 1
	pickle.dump(classes, open(path_classes,"wb"))
	pickle.dump(class_names, open(path_class_names,"wb"))

	i = 0
	for y in classes:
		print(class_names[y], flush=True)
		for img_id in classes[y]:
			img = imread(path_train + "/" + class_names[y] + "/" + str(img_id) + ".jpg", as_grey=True)/255
			ones = np.ones(img.shape)
			(rows,cols) = img.shape

			## Remove noise
			img_filtered = np.minimum(ones, img + noise_filter(img, disk(1))/255)
			## Reinforce the edges (can be lost through denoising and thresholding)
			swig.reinforceEdges(img_filtered)

			## Threshold pixels to get a binary image
			thresh = 0.6 + 0.4*np.mean(img_filtered[img_filtered>0.2])**12
			img_thresh = np.where(img_filtered > thresh, 0.,1.0)

			## Dilate image
			img_dilated = dilation(img_thresh, disk(2))

			## Extract largest component and discard the rest
			region = getLargestRegion(img_dilated,img_thresh)
			if region == None:
				img_cropped = img
			else:
				img_cropped = region.convex_image
				(minx,miny,maxx,maxy) = region.bbox
				img_cropped = np.maximum(np.ones(img_cropped.shape) - img_cropped, img[minx:maxx, miny:maxy])

			pickle.dump(region, open(path_train_regions + "/" + img_id + ".dat", "wb") )
			imsave(path_train_cropped  + "/" + img_id + ".jpg", img_cropped)

			# Show progress
			if (i%300 == 0):
				print("Preprocessing images: " + str(100*i//n) + " %", flush=True)
			i += 1

def preprocessTest():
	## Count images and create directories
	n = 0
	for image_name in os.listdir(path_test):
		if (image_name[-4:] != ".jpg"):
			print("non-image file")
			return(":(")
			# os.remove(path_test + "/" + image_name)
		n += 1

	i = 0
	
	for image_name in os.listdir(path_test):
		img = imread(path_test + "/" + image_name, as_grey=True)/255
		ones = np.ones(img.shape)
		(rows,cols) = img.shape

		## Remove noise
		img_filtered = np.minimum(ones, img + noise_filter(img, disk(1))/255)
		## Reinforce the edges (can be lost through denoising and thresholding)
		swig.reinforceEdges(img_filtered)

		## Threshold pixels to get a binary image
		thresh = 0.6 + 0.4*np.mean(img_filtered[img_filtered>0.2])**12
		img_thresh = np.where(img_filtered > thresh, 0.,1.0)

		## Dilate image
		img_dilated = dilation(img_thresh, disk(2))

		## Extract largest component and discard the rest
		region = getLargestRegion(img_dilated,img_thresh)
		if region == None:
			img_cropped = img
		else:
			img_cropped = region.convex_image
			(minx,miny,maxx,maxy) = region.bbox
			img_cropped = np.maximum(np.ones(img_cropped.shape) - img_cropped, img[minx:maxx, miny:maxy])

		pickle.dump(region, open(path_test_regions + "/" + image_name[:-4] + ".dat", "wb") )
		imsave(path_test_cropped  + "/" + image_name, img_cropped)

		# Show progress
		if (i%1300 == 0):
			print("Preprocessing images: " + str(100*i//n) + " %", flush=True)
		i += 1

if __name__ == '__main__':
	preprocessTest()

