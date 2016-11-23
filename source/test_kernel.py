from preprocess_images import getLargestRegion
# from feature_extraction import computeDictCodes

import skimage
from skimage.filters.rank import median, noise_filter
from skimage.io import imread
from skimage import segmentation, measure, morphology
from scipy.ndimage.filters import laplace
from skimage.draw import line, set_color

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# from sklearn.cluster import Kmeans

from matplotlib import pyplot as plt
from matplotlib import cm

from time import time
import random
import math

import numpy as np
import pickle
import os 	

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


	

def fixedSOM(img,k,iterations=35):				## Unused
	img = np.ones(img.shape) - img
	h, w = img.shape
	diag = math.sqrt(h*h + w*w)
	old_centroids = np.zeros((k,2))
	old_centroids[:,0] = np.round(np.arange(k)/k * h/2 + h/4)
	old_centroids[:,1] = np.round(np.arange(k)/k * w/2 + w/4)

	for i in range(iterations):
		centroids = np.zeros((k,2))
		norm_const = [0]*k
		cent_dist = [0]*(k-1)
		for c in range(k-1):
			cent_dist[c] = np.exp(-diag*np.linalg.norm(old_centroids[c]-old_centroids[c+1])**2)
		for x in range(h):
			for y in range(w):
				pos = np.array([x,y])
				dist = np.sum((np.tile(pos,(k,1)) - old_centroids)**2,axis=1)
				c_opt = dist.argmin()
				centroids[c_opt,0] += img[x,y]*x
				centroids[c_opt,1] += img[x,y]*y
				norm_const[c_opt] += img[x,y]
				if c_opt > 0:
					centroids[c_opt-1,0] += img[x,y]*x * cent_dist[c_opt-1]
					centroids[c_opt-1,1] += img[x,y]*y * cent_dist[c_opt-1]
					norm_const[c_opt-1] += img[x,y] * cent_dist[c_opt-1]
				if c_opt < k-1:
					centroids[c_opt+1,0] += img[x,y]*x * cent_dist[c_opt]
					centroids[c_opt+1,1] += img[x,y]*y * cent_dist[c_opt]
					norm_const[c_opt+1] += img[x,y] * cent_dist[c_opt]

		for c in range(k):
			if norm_const[c] > 0:
				centroids[c,:] = np.round(centroids[c,:]/norm_const[c])
			else:
				centroids[c,0] = np.round(random.randrange(h))
				centroids[c,1] = np.round(random.randrange(w))
		old_centroids = centroids
	return centroids.astype(int)				

def growSOM(img,k,iterations=100,alpha=0.5):	## Unfinished + unused
	img = np.ones(img.shape) - img
	h, w = img.shape
	moments = skimage.measure.moments(img, order=1)
	init_x = moments[0,1] / moments[0,0]
	init_y = moments[1,0] / moments[0,0]
	centroids = [np.array([init_x,init_y])]
	adj_lst = [[]]
	p = np.zeros(2)
	l = len(centroids)
	while l < k:
		# for x in range(w):
		l = len(centroids)

		for i in range(iterations):
			p[0] = random.randrange(h)
			p[1] = random.randrange(w)
			min_dist = 10000000
			for c in range(l):
				dist = np.linalg.norm(centroids[i]-p)
				if dist < min_dist:
					winner = c
					min_dist = dist
			centroids[winner] += alpha/i * (p - centroids[winner])
			for neighbour in adj_lst[winner]:
				n_dist = np.linalg.norm(centroids[winner] - centroids[neighbour])
				centroids[neighbour] += np.exp(-n_dist) * alpha/i * (p - centroids[neighbour]) 

def KMeans(img,k,iterations=40):				## Unused
	img = np.ones(img.shape) - img
	h, w = img.shape
	old_centroids = np.zeros((k,2))
	old_centroids[:,0] = random.sample(range(h),k)
	old_centroids[:,1] = random.sample(range(w),k)
	for _ in range(iterations):
		centroids = np.zeros((k,2))
		norm_const = [0]*k
		for x in range(h):
			for y in range(w):
				pos = np.array([x,y])
				dist = np.sum((np.tile(pos,(k,1)) - old_centroids)**2,axis=1)
				c_opt = dist.argmin()
				centroids[c_opt,0] += img[x,y]*x
				centroids[c_opt,1] += img[x,y]*y
				norm_const[c_opt] += img[x,y]
		for c in range(k):
			if norm_const[c] > 0:
				centroids[c,:] = np.round(centroids[c,:]/norm_const[c])
			else:
				centroids[c,0] = np.round(random.randrange(h))
				centroids[c,1] = np.round(random.randrange(w))
		old_centroids = centroids
	return centroids.astype(int)


num_bins = 10

def massAroundCentroids(img, centroids):		## Unused
	k = centroids.shape[0]
	(h, w) = img.shape
	diag = h**2 + w**2
	mass = [0.0001]*num_bins
	for x in range(h):
		for y in range(w):
			dist = float("inf")
			for c in range(k):
				cx, cy = centroids[c,:]
				dist = min(dist, (cx-x)**2 + (cy-y)**2 )
			idx = min(math.floor(k * num_bins * dist / diag ), num_bins-1 )
			mass[idx] += 1-img[x,y]
	return mass

def massAroundTree(img, segments):				## Unused
	k = segments.shape[0]
	(h, w) = img.shape
	diag = h**2 + w**2
	mass = [0.0001]*num_bins
	for x in range(h):
		for y in range(w):
			p = np.array([x,y])
			dist = float("inf")
			for c in range(k):
				s0 = segments[c,0,:]
				s1 = segments[c,1,:]
				dist = min(dist, pointSegmentDistance(p, s0,s1))
			idx = min(math.floor(2 * math.sqrt(k) * num_bins * (dist/diag)**(0.5) ), num_bins-1 )
			mass[idx] += 1-img[x,y]
	return mass

def putCross(img,pos):
	(h,w,_) = img.shape
	color = [1,1,0]
	img[pos[0],pos[1],:] = color
	if pos[0] > 0:
		img[pos[0]-1,pos[1],:] = color
	if pos[0] < h:
		img[pos[0]+1,pos[1],:] = color
	if pos[1] > 0:
		img[pos[0],pos[1]-1,:] = color
	if pos[1] < w:
		img[pos[0],pos[1]+1,:] = color
	return img

def pointSegmentDistance(p,s0,s1):				## Unused
	s = s1-s0
	p = p-s0
	s_len = np.linalg.norm(s)
	s /= s_len
	s_orth = np.array([-s[1],s[0]])
	t = np.dot(p,s)
	u = np.dot(p,s_orth)
	if t < 0:
		return u**2 + t**2
	elif t > s_len:
		return u**2 + (t-s_len)**2
	else:
		return u**2

def euclideanMST(points):						## Unused
	(n, d) = points.shape
	adj_matrix = np.zeros((n,n))
	for i in range(1,n):
		for j in range(0,i):
			dist = np.linalg.norm(points[i,:]-points[j,:])
			adj_matrix[i,j] = dist
			adj_matrix[j,i] = dist
	mst = minimum_spanning_tree(adj_matrix)
	return np.array(mst.nonzero()).astype(int)

def invertImage(img):
	return np.ones(img.shape) - img

now = time()
print(now)
random.seed(1426119383.2767782)

# path_train = "../data/train_feature_extraction/cropped"
path_train = "../data/train"
classes = random.sample(os.listdir(path_train),2)
# n = 3
k = 3
# file_names = random.sample(os.listdir(path_train),2*n)
# file_names += random.sample(os.listdir(path_train + classes[1]),n)
# i = 0
x1 = []
x2 = []
x3 = []
x4 = []
for img_name in os.listdir(path_train + "/" + classes[0]):
	img =  imread(path_train + "/" + classes[0] + "/" + img_name, as_grey=True)/255
	img_laplace = laplace(img)
	img_laplace -= np.min(img_laplace)
	img_laplace /= np.max(img_laplace)
	# print(np.unique(img_laplace),flush=True)
	centroids = swigSOM(invertImage(invertImage(img)),k)
	img_dist = swigDistToTree(centroids,img.shape)
	values = np.repeat(img_dist.flatten(), (img.flatten()*51).astype(int) )
	values_laplace = np.repeat(img_dist.flatten(), (img_laplace.flatten()*50).astype(int) )
	x1 += [np.percentile(values,30)]
	x2 += [np.percentile(values_laplace,30)]
	x3 += [np.percentile(values,70)]
	x4 += [np.percentile(values_laplace,70)]
sub1 = plt.subplot(2,2,3)
plt.imshow(img,cmap=cm.gray)
y1 = []
y2 = []
y3 = []
y4 = []
for img_name in os.listdir(path_train + "/" + classes[1]):
	img =  imread(path_train + "/" + classes[1] + "/" + img_name, as_grey=True)/255
	img_laplace = laplace(img)
	img_laplace -= np.min(img_laplace)
	img_laplace /= np.max(img_laplace)
	centroids = swigSOM(invertImage(invertImage(img)),k)
	img_dist = swigDistToTree(centroids,img.shape)
	values = np.repeat(img_dist.flatten(), (img.flatten()*51).astype(int) )
	values_laplace = np.repeat(img_dist.flatten(), (img_laplace.flatten()*50).astype(int) )
	y1 += [np.percentile(values,30)]
	y2 += [np.percentile(values_laplace,30)]
	y3 += [np.percentile(values,70)]
	y4 += [np.percentile(values_laplace,70)]
sub1 = plt.subplot(2,2,4)
plt.imshow(img,cmap=cm.gray)

sub1 = plt.subplot(2,2,1)
plt.scatter(x1,x2, c=[1,0,0])
plt.scatter(y1,y2, c=[0,0,1])

sub2 = plt.subplot(2,2,2)
plt.scatter(x3,x4, c=[1,0,0])
plt.scatter(y3,y4, c=[0,0,1])

plt.show()



# for file_name in file_names:
# 	img =  imread(path_train + "/" + file_name)/255
# 	SOMstart = time()
# 	centroids = swigSOM(invertImage(invertImage(img)),k)
# 	# print(centroids)
# 	# print("Time for SOM: " + str(time() - SOMstart), flush=True)
# 	otherStart = time()
# 	img_rgb = skimage.color.gray2rgb(img)
# 	som_len = 0
# 	for c in range(k):
# 		putCross(img_rgb,centroids[c,:])
# 		# print(c)
# 		# print(centroids[c,:])
# 	for c in range(k-1):
# 		p0 = centroids[c,:]
# 		p1 = centroids[c+1,:]

# 		som_len += np.linalg.norm(p0-p1)

# 		rr, cc = line(p0[0],p0[1],p1[0],p1[1])
# 		set_color(img_rgb,(rr, cc), [(1+c)/k,1-(1+c)/k,0])

# 	sub1 = plt.subplot(2,2*n,2*i+1)
# 	sub1.set_title(file_name)
# 	plt.imshow(img_rgb, interpolation='none')
# 	img_dist = swigDistToTree(centroids,img.shape);
# 	hist = np.zeros(num_bins)
# 	def histIndex(x):
# 		return min(math.floor(2 * math.sqrt(k) * num_bins * x), num_bins-1 )
# 	vHistIndex = np.vectorize(histIndex);
# 	binAreas = vHistIndex(img_dist)
# 	for b in range(num_bins):
# 		bin_area = (binAreas==b)
# 		hist[b] += np.sum(bin_area*invertImage(img))
# 	diag = np.linalg.norm(np.array(img.shape))
# 	color_value = math.sqrt(som_len/((k-1)*diag))
# 	# print("Time for other stuff: " + str(time() - otherStart), flush=True)
# 	sub2 = plt.subplot(2,2*n,2*i+2)
# 	# plt.imshow(invertImage((img_dist**2)*invertImage(img)),interpolation='none',cmap=cm.gray)
# 	sub2.set_title(file_name)
# 	plt.bar(range(num_bins),hist,color=cm.Greys(color_value))

# 	i += 1
# print(time()-now)
# plt.show()
