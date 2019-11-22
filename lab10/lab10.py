import cv2
import numpy as np
from matplotlib import pyplot as plt

# Vars
nrows = 3
ncols = 2
val = 0

#read in image
image = cv2.imread('building2.jpg')

#convert images to rgb, grayscale etc...
imgOrig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
harris = imgOrig.copy()
corners = cv2.goodFeaturesToTrack(imgGray,200,0.01,10)
shiTomasi = imgOrig.copy()
imgSift = imgOrig.copy()

print("Looping over Image")
threshold = 0.05;
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold * dst.max()):
            cv2.circle(harris,(j,i), 3,(255, 0, 0), -1)

for i in corners:
    x,y = i.ravel()
    cv2.circle(shiTomasi,(x, y), 3,(255, 0, 0), -1)

#SIFT
sift = cv2.xfeatures2d.SIFT_create(50)
(kps, descs) = sift.detectAndCompute(imgGray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))


imgSift = cv2.drawKeypoints(imgSift, kps, outImage=None, color=(255, 0, 0), flags=4)

print("Plotting Images")
# Plot images
plt.subplot(nrows, ncols, 1),plt.imshow(imgOrig, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 2),plt.imshow(imgGray, cmap = 'gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 3),plt.imshow(dst, cmap = 'gray')
plt.title('DST'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 4),plt.imshow(harris, cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 5),plt.imshow(shiTomasi, cmap = 'gray')
plt.title('Tomasi'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols, 6),plt.imshow(imgSift, cmap = 'gray')
plt.title('Sift'), plt.xticks([]), plt.yticks([])
plt.show()

################################################################################

# Advanced Exercises
print("Comparing image with Brute Force Matcher")
#Brute Force Matcher
imgBf1 = cv2.imread('GMIT1.jpg')
imgCompare1 = cv2.cvtColor(imgBf1, cv2.COLOR_BGR2RGB)

imgBf2 = cv2.imread('GMIT2.jpg')
imgCompare2 = cv2.cvtColor(imgBf2, cv2.COLOR_BGR2RGB)

# Init SIFT Detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(imgCompare1, None)
kp2, des2 = orb.detectAndCompute(imgCompare2, None)

# create BFMatcher obj
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#match descriptors
matches = bf.match(des1, des2)

# sort in order of distance
matches = sorted(matches, key = lambda x:x.distance)

# draw first 10 matches
imgBfOut = cv2.drawMatches(imgCompare1, kp1, imgCompare2, kp2,
                    matches[:10], None, flags=2)

# End of Brute Force Matcher
print("Comparing image with FLANN Matcher")
#FLANN Matcher
imgFlan1 = imgCompare1.copy()
imgFlan2 = imgCompare2.copy()

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(imgFlan1,None)
kp2, des2 = sift.detectAndCompute(imgFlan2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange
                        (len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

imgFlannOut = cv2.drawMatchesKnn(imgFlan1,kp1,imgFlan2,kp2,matches,None,**draw_params)

# Plot Matchers
print("Plotting Images")
#Brute Force
plt.subplot(2, 1, 1),
plt.imshow(imgBfOut)
plt.title('Brute Force'), plt.xticks([]), plt.yticks([])

#FLANN Mactcher
plt.subplot(2, 1, 2),
plt.imshow(imgFlannOut),
plt.title('Flann Matcher'), plt.xticks([]), plt.yticks([])
plt.show()
