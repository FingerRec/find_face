#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-25 20:06
     # @Author  : Awiny
     # @Site    :
     # @Project : FindFace
     # @File    : sift.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import cv2
import numpy as np
from matplotlib import pyplot as plt

class SiftFeatureExtract(object):
    def __init__(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        #self.feature = 'SIFT'
        self.feature = 'SURF'
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=100)
        search_params = dict(checks=50)  # or pass empty dictionary
        #self.match_method = cv2.FlannBasedMatcher(index_params, search_params)
        self.match_method = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        self.match_method_name = 'flann'
        #self.match_method = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #self.match_method_name = 'bf'
        self.draw_params = None

    def calculate_sift_feature(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.feature == 'SIFT':
            feature = cv2.xfeatures2d.SIFT_create()
        elif self.feature == 'SURF':
            feature = cv2.xfeatures2d.SURF_create(400)
        else:
            raise TypeError('not support type now!')
        kp, des = feature.detectAndCompute(gray, None)
        return kp, des

    def calculate_similarity(self, sift1, sift2):
        if self.match_method_name == 'bf':
            matches = self.match_method.knnMatch(sift1, sift2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            return len(good), good
        else:
            # Apply ratio test
            matches = self.match_method.knnMatch(sift1, sift2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            count = 0
            trainIdx = list()
            queryIdx = list()
            imgIdx = list()
            #some problem here, most points point to one same point
            for i, (m, n) in enumerate(matches):
                #print(trainIdx)
                trainIdx.append(m.trainIdx)
                queryIdx.append(m.queryIdx)
                imgIdx.append(m.imgIdx)
                if trainIdx.count(m.trainIdx) > 6:
                    continue
                if queryIdx.count(m.queryIdx) > 6:
                    continue
                if imgIdx.count(m.imgIdx) > 6:
                    continue
               # if m.distance < 0.8 * n.distance:
                matchesMask[i] = [1, 0]
                count += 1
            self.draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            return count, matches

    def draw_match(self, img1, img2, kp1, kp2, good):
        img3 = img1.copy()
        if self.match_method_name == 'bf':
            match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3)
        else:
            match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3, **self.draw_params)
        return match_img


def sift_test():
    img = cv2.imread('test/home.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)

    img=cv2.drawKeypoints(gray,kp, img)
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.imwrite('test/sift_keypoints.jpg',img)
    '''
    img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('test/sift_keypoints.jpg',img)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray,None)
    '''
#Here kp will be a list of keypoints and des is a numpy array of shape Number\_of\_Keypoints \times 128.

def sift_bf_matcher():
    img1 = cv2.imread('test/imgs_part_trump/trump_1.jpg', 0)  # queryImage
    img2 = cv2.imread('test/imgs_part_trump/multi_0.jpg', 0)  # trainImage
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = img1.copy()
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img3)
    plt.imshow(img3), plt.show()

def flann():
    img1 = cv2.imread('test/imgs_part_trump/trump_2.jpg', 0)  # queryImage
    img2 = cv2.imread('test/imgs_part_trump/multi_0.jpg', 0)  # trainImage
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

def main():
    flann()
    #sift_bf_matcher()
    #sift_test()

if __name__ == '__main__':
    main()