#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-25 20:05
     # @Author  : Awiny
     # @Site    :
     # @Project : FindFace
     # @File    : main.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import cv2
from face_detect import FaceDetection
from sift import SiftFeatureExtract
from utils import img_crop, draw_box, img_normalize, img_resize
from matplotlib import pyplot as plt
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trump', type=str, default='test/imgs_part_trump/trump_1.jpg', help='a person img')
parser.add_argument('--scene', type=str, default='test/imgs_part_trump/multi_1.jpg', help='multiple person imgs')
args = parser.parse_args()

# ====================================================================================================================================
# Step1: detect the first image with one person and find faces, process and save sift feature
# Step2: detect the second image with multi person and find faces, process and save sift features
# Step3: find the simliarist face and drwa it in origin image
# ====================================================================================================================================
def find_trump(trump_face_path, multi_person_path):
    """
    find trump in a image and plot bounding box
    :param trump_face_path:
    :param multi_person_path:
    :return:
    """
    faceDetector = FaceDetection()
    siftFeatureExtract = SiftFeatureExtract()
    _, boxes = faceDetector.face_detection_api(trump_face_path)
    for box in boxes:
        print("trump face: ",box.xmin, box.ymin, box.xmax, box.ymax)
    trump_face = img_crop(cv2.imread(trump_face_path), box)
    trump_face = img_normalize(trump_face)
    trump_face = img_resize(trump_face, (360, 240))
    key_point_1, trump_sift = siftFeatureExtract.calculate_sift_feature(trump_face)
    _, boxes = faceDetector.face_detection_api(multi_person_path)
    mutil_person_img = cv2.imread(multi_person_path)
    best_score = 0
    best_box = None
    best_good = None
    nearset_face = None
    #may need do resize and normalization ?
    for box in boxes:
        if box.xmax - box.xmin < 20 or box.ymax - box.ymin < 20 or box.ymin < 10 or box.xmin < 10:
            continue
        elif box.xmax < 0 or box.xmin < 0 or box.ymax < 0 or box.ymin < 0:
            continue
        print("x:{}-{}, y:{}-{}".format(box.xmin, box.xmax, box.ymin, box.ymax))
        single_face = img_crop(mutil_person_img, box)
        single_face = img_normalize(single_face)
        single_face = img_resize(single_face, (360, 240))
        key_point_2, single_face_sift = siftFeatureExtract.calculate_sift_feature(single_face)
        match_score, good = siftFeatureExtract.calculate_similarity(trump_sift, single_face_sift)
        print("match_scroe is: ", match_score)
        if match_score > best_score:
            best_score = match_score
            best_box = box
            best_good = good
            nearset_face = single_face
    print("best_box face: ", best_box.xmin, best_box.ymin, best_box.xmax, best_box.ymax)
    match_img = siftFeatureExtract.draw_match(trump_face, nearset_face, key_point_1, key_point_2, best_good)
    detect_img = draw_box(mutil_person_img.copy(), best_box)
    return detect_img, match_img

def main():
    trump_face_path = args.trump
    multi_person_path = args.scene
    box_img, match_img = find_trump(trump_face_path, multi_person_path)
    plt.subplot(221)
    plt.imshow(Image.open(trump_face_path).convert('RGB'))
    plt.subplot(222)
    plt.imshow(Image.open(multi_person_path).convert('RGB'))
    plt.subplot(223)
    plt.imshow(match_img[:,:,(2,1,0)])
    plt.subplot(224)
    plt.imshow(box_img[:,:,(2,1,0)])
    plt.show()
    '''
    cv2.imshow('box img', box_img)
    while True:
        if cv2.waitKey(1) == 27:
            break
    '''

if __name__ == '__main__':
    main()