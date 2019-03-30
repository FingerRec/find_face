#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-30 17:53
     # @Author  : Awiny
     # @Site    :
     # @Project : FindFace
     # @File    : dl_main.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import cv2
from face_detect import FaceDetection
from sift import SiftFeatureExtract
from utils import img_crop, draw_box, img_normalize, img_resize,  img_mask
from matplotlib import pyplot as plt
from PIL import Image
import face_recognition

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input1', help='Path to input image 1.', default='test/imgs_part_trump/trump_1.jpg')
parser.add_argument('--input2', help='Path to input image 2.', default='test/imgs_part_trump/G20_1.jpg')
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
    _, boxes = faceDetector.face_detection_api(trump_face_path)
    for box in boxes:
        print("trump face: ",box.xmin, box.ymin, box.xmax, box.ymax)

    trump_face = img_mask(cv2.imread(trump_face_path), box)
    cv2.imwrite('output/cropped_faces/1.png', trump_face)
    trump_face = face_recognition.load_image_file('output/cropped_faces/1.png')
    trump_face_encoding = face_recognition.face_encodings(trump_face)[0]
    mp_detected_img, boxes = faceDetector.face_detection_api(multi_person_path)
    mutil_person_img = cv2.imread(multi_person_path)
    best_box = None
    best_distance = 999999
    nearset_face = None
    #may need do resize and normalization ?
    for box in boxes:
        if box.xmax - box.xmin < 20 or box.ymax - box.ymin < 20 or box.ymin < 10 or box.xmin < 10:
            continue
        elif box.xmax < 0 or box.xmin < 0 or box.ymax < 0 or box.ymin < 0:
            continue
        print("x:{}-{}, y:{}-{}".format(box.xmin, box.xmax, box.ymin, box.ymax))
        single_face = img_mask(mutil_person_img, box)
        cv2.imwrite('output/cropped_faces/2.png',single_face)
        single_face = face_recognition.load_image_file('output/cropped_faces/2.png')
        #single_face = face_recognition.api.load_image_file(single_face)
        single_face_encoding = face_recognition.face_encodings(single_face)[0]
        match_distance = face_recognition.face_distance([trump_face_encoding], single_face_encoding)
        print("match_distance is: ", match_distance)
        if match_distance < best_distance:
            best_distance = match_distance
            best_box = box
            nearset_face = single_face
    print("best_box face: ", best_box.xmin, best_box.ymin, best_box.xmax, best_box.ymax)
    detect_img = draw_box(mutil_person_img.copy(), best_box)
    return detect_img, nearset_face, mp_detected_img

def main():
    trump_face_path = args.input1
    multi_person_path = args.input2
    box_img, nearset_face, mp_detected_img= find_trump(trump_face_path, multi_person_path)
    plt.subplot(221)
    plt.title('person to be find')
    plt.imshow(Image.open(trump_face_path).convert('RGB'))
    plt.subplot(222)
    plt.title('yolo face detect result')
    plt.imshow(mp_detected_img[:,:,(2,1,0)])
    plt.subplot(223)
    plt.title('mask img(nearest face)')
    plt.imshow(nearset_face[:,:,(2,1,0)])
    plt.subplot(224)
    plt.title('Trump is here!')
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