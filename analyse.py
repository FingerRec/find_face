#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-02 17:25
     # @Author  : Awiny
     # @Site    :
     # @Project : FindFace
     # @File    : analyse.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import cv2
from utils import img_crop, img_normalize, img_resize, img_mask
import time
import numpy as np
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
#==============================================================================================================
#This script analyse different operator performance in face match, divided into 4 steps
#1.Use yolo detect faces and consturct two subdir, one include trump imgs, another include others
#2.using different method and get average acc and average time
#3.plot different method compare figure
#==============================================================================================================
import matlab.engine
import face_recognition
from sift import SiftFeatureExtract

eng = matlab.engine.start_matlab()

def removeMacDsStore(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os.remove(item)

def dir_face_crop(dirA, dirB):
    """
    give an input dir, detect all faces and clip them and saved into a dir
    :return:
    """
    from face_detect import FaceDetection
    begin_time = time.time()
    if not os.path.exists(dirB):
        os.makedirs(dirB)
    faceDetector = FaceDetection()
    count = 0
    img_count = 0
    all_num = len(os.listdir(dirA))
    for img in os.listdir(dirA):
        try:
            _, boxes = faceDetector.face_detection_api(os.path.join(dirA, img))
        except cv2.error as e:
            print(e)
            continue
        img_copy = cv2.imread(os.path.join(dirA, img)).copy()
        if img_copy.size == 0:
            continue
        for box in boxes:
            try:
                face = img_crop(img_copy, box, scale=25)
                cv2.imwrite(dirB+'/_{}.jpg'.format(count), face)
                count += 1
            except cv2.error as e:
                print(e)
                break
        img_count += 1
        if img_count % 100 == 0:
            print("have processed {}/{} imgs!".format(img_count, all_num))
    end_time = time.time()
    print("process {} imgs, cost {} seconds".format(count, end_time-begin_time))
    return count


def feature_match(detect_method, descriptor_method, imga, imgb):
    if detect_method=='deep_metric_learning' and descriptor_method == 'deep_metric_learning':
        face_a = cv2.imread(imga)
        face_b = cv2.imread(imgb)
        encoding_a = face_recognition.face_encodings(face_a)[0]
        encoding_b = face_recognition.face_encodings(face_b)[0]
        if face_recognition.compare_faces([encoding_a], encoding_b)[0]:
            matched_point, res = 0, 1
        else:
            matched_point, res = 0, 0
    elif detect_method=='sift' or detect_method=='pca_sift' or detect_method=='modifed_sift':
        face_1 = cv2.imread(imga)
        face_1 = img_normalize(face_1)
        face_1 = img_resize(face_1, (153, 200))
        face_2 = cv2.imread(imgb)
        face_2 = img_normalize(face_2)
        face_2 = img_resize(face_2, (153, 200))
        key_point_1, trump_sift = siftFeatureExtract.calculate_sift_feature(face_1)
        key_point_2, single_face_sift = siftFeatureExtract.calculate_sift_feature(face_2)
        try:
            match_score, good = siftFeatureExtract.calculate_similarity(trump_sift, single_face_sift, key_point_1, key_point_2)
        except cv2.error as e:
            print(cv2.error)
            match_score = 0
            good = None
        if match_score > 1:
            matched_point = match_score
            res = 1
        else:
            matched_point = 0
            res = 0
    else:
        matched_point, res = eng.match_img(detect_method, descriptor_method, imga, imgb, nargout=2)
        with open('test/mean_matched_point.txt', 'r') as f:
            dict = json.load(f)
        if matched_point > dict[detect_method + '+' + descriptor_method][5] / 2:
            res = 1
    return matched_point, res

def analyse(detect_method, descriptor_method, samples_dir, postive_dir, negative_dir):
    """
    given a method, calculate it's average accuracy and average time, return a dict, in format {'method':[]} []:average_acc/average_time
    input should contain 3 dir, 1: test imgs 2:postive ns 3:negative ns
    matlab method will be called here
    :return:
    """
    print('*****'*15)
    print("begin processDT:{}/DM:{}".format(detect_method, descriptor_method))
    begin_time = time.time()
    samples_num = len(os.listdir(samples_dir))
    np_num = len(os.listdir(postive_dir))
    ns_num = len(os.listdir(negative_dir))
    pairs_num = samples_num * np_num + samples_num * ns_num
    pair_count = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    avg_point = 0
    for img in os.listdir(samples_dir):
        if '.jpg' not in img:
            continue
        for np in os.listdir(postive_dir):
            if '.jpg' not in np:
                continue
            pair_count += 1
            matched_point, res = feature_match(detect_method, descriptor_method, os.path.join(samples_dir,img), os.path.join(postive_dir,np))
            avg_point += matched_point
            if res:
                TP += 1
            else:
                FN += 1
        for nn in os.listdir(negative_dir):
            if '.jpg' not in nn:
                continue
            pair_count += 1
            matched_point, res = feature_match(detect_method, descriptor_method, os.path.join(samples_dir,img), os.path.join(negative_dir,nn))
            avg_point += matched_point
            if res:
                FP += 1
            else:
                TN += 1
        print("{}/{} pair imgs have processed. wait a moment...".format(pair_count, pairs_num))
    print("DT:{}/DM:{}/TP:{}/FP:{}/FN:{}/TN:{}".format(detect_method, descriptor_method,TP,FP,FN,TN))
    avg_point /= pair_count
    P = TP + FN
    N = FP + TN
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    try:
        PPV = TP/(TP+FP)
    except ZeroDivisionError:
        PPV = 0
    ACC = (TP+TN)/(P+N)
    end_time = time.time()
    return TPR, FPR, PPV, ACC, (end_time - begin_time)/pairs_num, avg_point

def plot(dict):
    """
    given dict, plot img
    :return:
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for key,values in dict.items():
        print(key, values)
        ax.scatter(values[4], values[3], s=200, label = key, alpha=0.3, edgecolors='none')
    for key,values in dict.items():
        ax.annotate(key,(values[4], values[3]))
    ax.set(xlabel='time', ylabel='acc')
    plt.show()

def modifyed_sift():
    return

def main():
    #cropped faces
    dir = 'test/trump'
    out_crop_dir = 'test/croped_faces'
    all_img_num = dir_face_crop(dir, out_crop_dir)
    # test for metric_learning
    #dict = {}
    with open('test/main_result.txt', 'r') as f:
        dict = json.load(f)
    dict.pop('deep_metric_learning' + '+' + 'deep_metric_learning')
    big_s_dir = 'test/big_ns'
    big_p_dir = 'test/big_np'
    big_n_dir = 'test/big_nn'
    TPR, FPR, PPV, ACC, average_time, average_point = analyse('deep_metric_learning', 'deep_metric_learning', big_s_dir,
                                                              big_p_dir, big_n_dir)
    dict['deep_metric_learning' + '+' + 'deep_metric_learning'] = [TPR, FPR, PPV, ACC, average_time, average_point]
    s_dir = 'test/ns'
    p_dir = 'test/np'
    n_dir = 'test/nn'
    #use matlab to detect these methods
    detect_method = ['FAST', 'ME', 'Corner', 'SURF', 'KAZE', 'BRISK', 'MSER']
    descriptor_method = ['HOG', 'SURF', 'KAZE', 'FREAK', 'BRISK']
    for m in detect_method:
        for n in descriptor_method:
            TPR, FPR, PPV, ACC, average_time, average_point = analyse(m, n, s_dir, p_dir, n_dir)
            dict[m+'+'+n] = [TPR, FPR, PPV, ACC, average_time, average_point]
    global siftFeatureExtract
    #test for sift
    siftFeatureExtract = SiftFeatureExtract(feature='SIFT')
    TPR, FPR, PPV, ACC, average_time, average_point = analyse('sift', 'sift', s_dir, p_dir, n_dir)
    dict['sift' + '+' + 'sift'] = [TPR, FPR, PPV, ACC, average_time, average_point]
    #test for PCA_sift
    siftFeatureExtract = SiftFeatureExtract(feature='PCA_SIFT', match_method_name ='flann')
    TPR, FPR, PPV, ACC, average_time, average_point = analyse('pca_sift', 'pca_sift', s_dir, p_dir, n_dir)
    dict['pca_sift' + '+' + 'pca_sift'] = [TPR, FPR, PPV, ACC, average_time, average_point]
    #test for modifed_sift
    siftFeatureExtract = SiftFeatureExtract(feature='SIFT', match_method_name ='flann')
    TPR, FPR, PPV, ACC, average_time, average_point = analyse('modifed_sift', 'modifed_sift', s_dir, p_dir, n_dir)
    dict['modifed_sift' + '+' + 'modifed_sift'] = [TPR, FPR, PPV, ACC, average_time, average_point]
    with open('test/main_result.txt', 'w') as f:
        f.write(json.dumps(dict))
    plot(dict)

if __name__ == '__main__':
    main()