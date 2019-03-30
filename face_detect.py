#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-25 20:05
     # @Author  : Awiny
     # @Site    :
     # @Project : FindFace
     # @File    : face_detect.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import cv2
import tensorflow as tf
import numpy as np
import os
import argparse
import json
from utils import get_yolo_boxes, makedirs, preprocess_input, draw_boxes, write2txt
from tqdm import tqdm

pretrained_model = "pretrained_model/yolo_v3_face_detect.pb"
net_h = 416
net_w = 416
obj_thresh = 0.5
nms_thresh = 0.5


class FaceDetection(object):
    def __init__(self):
        self.graph, self.sess = load_face_detect_model(pretrained_model)

    def face_detection_api(self, img_path):
        """
        input: a image with face
        :return: image with bounding box
        """
        detected_img, bounding_boxes = frame_face_detect(self.graph, self.sess, net_h, net_w, obj_thresh, nms_thresh,
                                         img_path)
        return detected_img, bounding_boxes

def load_face_detect_model(pretrained_model):
    '''
    load pretrained yolo model from pt file.
    :return:
    '''
    with open(pretrained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    sess = tf.Session(graph=graph)
    return graph, sess

def video_face_detect(graph, sess, net_h, net_w, obj_thresh, nms_thresh):
    '''

    :return:
    '''
    anchors = [55, 69, 75, 234, 133, 240, 136, 129, 142, 363, 203, 290, 228, 184, 285, 359, 341, 260]
    x = graph.get_tensor_by_name('prefix/input_1:0')
    y0 = graph.get_tensor_by_name('prefix/k2tfout_0:0')
    y1 = graph.get_tensor_by_name('prefix/k2tfout_1:0')
    y2 = graph.get_tensor_by_name('prefix/k2tfout_2:0')
    video_reader = cv2.VideoCapture(0)
    while True:
        ret_val, img = video_reader.read()
        if ret_val == True:
            img = cv2.resize(img, (640, 480))
            img_h, img_w, _ = img.shape
            batch_input = preprocess_input(img, net_h, net_w)

        inputs = np.zeros((1, net_h, net_w, 3), dtype='float32')
        inputs[0] = batch_input
        net_output = sess.run([y0, y1, y2], feed_dict={x: inputs})
        batch_boxes = get_yolo_boxes(net_output, img_h, img_w, net_h, net_w, anchors, obj_thresh, nms_thresh)
        _, _, facecen = draw_boxes(img, batch_boxes[0], ['face'], obj_thresh)
        # print(facecen)
        cv2.imshow('image with bboxes', img)
        yield facecen
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()

def frame_face_detect(graph, sess, net_h, net_w,  obj_thresh, nms_thresh, image_path):
    """

    :return:
    """
    anchors = [55, 69, 75, 234, 133, 240, 136, 129, 142, 363, 203, 290, 228, 184, 285, 359, 341, 260]
    x = graph.get_tensor_by_name('prefix/input_1:0')
    y0 = graph.get_tensor_by_name('prefix/k2tfout_0:0')
    y1 = graph.get_tensor_by_name('prefix/k2tfout_1:0')
    y2 = graph.get_tensor_by_name('prefix/k2tfout_2:0')

    img = cv2.imread(image_path)
    cv2.resize(img, (640, 480))
    img_h, img_w, _ = img.shape
    batch_input = preprocess_input(img, net_h, net_w)  # 416x416x3

    inputs = np.zeros((1, net_h, net_w, 3), dtype='float32')
    inputs[0] = batch_input
    net_output = sess.run([y0, y1, y2], feed_dict={x: inputs})  # output=1x13x13x18

    batch_boxes = get_yolo_boxes(net_output, img_h, img_w, net_h, net_w, anchors, obj_thresh, nms_thresh)
    _, _, facecen, save_box = draw_boxes(img, batch_boxes[0], ['face'], obj_thresh)
    detected_img = img
    return detected_img, save_box


def main():
    graph, sess = load_face_detect_model(pretrained_model)
    image_path = 'test/trump/google_0000.jpg'
    count = 0
    for image in os.listdir('test/trump/'):
        detected_img, _ = frame_face_detect(graph, sess, net_h, net_w, obj_thresh, nms_thresh, 'test/trump/'+image)
        #cv2.imshow('', detected_img)
        cv2.imwrite('output/img_{}.png'.format(count), detected_img)
        count += 1
        print('have processed {} images'.format(count))
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print('have process {} imgs'.format(count))

if __name__ == '__main__':
    main()