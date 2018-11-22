#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:15:41 2018

@author: jsaavedr
"""

import argparse
import cv2

if __name__ == "__main__" :
    parser = argparse.ArgumentParser("example on local descriptors")
    parser.add_argument("-image", type = str, required = True)
    parser.add_argument("-query", type = str, required = True)
    _input = parser.parse_args()
    image_f = _input.image
    query_f = _input.query
    image = cv2.imread(image_f, cv2.IMREAD_GRAYSCALE)
    query_image = cv2.imread(query_f, cv2.IMREAD_GRAYSCALE)
    #    
    #create sift desscritpor#
    sift = cv2.xfeatures2d.SIFT_create()
    kp_im = sift.detect(image)
    kp_qu = sift.detect(query_image)
    #kp, des = sift.compute(gray, kp)    
    image = cv2.drawKeypoints(image, kp_im, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    query_image = cv2.drawKeypoints(query_image, kp_im, query_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("kp_image", image)
    cv2.imshow("kp_query", query_image)
    cv2.waitKey()
    
    