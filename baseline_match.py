#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:15:41 2018

@author: jsaavedr
"""

import argparse
import numpy as np
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
    kp_im, des_im = sift.compute(image, kp_im)
    
    kp_q = sift.detect(query_image)
    kp_q, des_q = sift.compute(query_image, kp_q)
    print(des_im)
    print(des_q)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_im, des_q, k = 2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    ##apply homography
    if len(good)>4:    
        src_pts = np.float32([ kp_im[m.queryIdx].pt for m in good ]).reshape(-1,2)
        dst_pts = np.float32([ kp_q[m.trainIdx].pt for m in good ]).reshape(-1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        matchesMask = mask.ravel().tolist()
    else:
        matchesMask = None

    # cv2.drawMatchesKnn expects list of lists as matches.
    match_image = cv2.drawMatches(image, kp_im, query_image, kp_q,  good, None, flags=2, matchesMask=matchesMask)

    cv2.imshow("matches", match_image)
    cv2.waitKey()
    
