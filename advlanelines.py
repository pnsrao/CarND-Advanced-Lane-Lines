#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:57:32 2017

@author: subbu
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from project_utils import *

dist, mtx  = get_dist_mtx()

timages = glob.glob('./test_images/test2.jpg')
ksize = 15
for idx, fname in enumerate(timages):
    img = cv2.imread(fname)
    img_size = img.shape[1::-1]
    dst_img = cv2.undistort(img, mtx, dist, None, mtx)
#    cv2.imshow(fname+'.orig',img)
#    cv2.imshow(fname+'.undistort',dst_img)
#    filename = fname.split('/')[-1]
#    cv2.imwrite('./output_images/w_'+filename,dst)
#    cv2.waitKey(1)
    dir_binary = dir_threshold(dst_img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    result = pipeline(dst_img)
    combined = np.zeros_like(dir_binary)
    combined[(((result[:,:,1]==1) | (result[:,:,2]==1)) & (dir_binary == 1))] = 1 
    #combined[(((result[:,:,1]==1) | (result[:,:,2]==1)))] = 1 

    #src = np.float32([[578, 480],[236, 720],[1124, 720],[755, 480]]) bad
    src = np.float32([[577, 460],[200, 720],[1110, 720],[702, 460]]) #lines 1
    #src = np.float32([[578, 480],[220, 720],[1110, 720],[688, 450]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    persp = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(dst_img)
#    draw_lines(src,ax1)
#    ax1.set_title('Original Image', fontsize=50)
#    ax2.imshow(persp)
#    draw_lines(dst,ax2)
#    ax2.set_title('Perspective Tranform.', fontsize=50)
#    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)    

    left_fit, right_fit, lroc, rroc, d_stdev = findLanes(persp, fname, False)
    ploty = np.linspace(0, persp.shape[0]-1, persp.shape[0] )
    left_fitx = get_xvals(ploty, left_fit)
    right_fitx = get_xvals(ploty, right_fit)
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    veh_offset = xm_per_pix * ((right_fitx[-1]+left_fitx[-1])/2  - persp.shape[1]/2)
    if veh_offset > 0:
        off_string = 'Vehicle is %4.3f m left of center' % (veh_offset)
    else:
        off_string = 'Vehicle is %4.3f m right of center' % (-veh_offset)
    roc = (lroc+rroc)/2
    rocstring = 'Radius of curvature = %5.3f km' % roc
    
    warp_zero = np.zeros_like(persp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(dst_img, 1, newwarp, 0.3, 0)
    result = cv2.putText(result, rocstring, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
    result = cv2.putText(result, off_string, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
    cv2.imshow('image',result) #WaitKey does not work without an image window
    cv2.waitKey(0)
    plt.close('all')
