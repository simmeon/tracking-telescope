# use ORB features and matching to track motion
# 16/03/24

import cv2 as cv
import numpy as np
import requests
import imutils

url = 'http://192.168.68.52:8080/shot.jpg'

cap = cv.VideoCapture('vid/jupiter.mp4')

# Webcam video
# Capture frame-by-frame 
img_resp = requests.get(url) 
img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
img = cv.imdecode(img_arr, -1) 
frame = imutils.resize(img, width=1000, height=1800) 
frame0 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# ORB detector
orb = cv.ORB_create()

# creat brute force matcher
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# # get first frame from file
# ret, frame0 = cap.read()
# frame0 = frame0[200:1000, :]
# frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

# get ORB keypoints and descriptors for img
kp0, des0 = orb.detectAndCompute(frame0, None)


while True:
    # # get next frame from file
    # ret, frame = cap.read()
    # frame = frame[200:1000, :]
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # get next frame from webcam
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    img = cv.imdecode(img_arr, -1) 
    frame = imutils.resize(img, width=1000, height=1800) 
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    keypress = cv.waitKey(1)

    # set reference frame to current frame on spacebar press
    if keypress == 32:
       kp0, des0 = orb.detectAndCompute(frame, None)
       frame0 = frame
       print('Reference set!')
    
    # get ORB kp and descriptors for current frame
    kp, des = orb.detectAndCompute(frame, None)

    # match ref and current ORB kp
    matches = bf.match(des0, des)

    # sort matches in order of distance
    matches = sorted(matches, key=lambda x:x.distance)

    # draw first 10 matches
    img = cv.drawMatches(frame0,kp0,frame,kp,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    x_sum = 0
    y_sum = 0
    # print match distance
    for match in matches:
        # get points from matches
        pt0 = kp0[match.queryIdx].pt
        pt = kp[match.trainIdx].pt

        # add to sum
        x_sum = x_sum + pt[0] - pt0[0]
        y_sum = y_sum + pt[1] - pt0[1]

    # avg point distance
    if (matches != []):
        pt_avg = (x_sum / len(matches), y_sum / len(matches))
        print(pt_avg) 
    

    # show
    cv.imshow("Frame", img)

    if(keypress == 27):
        break

# Closes all the frames 
cv.destroyAllWindows() 