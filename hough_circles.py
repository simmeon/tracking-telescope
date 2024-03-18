# Getting features of amateur star photos
# 11/03/24

import cv2 as cv
import numpy as np
import requests
import imutils

url = 'http://192.168.68.52:8080/shot.jpg'

# IMAGES

# img = cv.imread("img/stars2.webp", cv.IMREAD_GRAYSCALE)

# # Initiate FAST object with default values
# fast = cv.FastFeatureDetector_create()

# # Find and draw the keypoints
# kp = fast.detect(img, None)
# img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255))

# cv.imshow("Image", img2)

# k = cv.waitKey(0)

# cv.destroyAllWindows()



# VIDEOS

# Create a VideoCapture object and read from input file 
#cap = cv.VideoCapture('vid/jupiter.mp4')


  


# Read until video is completed 
while True: 
      
    # Capture frame-by-frame 
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    img = cv.imdecode(img_arr, -1) 
    frame = imutils.resize(img, width=1000, height=1800) 

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8, param1=50, param2=30, minRadius=1, maxRadius=50)
    
    cv.imshow('Frame', frame)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255), 3)


        # Display the resulting frame 
        cv.imshow('Frame', frame) 
            
    # Press esc on keyboard to exit 
    if cv.waitKey(1) == 27: 
        break
  

  
# Closes all the frames 
cv.destroyAllWindows() 


