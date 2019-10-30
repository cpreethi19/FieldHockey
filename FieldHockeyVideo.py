from collections import deque
#from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
#import imutils
import time

cap = cv2.VideoCapture("UConnFockey.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

if cap.isOpened()== False:
    print('Error opening video file')

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break;
    frame_med = cv2.medianBlur(frame, 5)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = fgbg.apply(frame)

    #ret, threshold = cv2.threshold(grayFrame, 240, 255, 0)
    #contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #if len(contours)>0:
    #    cv2.drawContours(threshold, contours, -1, (0, 0, 255), 3)

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 255, 255])
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(frame, lower_white, upper_white)
    #mask = cv2.erode(mask, None, iterations=2)
    #mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours)>0:
        for contour in contours:
            #cv2.drawContours(frame, contour, -1, (255, 0, 0), 3)
            x, y, w, h = cv2.boundingRect(contour)
            #print(cv2.contourArea(contour))
            if (h >= (1.5) * w):
                if (w > 15 and h >= 15):
                    player_img = frame[y:y + h, x:x + w]
                    player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                    mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                    res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                    nzCount = cv2.countNonZero(res1)
                    if (nzCount >= 10):
                        # Mark blue jersy players as france
                        cv2.rectangle(frame_med, (x, y), (x + w, y + h), (255, 0, 0), 3)
            #c = min(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            #M = cv2.moments(c)
            #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius < 2 and radius > 0.5:
            # draw the circle and centroid on the frame,
                cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            #cv2.circle(frame, center, 5, (0, 0, 255), -1)

    #hsv_img[mask > 0] = ([75, 255, 200])
    #different method for extracting foreground

    """res = cv2.bitwise_and(frame, frame, mask=mask)
    res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((13, 13), np.uint8)
    thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #while ret == True:"""
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
