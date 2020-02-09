import json
import numpy as np
import cv2
import math
from camera import Camera
from squareTargetFinder import SquareTargetFinder

data = {}
cam = Camera(0)
finder = None
ret = mtx = dist = rvecs = tvecs = None 
if __name__ == '__main__':
    with open('camera.json') as fs:
        data = json.load(fs)
        for data in data['cameracoeffs']:
            ret   = data['ret']
            mtx   = np.asarray(data['mtx'])
            dist  = np.asarray(data['dist'])
            rvecs = np.asarray(data['rvecs'])
            tvecs = np.asarray(data['tvecs'])
    finder = SquareTargetFinder(mtx,dist)
    while(True):
        frame = cam.grabFrame()
        frame = cv2.medianBlur(frame,5)
        hsv_thresh = finder.hsvThreshold(frame)
        contours,distance = finder.processImage(frame)
        if contours is not None:
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        print(distance)    
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break