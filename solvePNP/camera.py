import cv2 
import numpy as np

class Camera(object):
    def __init__(self,usbPort):
        self.port = usbPort
        self.cap = cv2.VideoCapture(self.port)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -1.0)
    
    def grabFrame(self):
        _, self.frame = self.cap.read() 
        return self.frame