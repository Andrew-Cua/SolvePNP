import cv2
import numpy as np
import math
class SquareTargetFinder(object):

    TARGET_WIDTH = 2
    TARGET_HEIGHT = 2
    TARGET_STRIP_WIDTH = 2
    def __init__(self, cameraMtx, disMtx):
        self.cameraMatrix = cameraMtx
        self.distortionMatrix = disMtx
        self.upperHSV = np.array([100,255,255])
        self.lowerHSV = np.array([70,100,130])
        self.contourList = []
        self.dimensionList = []
        self.target_contour = None
        self.target_coords = np.array([[-SquareTargetFinder.TARGET_WIDTH/2.0,  SquareTargetFinder.TARGET_HEIGHT/2.0, 0.0],
                                          [-SquareTargetFinder.TARGET_WIDTH/2.0, -SquareTargetFinder.TARGET_HEIGHT/2.0, 0.0],
                                          [ SquareTargetFinder.TARGET_WIDTH/2.0, -SquareTargetFinder.TARGET_HEIGHT/2.0, 0.0],
                                          [ SquareTargetFinder.TARGET_WIDTH/2.0,  SquareTargetFinder.TARGET_HEIGHT/2.0, 0.0]])
        self.distance = 0
        self.tilt_angle = math.radians(0)
    def hsvThreshold(self,frame):
        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        self.adjustedFrame = cv2.inRange(hsv_frame,self.lowerHSV,self.upperHSV)
        return self.adjustedFrame
    
    def __findContours__(self,frame):
        _,thresh_img = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
        self.contours, hierarchy = cv2.findContours(thresh_img,1,2)
        return self.contours,0

    def processImage(self,frame):
        self.contourList = []
        self.dimensionList = []
        contours,_ = self.__findContours__(frame)
        self.target_contour = None
        shape = frame.shape
        for cnt in contours:
            #cnt = np.float32(cnt)
            perimeter = cv2.arcLength(cnt,True)
            self.epsilon = 0.01*cv2.arcLength(cnt,True)
            self.approx = cv2.approxPolyDP(cnt,self.epsilon,True)
            area = cv2.contourArea(cnt)
            print(area)
            #if the contour isnt a square, dont need
            if len(self.approx) != 4:
                continue
            if area < 100 or area > 150_000:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            aspect = w/h
            if aspect < 0.9 or aspect > 1.5:
                continue
            center, widths = SquareTargetFinder.__contourCenterWidth__(cnt)
            rArea = widths[0]*widths[1]
            self.dimensionList.append({'contour':cnt,'center':center,'widths':widths,'area':rArea})
            self.contourList.append(cnt)
        if self.contourList is not None:
            self.dimensionList.sort(key=lambda c: c['area'], reverse=True)
            for candidate_idx in range(min(3,len(self.dimensionList))):
                self.target_contour = self.__testContour__(self.dimensionList,candidate_idx)
                if self.target_contour is not None:
                    break
        if self.target_contour is not None:
            cnrlist = []
            for cnr in self.target_contour:
                cnrlist.append((float(cnr[0][0]), float(cnr[0][1])))
            self.sort_corners(cnrlist)
            image_corners = np.array(cnrlist)
            retval, rvec, tvec = cv2.solvePnP(self.target_coords, image_corners,
                                              self.cameraMatrix, self.distortionMatrix)
            if retval:
                distance,angle1,angle2 = self.compute_output_values(rvec,tvec)
                self.distance = distance
                print("here")
                return self.contourList, self.distance
        return None,0
            


    def __contourCenterWidth__(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (x + int(w / 2), y + int(h / 2)), (w, h)        


    def __testContour__(self,contour_list, cand_idx):
        candidate = contour_list[cand_idx]

        cand_x = candidate['center'][0]
        cand_width = candidate['widths'][0]
        cand_height = candidate['widths'][0]

        hull = cv2.convexHull(candidate['contour'])
        target_contour = self.__quadfit__(hull,0.06)
        if len(target_contour) == 4:
            return target_contour
        return None

    def __quadfit__(self,contour,approx_dp_error):
        peri = cv2.arcLength(contour,True)
        return cv2.approxPolyDP(contour,approx_dp_error*peri,True)


    def sort_corners(self,cnrlist):
        '''Sort a list of 4 corners so that it goes in a known order. Does it in place!!'''
        cnrlist.sort()

        # now, swap the pairs to make sure in proper Y order
        if cnrlist[0][1] > cnrlist[1][1]:
            cnrlist[0], cnrlist[1] = cnrlist[1], cnrlist[0]
        if cnrlist[2][1] < cnrlist[3][1]:
            cnrlist[2], cnrlist[3] = cnrlist[3], cnrlist[2]
        return

    def compute_output_values(self, rvec, tvec):
        '''Compute the necessary output distance and angles'''

        # The tilt angle only affects the distance and angle1 calcs

        x = tvec[0][0]
        z = math.sin(self.tilt_angle) * tvec[1][0] + math.cos(self.tilt_angle) * tvec[2][0]

        # distance in the horizontal plane between camera and target
        distance = math.sqrt(x**2 + z**2)

        # horizontal angle between camera center line and target
        angle1 = math.atan2(x, z)

        rot, _ = cv2.Rodrigues(rvec)
        rot_inv = rot.transpose()
        pzero_world = np.matmul(rot_inv, -tvec)
        angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

        return distance, angle1, angle2