import cv2
import numpy as np
import json, codecs
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

addStatus = False
def nothing(x):
    pass
def createWindows():
    cv2.namedWindow('Calibration')
    cv2.createTrackbar('Add Frame','Calibration',0,1,nothing)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*7,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)
cam = cv2.VideoCapture(0)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
data = {}
data['cameracoeffs'] = []
frames = []
gray = None
createWindows()
while(True):
    _, frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,(9,7),None)
    img = frame
    print(addStatus)
    if cv2.getTrackbarPos('Add Frame','Calibration') == 1:
        cv2.setTrackbarPos('Add Frame','Calibration',0)
        frames.append(gray)
        print('added frame')
        addStatus = False
        
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img = cv2.drawChessboardCorners(frame,(9,7),corners2,ret)
        cv2.imshow("Calibration",img)
    else:
        cv2.imshow("Calibration",img)
    if len(frames) != 0:
        print(len(frames))
        for frame_idx in range(len(frames)):
            name = str(frame_idx)
            cv2.imshow(name,frames[frame_idx])
    if cv2.waitKey(1) & 0xFF == ord('q'):
            gray = frames[len(frames)-1]
            break

for frame in frames:
    cv2.imshow('frames',frame)
    ret, corners = cv2.findChessboardCorners(frame,(9,7),None)
    print(ret)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(frame,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

retu, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
#nRet = retu.tolist()
#nMtx = mtx.tolist()
#nDist = dist.tolist()
#nRvecs = rvecs.tolist()
#nTvecs = nTvecs.tolist()
data['cameracoeffs'].append({'ret':retu,
                              'mtx':mtx,
                              'dist':dist,
                              'rvecs': rvecs,
                              'tvecs': tvecs
})

with open('camera.json','w') as fs:
    json.dump(data,fs,cls = NumpyEncoder)
cam.release()
cv2.destroyAllWindows()


