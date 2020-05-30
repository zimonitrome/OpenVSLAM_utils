import numpy as np
import cv2 as cv
import glob

# Width and height of intersection points on chess board.
w = 7
h = 7

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(h,5,0)
objp = np.zeros((h*w,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (w,h), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (w,h), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(1)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv.CALIB_RATIONAL_MODEL)

# Print camera properties in format of OpenVSLAM config file.
print(ret, mtx, dist, rvecs, tvecs)
print(f"Camera.fx: {float(mtx[0][0])}")
print(f"Camera.fy: {float(mtx[1][1])}")
print(f"Camera.cx: {float(mtx[0][2])}")
print(f"Camera.cy: {float(mtx[1][2])}")
print("")
print(f"Camera.k1: {float(dist[0][0])}")
print(f"Camera.k2: {float(dist[0][1])}")
print(f"Camera.p1: {float(dist[0][2])}")
print(f"Camera.p2: {float(dist[0][3])}")
print(f"Camera.k3: {float(dist[0][4])}")
print(f"Camera.k4: {float(dist[0][5])}")