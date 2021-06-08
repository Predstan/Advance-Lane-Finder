import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dill   

# Read this image folder into a list 
cal_cam = glob.glob("camera_cal/*")


def collect_point(image, grid = [9, 6], display=False):
    """
    Helper function to collect grids and points from image
    
    args:
        image: RGB/BGR image to be used
        grid:  row and col
        display: True or False
    returns:
        Display points on image if display is true else
        Ruturn Corners or real points and image points
    """
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    row, col = tuple(grid)
    
    objp = np.zeros((row * col, 3), np.float32)
    objp[:,:2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    
    ret, corners = cv2.findChessboardCorners(gray, (row, col), None)
    
    if ret:
        if display:
            plt.imshow(cv2.drawChessboardCorners(img, (row, col), corners, ret))
        else:
            return True, corners, objp
        
    return False, 0, 1

imgpoints = []
objpoints = []

for path in cal_cam:
    img  = mpimg.imread(path)
    print(path)
    ret, corners, objp = collect_point(img)
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Collect the Calibration contant values that will be used for this image caliberation
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        

#Save Caliberation values 

dis_param = {"mtx" : mtx, "dist": dist}
import dill
with open("dist_param",'wb') as f:
    dill.dump(dis_param, f )
