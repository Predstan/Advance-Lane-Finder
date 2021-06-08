import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import dill
import glob






with open("dist_param",'rb') as f:
    dis_param = dill.load(f)
    
mtx = dis_param['mtx']
dist = dis_param['dist']




def undistort(img, mtx=mtx, dist=dist):
    return cv2.undistort(img, mtx, dist, None, mtx)



def perspective(img,):
    # Undistort Image
    undist = undistort(img)

    
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

 
    h, w = img.shape[:2]

    src = np.float32([[560,460],[180,690],[1130,690],[750,460]])
    
    
    dst = np.float32([[320,0],[320,720],[960,720],[960,0]]) 
    
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])
    # For source points I'm grabbing the outer four detected corners


 
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    
    MinV = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M, MinV


def hls_select(img, thresh=(109, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    return binary_output


def sob_threshold(img, sobel_kernel=3, thresh = (67, 255)):
                

    thresh_min, thresh_max = thresh
    

    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    
    return binary_output
                
                
def mag_thresh(img, sobel_kernel=3, mag_thresh=(63, 255)):
        
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh_min, thresh_max = mag_thresh
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))

    sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    mag = np.sqrt(sobelx**2 + sobely**2)
    
    scaled_sobel = np.uint8(255*mag/np.max(mag))
    
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

def yellow(img):

    mask = cv2.inRange(img, np.array([150, 150, 150]), np.array([255, 255, 255]))
    binary = (mask // 255).astype(np.uint8)
    return binary


def white(img):
    mask = cv2.inRange(img, np.array([185, 185, 185]), np.array([255, 255, 255]))
    binary = (mask // 255).astype(np.uint8)
    return binary


def hsv_select(img, hsv_s_treshold = (200, 215)):
    hsv_channel = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]
    binary = np.zeros_like(hsv_channel)
    binary[(hsv_channel >= hsv_s_treshold[0]) & (hsv_channel <= hsv_s_treshold[1])] = 1
    return binary




def combined_thresh(img, region = True):


    hls = hls_select(img)
    sob = sob_threshold(img)
    mag = mag_thresh(img)
    yellows = yellow(img)
    whites = white(img)
    hsv = hsv_select(img)

    combined = np.zeros_like(hls)
    combined[((hls==1) | (mag == 1)) | (sob==1) ] = 1

    img = combined*255

    if region:
        vertices = np.array([[130, img.shape[0]], 
                     [250, img.shape[0]*0.6], 
                     [800, img.shape[0]*0.6], 
                     [1200, img.shape[0]]], np.int32)

        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        ignore_mask_color = (255,)

        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, 
                    np.int32([vertices]), 
                    ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        img = cv2.bitwise_and(img, mask)

    return np.dstack([img, img, img])