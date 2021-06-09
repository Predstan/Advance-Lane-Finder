import cv2
import numpy as np
from utils import *
from camera_utils import *

class EdgeFinder:
    def __init__(self, image, img):
        self.image = image
        self.col = undistort(img)
        self._hlsmin = 116
        self._hlsmax  = 255

        self._magmin = 67
        self.magmax = 255

        self.dirmin = 63
        self.dirmax = 255

        self.kernel = 3


        self.hsvmin = 200
        self.hsvmax = 215
           

        def onchangehlsmin(pos):
            self._hlsmin = pos
            self._render()

        def onchangehlsmax(pos):
            self._hlsmax = pos
            self._render()

        def onchangemagmin(pos):
            self._magmin = pos
            self._render()
            
        def onchangemagmax(pos):
            self.magmax = pos
            self._render()
            
        def onchangedirmin(pos):
            self.dirmin = pos
            self._render()
            
        def onchangekernel(pos):
            self.kernel = pos
            self._render()
            
            
        def onchangedirmax(pos):
            self.dirmax = pos
            self._render()


        def onchangehsvmin(pos):
            self.hsvmin = pos
            self._render()
            
            
        def onchangehsvmax(pos):
            self.hsvmax = pos
            self._render()

        cv2.namedWindow('edges')

        cv2.createTrackbar('hlsmin', 'edges', self._hlsmin, 255, onchangehlsmin)
        cv2.createTrackbar('hlsmax', 'edges', self._hlsmax, 255, onchangehlsmax)
        cv2.createTrackbar('magmin', 'edges', self._magmin, 255, onchangemagmin)
        cv2.createTrackbar('magmax', 'edges', self.magmax, 255, onchangemagmax)
        cv2.createTrackbar('dirmin', 'edges', self.dirmin, 255, onchangedirmin)
        cv2.createTrackbar('dirmax', 'edges', self.dirmax, 255, onchangedirmax)
        cv2.createTrackbar('kernel', 'edges', self.kernel, 9, onchangekernel)
        cv2.createTrackbar('hsvmin', 'edges', self.hsvmin, 255, onchangehsvmin)
        cv2.createTrackbar('hsvmax', 'edges', self.hsvmax, 255, onchangehsvmax)

        self._render()

        print ("Adjust the parameters as desired.  Hit any key to close.")


        cv2.waitKey(0)

        cv2.destroyWindow('edges')
        cv2.destroyWindow('smoothed')


    def _render(self):
        left_line = Line(1)
        right_line = Line(1)
        hls = hls_select(self.col, thresh=(self._hlsmin, self._hlsmax))
        mag = mag_thresh(self.col, mag_thresh = (self._magmin, self.magmax))
        dire = sob_threshold(self.col, thresh = (self.dirmin, self.dirmax))
        hsv = hsv_select(self.col, (self.hsvmin, self.hsvmax))
        com, v = comb(self.col, hls, dir, mag, hsv, True)
        img, M, MinV = perspective(com)
        p = img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fit_polynomial(img, left_line, right_line, verbose=1)
        im = draw_onto_the_road(self.col, MinV, left_line, right_line)
      
        for ve in v:
            im = cv2.circle(im, (ve[0],ve[1]), radius=1, color=(0, 0, 0), thickness=10)
        cv2.imshow('hls', hls*255 )
        cv2.imshow('mag', mag*255)
        cv2.imshow('direction', dire*255)
        cv2.imshow('Combined', com)
        cv2.imshow('Final', im)
        cv2.imshow('pers', p)




        
def comb(img, hls, sob, mag, hsv, region = True):


    
    yellows = yellow(img)
    whites = white(img)

    combined = np.zeros_like(hls)
    combined[((hls==1) | (mag == 1)) | (sob==1) | (whites==1) | (hsv ==1)] = 1

    img = combined*255
    vertices =  np.array([[100, img.shape[0]], 
                     [350, img.shape[0]*0.6], 
                     [600, img.shape[0]*0.6], 
                     [1100, img.shape[0]]], np.int32)

    if region:
        vertices = np.array([[50, img.shape[0]], 
                     [700, img.shape[0]*0.7], 
                     [320, img.shape[0]*0.7], 
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

    return np.dstack([img, img, img]), vertices
    
