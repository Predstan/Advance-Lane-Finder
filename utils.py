import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import dill
import glob







class Line():
    def __init__(self, n_window):
        # was the line detected in the last iteration?
        self.n_window = n_window

        self.detected = False  
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = np.mean(self.recent_xfitted, axis=0) if len(self.recent_xfitted) !=0 else []    
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.mean(self.current_fit, axis=0) if len(self.current_fit) !=0 else []   
        

    def update(self, Detected, recent_xfitted, current_fit, x_values, y_values, radius, distance,):

        self.detected = Detected
        self.allx=x_values
        self.ally=y_values
        if len(self.current_fit) > 1: self.diff = self.current_fit[-1] - current_fit 
        if len(self.recent_xfitted) == self.n_window: self.recent_xfitted.pop(0)
        self.recent_xfitted.append(recent_xfitted)
        if len(self.current_fit) == self.n_window: self.current_fit.pop(0)
        self.current_fit.append(current_fit)
        self.best_fit = np.mean(self.current_fit, axis=0)
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.radius_of_curvature = radius
        self.line_base_pos = distance






    

# left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
# right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 15
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, left_line, right_line, verbose=0):
    
    #global left_fit, right_fit, ploty
    # Find our lane pixels first

    if len(left_line.current_fit) % left_line.n_window ==0:
        
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    else:
        leftx, lefty, rightx, righty, out_img = search_previous(binary_warped, left_line.best_fit, right_line.best_fit, verbose=verbose)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    


    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # Fit a second order polynomial to each using `np.polyfit`
        left_current_fit = left_fit
        left_Detected = True
        left_recent_xfitted = left_fitx
    

    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_current_fit = left_line.best_fit
        left_Detected = False
        left_recent_xfitted = left_line.best_fit

    try:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Fit a second order polynomial to each using `np.polyfit`
        right_current_fit = right_fit
        right_Detected = True
        right_recent_xfitted = right_fitx

    except TypeError:
        print("Error")
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        right_current_fit = right_line.best_fit
        right_Detected = False
        right_recent_xfitted = left_line.best_fit

        
    if verbose:
        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')

        
    left_line.allx, left_line.ally, right_line.allx, right_line.ally = leftx, lefty, rightx, righty,
    l_c, r_c, radius, offset = measure_curvature_pixels(ploty, left_fit, right_fit, left_line, right_line)

    left_line.update( left_Detected, left_recent_xfitted, left_current_fit, leftx, lefty, radius=l_c, distance=offset )
    right_line.update( right_Detected, right_recent_xfitted, right_current_fit, rightx, righty, radius=r_c, distance=offset)
    
    return out_img, left_fit, right_fit, ploty, l_c, r_c

def search_previous(binary_warped, left_fit, right_fit, verbose):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 60

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

     # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)


    

    # Fit new polynomials

    if verbose:
        left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        ## Visualization ##
    
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plot the polynomial lines onto the image
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
    
    return leftx, lefty, rightx, righty, result


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    #Fit a second order polynomial to each with np.polyfit() ###
    #global left_fit, right_fit, ploty
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    #Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty



def measure_curvature_pixels(ploty, left_fit, right_fit, left_line, right_line):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
  
    

    y_per_pix = 30 / 720   # meters per pixel in y dimension
    x_per_pix = 3.7 / 700  # meters per pixel in x dimension
    
    leftx, lefty, rightx, righty = left_line.allx, left_line.ally, right_line.allx, right_line.ally

    left_fit = np.polyfit(lefty*y_per_pix, leftx*x_per_pix, 2)
    right_fit = np.polyfit(righty*y_per_pix, rightx*x_per_pix, 2)

    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2 * left_fit[0]*y_per_pix + left_fit[1]) **2 ) **1.5)/np.abs(2 * left_fit[0]) ## Implement the calculation of the left line here
    right_curverad = ((1 + (2 * right_fit[0]*y_per_pix + right_fit[1]) **2 ) **1.5)/np.abs(2 * right_fit[0])  ## Implement the calculation of the right line here


    left = left_fit[0]*(720*y_per_pix)**2 + left_fit[1]*720*y_per_pix + left_fit[2]
    right = right_fit[0]*(720*y_per_pix)**2 + right_fit[1]*720*y_per_pix + right_fit[2]

    # line_lt_bottom = np.mean(left_line.allx[left_line.ally > 0.95 * left_line.ally.max()])
    # line_rt_bottom = np.mean(right_line.allx[left_line.ally > 0.95 * left_line.ally.max()])
    # lane_width = line_rt_bottom - line_lt_bottom
    # midpoint = 1280 / 2
    # offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
    # offset_meter = x_per_pix * offset_pix
    offset_meter = np.abs(640*x_per_pix - np.mean([left, right]))
    radius = np.mean([left_curverad, right_curverad])

    return left_curverad, right_curverad, radius, offset_meter
    




def draw_on_road(img_undistorted, Minv, left_line, right_line, binary, pers_image):
    """
    Draw both the drivable lane area and the detected lane-lines onto the original (undistorted) frame.

    Args:
        img_undistorted: original undistorted color frame
        Minv: (inverse) perspective transform matrix used to re-project on original frame
        left_line: left lane-line previously detected
        rigt_line: right lane-line previously detected
    Return:
        image with color Blend
    """
    height, width, _ = img_undistorted.shape
    x, y = 20, 14

    left_fit = left_line.current_fit[-1]
    right_fit = right_line.current_fit[-1]

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))  # Warp back to original image space

    blend = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)

    # now separately draw solid lines to highlight them
    line_warp = np.zeros_like(img_undistorted)
    line_warp = draw(line_warp, color=(255, 0, 0), L= left_line, )
    line_warp = draw(line_warp, color=(0, 0, 255), L=right_line,)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    lines_mask = blend.copy()
    idx = np.any([line_dewarped != 0][0], axis=2)
    lines_mask[idx] = line_dewarped[idx]
    font = cv2.FONT_HERSHEY_SIMPLEX
    blend = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend, beta=0.5, gamma=0.)


    # add binary image
    h, w = int(0.2 * height), int(0.2*width)
    binary = cv2.resize(binary, dsize=(w, h))
    blend[y:h+y, x:x+w, :] = binary

    # add perspective image
    pers_image= cv2.resize(pers_image, dsize=(w, h))
    blend[y:h+y, 2*x+w:2*(x+w), :] = pers_image


    mean_curvature_meter = np.mean([right_line.radius_of_curvature, left_line.radius_of_curvature ])/2
    cv2.putText(blend, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend, 'Offset from center: {:.02f}m'.format(left_line.line_base_pos), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend

def draw(mask, color=(255, 0, 0), L="p", line_width=50):
    """
    Draw the line on a color mask image.
    """
    h, w, c = mask.shape

    plot_y = np.linspace(0, h - 1, h)
    coeffs = L.current_fit[-1] 

    line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
    line_left_side = line_center - line_width // 2
    line_right_side = line_center + line_width // 2

    # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array(list(zip(line_left_side, plot_y)))
    pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
    pts = np.vstack([pts_left, pts_right])

    # Draw the lane onto the warped blank image
    return cv2.fillPoly(mask, [np.int32(pts)], color)


         
                  
                  

    

