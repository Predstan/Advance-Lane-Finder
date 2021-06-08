import argparse
from utils import *
from camera_utils import *
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2



parser = argparse.ArgumentParser(description='Start processing Script')


parser.add_argument('--image',
                    type=int,
                    default=1,
                    help='Process Image or video frame, Default: 1')

parser.add_argument('--saveDirectory',
                    type=str,
                    default="save",
                    help='Directory for saving, Default: save')

parser.add_argument('--readDirectory',
                    type=str,
                    default="None",
                    help='Directory to Read from')

parser.add_argument('--verbose',
                    type=bool,
                    default=False,
                    help='show image process')


                    




args = parser.parse_args()
image = bool(args.image)
save = str(args.saveDirectory)
read = str(args.readDirectory)
verbose = bool(args.verbose)

right_line = Line(200)
left_line = Line(200)
def process_image(image):
    global right_line, left_line
    und = undistort(image)
    thresh = combined_thresh(image, region=True)
    pers, M, MinV=perspective(thresh)
    img = cv2.cvtColor(pers, cv2.COLOR_RGB2GRAY)
    h = fit_polynomial(img, left_line, right_line, verbose=1)
    return draw_on_road(und, MinV, left_line, right_line, thresh, h[0])

def main():
    if image:
        img  = mpimg.imread(read)
        img = process_image(img)
        if verbose:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("this",img)
            cv2.waitKey(0)


    else:
        import cv2
        cap = cv2.VideoCapture(read)
        i = 0
        while True:
            if cap.grab():
                flag, frame = cap.retrieve()
                if not flag:
                    continue
                else:
                    frames = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if i % 100 == 0:
                        mpimg.imsave(f"atlas/{save}{i}.jpg", frames)
                    i+= 1
                    frame = process_image(frames)
                    if verbose:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cv2.imshow('video', frame)
                    

            if cv2.waitKey(10) == 27:
                break

            
            

        #
        
        #mpimg.imsave(f"{save}.jpg", img)



main()

