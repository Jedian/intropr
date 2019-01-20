import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from scipy import ndimage 
def preProcessing(img):
    
    # scale image to 320 x 240 px
    img = cv2.resize(img, (320, 240))
    
    # Apply a binary threshold (115 = cutOff) and a gaussian filter
    th, dst = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)

    dst = cv2.GaussianBlur(dst, (5,5), 0)

    # find contours, get the largest and draw it
    dst, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    contour = max(contours, key=cv2.contourArea)
    #cv2.drawContours(dst, contour, -1, 255, 2)
    #cv2.imshow("Countour", img)
    #cv2.waitKey(0)
    
    ### Trace the boundary of the holes between fingers.

    ONC = 115
    # get 115 samples of the contour to work on
    x,y = contour.T
    x = x.tolist()[0]
    y = y.tolist()[0]
    tck, u = splprep([x,y], u=None, s=1.0)
    u_new = np.linspace(u.min(), u.max(), ONC)
    x_new, y_new = splev(u_new, tck, der=0)
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
    contour = np.array(res_array).reshape((-1,1,2)).astype(np.int32)
   
    ### Calculate the center of gravity of the holes and decide the key points | k1, k2, and k3.
    key_pts = []
    i = 1

    print len(contour)
    while i < ONC:
        while i<ONC and contour[i][0][0] > contour[i-1][0][0]:
            i += 1
        if contour[i][0][0] < 150 and contour[i][0][0] > 10:
            key_pts.append(contour[i][0])
        while i<ONC and contour[i][0][0] <= contour[i-1][0][0]:
            i += 1

    contour = np.array(key_pts).reshape((-1,1,2)).astype(np.int32)
    ### Line up k1 and k3 to get the Y-axis of the palmprint coordination system and
    ### then  make  a  line  through  k2  and  perpendicular  to  Y-axis  to  determine  the
    ### origin of the palmprint coordination system.
    
    #get closest point to line
    k1k3 = [key_pts[2][0] - key_pts[0][0], key_pts[2][1] - key_pts[0][1]]
    k1k2 = [key_pts[1][0] - key_pts[0][0], key_pts[1][1] - key_pts[0][1]]
    mg = k1k3[0]**2 + k1k3[1]**2

    dt = (k1k2[0]*k1k3[0] + k1k2[1]*k1k3[1])/(mg+0.0)  
    orig = [key_pts[0][0] + k1k3[0]*dt, key_pts[0][1] + k1k3[1]*dt]

    ### Rotate the image to place the Y-axis on the vertical direction
    angle = np.arctan2(orig[1] - key_pts[1][1], orig[0] - key_pts[1][0])
    row,col = img.shape
    center = tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle*180/np.pi, 1.0)
    img = cv2.warpAffine(img, rot_mat, (col,row))

    return img
