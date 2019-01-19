import numpy as np
import cv2

def preProcessing(img):
    
    # scale image to 320 x 240 px
    img = cv2.resize(img, (320, 240))
    
    # Apply a binary threshold (115 = cutOff) and a gaussian filter
    th, dst = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    cv2.imshow("THreshold", dst)
    cv2.waitKey(0)

    img = cv2.GaussianBlur(dst, (5,5), 0)
    cv2.imshow("blur", img)
    cv2.waitKey(0)

    # find contours, get the largest and draw it
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contour = max(contours, key=cv2.contourArea)
    #cv2.drawContours(img, contour, -1, 127, 3)
    cv2.imshow("Countour", img)
    cv2.waitKey(0)
    

    ### Trace the boundary of the holes between fingers.
    







    ### Calculate the center of gravity of the holes and decide the key points | k1, k2, and k3.









    ### Line up k1 and k3 to get the Y-axis of the palmprint coordination system and
    ### then  make  a  line  through  k2  and  perpendicular  to  Y-axis  to  determine  the
    ### origin of the palmprint coordination system.






    ### Rotate the image to place the Y-axis on the vertical direction






    return img
