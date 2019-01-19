import numpy as np
import cv2
from scipy.interpolate import splprep, splev
def preProcessing(img):
    
    # scale image to 320 x 240 px
    img = cv2.resize(img, (320, 240))
    
    # Apply a binary threshold (115 = cutOff) and a gaussian filter
    th, dst = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)
    #cv2.imshow("THreshold", dst)
    #cv2.waitKey(0)

    dst = cv2.GaussianBlur(dst, (5,5), 0)
    #cv2.imshow("blur", img)
    #cv2.waitKey(0)

    # find contours, get the largest and draw it
    dst, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(dst, contour, -1, 127, 2)
    #cv2.imshow("Countour", img)
    #cv2.waitKey(0)
    
    ### Trace the boundary of the holes between fingers.

    ONC = 225

    x,y = contour.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = splprep([x,y], u=None, s=1.0)
    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    u_new = np.linspace(u.min(), u.max(), ONC)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = splev(u_new, tck, der=0)
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
    #contour = [[160, 120], [161, 121]]
    contour = np.array(res_array).reshape((-1,1,2)).astype(np.int32)
   
    ### Calculate the center of gravity of the holes and decide the key points | k1, k2, and k3.

    new_cont = []
    i = 1

    print len(contour)
    while i < ONC:
        while i<ONC and contour[i][0][0] > contour[i-1][0][0]:
            i += 1
            print i
        #if contour[i][0][0] > 200 and contour[i][0][0] < 320:
        new_cont.append(contour[i])
        print contour[i]
        i+=10
        while i<ONC and contour[i][0][0] <= contour[i-1][0][0]:
            i += 1
            print i

    contour = np.array(new_cont).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(dst, contour, -1, 255, 5)
    cv2.imshow("Countour", img)
    cv2.waitKey(0)
    ### Line up k1 and k3 to get the Y-axis of the palmprint coordination system and
    ### then  make  a  line  through  k2  and  perpendicular  to  Y-axis  to  determine  the
    ### origin of the palmprint coordination system.






    ### Rotate the image to place the Y-axis on the vertical direction






    return img
