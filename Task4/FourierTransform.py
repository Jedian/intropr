import numpy as np
import cv2
from matplotlib import pyplot as plt

def calcuateFourierParameters(img, k, sampling_steps):
    #TODO: calculate the fourier parameters
    
    #fft + magnitude
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(1+np.abs(fshift))

    plt.imshow(magnitude, cmap='gray')
    plt.show()

    #polar coordinate transformation
    def polar(r, theta):
        #calculation returning x,y in cartesian coordinates

        return 0,0
    
    #energy calculations
    
    
    # ring like
    R = np.zeros(k, float)






    #fan like
    Theta_i = np.zeros(k, float)






    return R, Theta_i
