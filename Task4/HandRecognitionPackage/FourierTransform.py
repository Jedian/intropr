import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
def calcuateFourierParameters(img, k, sampling_steps):
    #TODO: calculate the fourier parameters
    
    #fft + magnitude
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20*np.log(1+np.abs(fshift))

    #print magnitude
    #plt.imshow(magnitude, cmap='gray')
    #plt.colorbar()
    #plt.show()

    #polar coordinate transformation
    def polar(r, theta):
        #calculation returning x,y in cartesian coordinates
        return math.floor(120 + r*np.cos(theta)), math.floor(160 + r*np.sin(theta))
    
    #energy calculations
    
    thp = np.linspace(0, np.pi, sampling_steps) 
    # ring like
    R = np.zeros(k, float)
    for i in range(1, k+1):
        for th in thp:
            for r in range((i-1)*k, (k*i)+1):
                x, y = polar(r, th)
                R[i-1] += np.abs(magnitude[int(x)][int(y)])

    #fan like
    Theta_i = np.zeros(k, float)
    for i in range(1, k+1):
        for th in range(i-1, i+1):
            for r in range(120):
                x, y = polar(r, th*np.pi/(k + 0.0))
                Theta_i[i-1] += np.abs(magnitude[int(x)][int(y)])
    
    return R, Theta_i
