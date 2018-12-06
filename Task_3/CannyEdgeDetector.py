import numpy as np
import pylab as plt
from scipy.ndimage import convolve
#
# NO MORE MODULES ALLOWED
#

# todo - implement a gaussian filter
# input:	img_in - input image 	[2-D image]
#	  	ksize - kernel size	[integer]
#		sigma - sigma		[float]
# return: gaussian fitlered image	[2-D image]
def gaussFilter(img_in,ksize,sigma):
    return 0

# todo - implement sobel filtering
# input:	img_in - input image					[2-D image]
# return: 	gx,gy - sobel filtered images in x- and y-direction 	[2-D image,2-D image]
def sobel(img_in):
    return 0,0

# todo - calculate gradient magnitude and direction images
# input:	gx,gy			[2-D image,2-D image]
# return: 	gradient,direction 	[2-D image,2-D image]
def gradientAndDirection(gx,gy):
    return 0,0

# todo - calculate maximum suppression
# input:	g,theta		[2-D image,2-D image]
# return: 	max_sup		[2-D image]
def maxSuppress(g, theta):    
    return 0

# todo - calculate hysteris thresholding
# input:	g		[2-D image]
#		t_low,t_high	[integer,integer]
# return: 	hysteris	[2-D image]
def hysteris(max_sup, t_low, t_high):
    return 0

def canny(img):
    #gaussian
    gauss = gaussFilter(img,5 ,2)

    #sobel
    gx,gy = sobel(gauss)
    
    #plotting
    plt.subplot(1,2,1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()
    
    #gradient directions
    g, theta = gradientAndDirection(gx, gy)
    
    #plotting
    plt.subplot(1,2,1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()
    #maximum suppression
    maxS_img = maxSuppress(g,theta)
    
    #plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()
    
    result = hysteris(maxS_img, 50, 75)
    
    return result
