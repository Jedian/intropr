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
    ax = np.arange((-ksize/2) + 1., (ksize/2) + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kern = np.exp(-(xx**2 + yy**2) / (2.*(sigma**2)))/(2*np.pi*(sigma**2))
    kern = kern/np.sum(kern)
    img_output = convolve(img_in, kern).astype(int)
    return img_output
# todo - implement sobel filtering
# input:	img_in - input image					[2-D image]
# return: 	gx,gy - sobel filtered images in x- and y-direction 	[2-D image,2-D image]
def sobel(img_in):
    kerngx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kerngy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    imgx = convolve(img_in, kerngx).astype(int)
    imgy = convolve(img_in, kerngy).astype(int)
    return imgx, imgy

# todo - calculate gradient magnitude and direction images
# input:	gx,gy			[2-D image,2-D image]
# return: 	gradient,direction 	[2-D image,2-D image]
def gradientAndDirection(gx,gy):
    gradient = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    return gradient, direction

# todo - calculate maximum suppression
# input:	g,theta		[2-D image,2-D image]
# return: 	max_sup		[2-D image]
def maxSuppress(g, theta):    
    max_sup = np.zeros(shape=g.shape)
    for x in range(1, g.shape[0]-1):
        for y in range(1, g.shape[1]-1):
            #1 Convert angle to degrees and make it be in range [0,180]
            converted_angle = theta[x][y]*(180/np.pi)
            if converted_angle < 0:
                converted_angle += 180
            #2 Round up angle to nearest neighbor [0, 45, 90, 135] 
            if converted_angle <= 22:
                converted_angle = 0
            elif converted_angle <= 67:
                converted_angle = 45
            elif converted_angle <= 112:
                converted_angle = 90
            elif converted_angle <= 157:
                converted_angle = 135
            else:
                converted_angle = 0
            #3 Store only if its local maximal
            if converted_angle == 0:
                max_sup[x][y] = max(g[x][y], g[x][y+1], g[x][y-1])
            elif converted_angle == 45:
                max_sup[x][y] = max(g[x][y], g[x-1][y-1], g[x+1][y+1])
            elif converted_angle == 90:
                max_sup[x][y] = max(g[x][y], g[x-1][y], g[x+1][y])
            else:
                max_sup[x][y] = max(g[x][y], g[x-1][y+1], g[x+1][y-1])

            if max_sup[x][y] != g[x][y]:
                max_sup[x][y] = 0

    return max_sup

# todo - calculate hysteris thresholding
# input:	g		[2-D image]
#		t_low,t_high	[integer,integer]
# return: 	hysteris	[2-D image]
def hysteris(max_sup, t_low, t_high):
    threshimg = np.zeros(shape=max_sup.shape)
    for x in range(0, max_sup.shape[0]):
        for y in range(0, max_sup.shape[1]):
            if max_sup[x][y] <= t_low:
                threshimg[x][y] = 0
            elif max_sup[x][y] <= t_high:
                threshimg[x][y] = 1
            else:
                threshimg[x][y] = 2

    for x in range(0, max_sup.shape[0]):
        for y in range(0, max_sup.shape[1]):
            if threshimg[x][y] == 2:
                threshimg[x][y] = 255
                for xx in [-1, 0, 1]:
                    for yy in [-1, 0, 1]:
                        if xx+x >= 0 and xx+x < max_sup.shape[0]:
                            if yy+y >= 0 and yy+y < max_sup.shape[0]:
                                if threshimg[xx+x][yy+y] >= 1:
                                    threshimg[xx+x][yy+y] = 255
            
    return threshimg

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
