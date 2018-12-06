import numpy as np
#
# NO MORE MODULES ALLOWED
#

#input -  	[2-D image]
#output -  	[2-D image]
#Implement Otsus thresholding

def otsu(image):
    histogram = np.histogram(image, bins=256,range = [0,256])[0]
    npixels = image.shape[0]*image.shape[1] + 0.0
    p0 = 0.0
    m0 = 0.0
    wtotal = 0.0
    for i in range(0, 256):
        wtotal = wtotal + i * (histogram[i].item()/npixels)
    maximum = (0, -1)
    for i in range(0, 256):
        p0 = p0 + histogram[i].item()/npixels
        p1 = 1 - p0
        if p0 == 0 or p1 == 0:
            continue
        m0 = m0 + i * (histogram[i].item()/npixels)
        m1 = wtotal - m0
        sigma = p0*p1*(((m1/p1)-(m0/p0))**2)

        if sigma > maximum[0]:
            maximum = (sigma, i)

    nimg = image.copy()
    for r in range(0, len(image)):
        for v in range(0, len(image[0])):
            if image[r][v] > maximum[1]:
                nimg[r][v] = 255
            else:
                nimg[r][v] = 0

    return nimg

