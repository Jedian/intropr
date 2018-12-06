'''
Created on 25.11.2017

@author: Daniel
'''
import cv2
import pylab as plt
from Task3.Otsus import otsu
from Task3.CannyEdgeDetector import canny

img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)

res = otsu(img)
    
plt.subplot(1,2,1); plt.imshow(img,'gray')
plt.title('Original')
plt.subplot(1,2,2);
plt.imshow(res,'gray')
plt.title('Otsu\'s - Threshold = 137')
plt.show()
    
img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)

result = canny(img)

plt.subplot(1,2,1);plt.imshow(img, 'gray')
plt.title('Original Image')
#plt.subplot(1,2,2);plt.imshow(result, 'gray')
plt.title('Canny Image')

plt.show()
