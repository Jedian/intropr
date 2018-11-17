import pylab as plt
import cv2
from Task_2.BilateralFilter import bilateralFilter
import skimage.util

fig= plt.figure()
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
#img = img /255
img = skimage.util.random_noise(img, mode='s&p')

plt.subplot(2,2,1);plt.imshow(img, 'gray')
plt.title('Original Image')

result = bilateralFilter(img, 5 ,2, 2)

plt.subplot(2,2,2);plt.imshow(result, 'gray')
plt.title('Bilateral Filtered Image')

plt.show()
