import numpy as np
import pylab as plt
import cv2

from HandRecognitionPackage.FourierTransform import calcuateFourierParameters
from HandRecognitionPackage.DistanceMeasure import calculate_Theta_Distance, calculate_R_Distance
from HandRecognitionPackage.PalmprintAlignment import preProcessing

k = 8
samplingSize = 100

img1 = cv2.imread('Hand1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Hand2.jpg',cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('Hand3.jpg',cv2.IMREAD_GRAYSCALE)

plt.subplot(2, 3, 1), plt.imshow(img1,'gray')
plt.axis('off')
plt.title('PP 1 unaligned');
plt.subplot(2, 3, 2), plt.imshow(img2, 'gray')
plt.axis('off')
plt.title('PP 2 unaligned');
plt.subplot(2, 3, 3), plt.imshow(img3,'gray')
plt.axis('off')
plt.title('PP 3 unaligned');

img1 = preProcessing(img1)
img2 = preProcessing(img2)
img3 = preProcessing(img3)

RX,ThetaX = calcuateFourierParameters(img1, k, samplingSize)
RY,ThetaY = calcuateFourierParameters(img2, k, samplingSize)
RZ,ThetaZ = calcuateFourierParameters(img3, k, samplingSize)

DR_xx       = calculate_R_Distance(RX,RX,k)
DTheta_xx   = calculate_Theta_Distance(ThetaX,ThetaX,k)

DR_xy       = calculate_R_Distance(RX,RY,k)
DTheta_xy   = calculate_Theta_Distance(ThetaX,ThetaY,k)

DR_yx       = calculate_R_Distance(RY,RX,k)
DTheta_yx   = calculate_Theta_Distance(ThetaY,ThetaX,k)

DR_xz       = calculate_R_Distance(RX,RZ,k)
DTheta_xz   = calculate_Theta_Distance(ThetaX,ThetaZ,k)

DR_yz       = calculate_R_Distance(RY,RZ,k)
DTheta_yz   = calculate_Theta_Distance(ThetaY,ThetaZ,k)

print("DR: Ring-like area")
print("DTheta: Fan-like area")
print("The smaller the value, the better the match. d=0: completely identical")

print("X-X")
print("DR: "+format(np.round(DR_xx,1)/samplingSize))
print("DTheta :"+format(np.round(DTheta_xx,2))+" %")

print("Y-X")
print("DR: "+format(np.round(DR_yx/samplingSize,1)))
print("DTheta :"+format(np.round(DTheta_yx,2))+" %")

print("X-Y")
print("DR: "+format(np.round(DR_xy/samplingSize,1)))
print("DTheta :"+format(np.round(DTheta_xy,2))+" %")

print("X-Z")
print("DR: "+format(np.round(DR_xz/samplingSize,1)))
print("DTheta :"+format(np.round(DTheta_xz,2))+" %")

print("Y-Z")
print("DR: "+format(np.round(DR_yz/samplingSize,1)))
print("DTheta :"+format(np.round(DTheta_yz,2))+" %")

plt.subplot(2, 3, 4), plt.imshow(img1,'gray')
plt.axis('off')
plt.title('PP 1 aligned');
plt.subplot(2, 3, 5), plt.imshow(img2, 'gray')
plt.axis('off')
plt.title('PP 2 aligned');
plt.subplot(2, 3, 6), plt.imshow(img3,'gray')
plt.axis('off')
plt.title('PP 3 aligned');
plt.show()
