'''
Created on 31.01.2017

@author: DanielSt
'''
import cv2
import glob
import numpy as np

#New Image Size
N = 64
#Path of Image - Path to store Image
path = 'C:/temp/'

def detectFaces(N):
    # get all files in folder
    filenames = np.sort(glob.glob(path+'*.jpg'))
    num_images = len(filenames)
    # face cascade to find face in image
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #iterate over all images in folder
    for n in range(num_images):
        #read image
        img = cv2.imread(filenames[n], cv2.IMREAD_GRAYSCALE)
        #detect face
        faces = faceCascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3,  minSize=(N-5, N-5))
        # crop rectangle and resize to NxN
        for (x, y, w, h) in faces:
            #new image with NxN
            image = img[y: y + h, x: x + w]
            image = cv2.equalizeHist(image)
            image=cv2.resize(image,(N,N))
            #saving path and name - replace test object for name
            filename = path+"TestObject"+str(n)+".png";
            #save image into the folder
            cv2.imwrite(filename,image)
            print("written to "+filename)

#run the code    
detectFaces(N)
