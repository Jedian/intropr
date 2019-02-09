import numpy as np
import numpy.matlib
import cv2
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#image size
N = 64

# Define the classifier in clf - Try a KNearestNeighbor Classifier with 9 neighbors
#clf = KNeighborsClassifier(n_neighbors=9)
# You can also try a Support Vector Machine with C = 0.025 and a linear kernel
clf = SVC(kernel="linear", C =0.025)

# Don't touch this method. If you run the Online Detection, this function will create and reshape the 
# images located in the database. You pass the path of the images and the function returns the labels,
# training data and number of images in the database
def createOnlineDatabase(path):
    labels = list()
    filenames = np.sort(path)
    num_images = len(filenames)
    train = np.zeros((N*N, num_images))
    for n in range(num_images):
        img = cv2.imread(filenames[n], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(N,N))
        assert img.shape == (N, N), 'Image {0} of wrong size'.format(filenames[n])
        train[:,n] = img.reshape((N*N))
        labels.append(filenames[n].split("eigenfaces/")[1].split("_")[0])
    print('Database contains {0} images'.format(num_images))
    labels = np.asarray(labels)
    return labels, train, num_images

# TODO: Train the database - you get an array of the labels and the training data, the number of images
# and the the height and width of one single image
# The faces are not stored in a 2-D matrix but in a 1-D vector (e.g. 64x64 image ->  4096 vector)
# h,w: height, widht
def trainDatabase(labels, train, num_images,h,w):
    avg_f = np.mean(train, 1)
    plt.imshow(avg.reshape((64, 64)), 'gray')
    plt.axis('off')

# Reconstruct an image with a specific number of eigenfaces
# img: img to reconstruct
# u: u from the trainMethod
# num_eigenfaces: the number of eigenfaces
# avg: average face
# h,w: height, widht
def reconstruct_image(img,u,num_eigenfaces,avg,h,w):


