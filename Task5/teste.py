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
    X = train - numpy.matlib.repmat(avg_f.reshape((h*w, 1)), 1, len(train[0]))
    neigh = KNeighborsClassifier(n_neighbors=9)
    
    Xclf = []
    for i in range(num_images):
        Xclf.append([])
        for j in range(h*w):
            Xclf[i].append(train[j][i])

    print len(Xclf)
    print len(Xclf[1])

    neigh.fit(X, labels)
    print labels
    u, s, v = np.linalg.svd(X)

    #print('Size of u: {0} x {1}'.format(len(u), len(u[0])))
    #print('Size of s: {0}'.format(len(s)))
    #print('Size of v: {0} x {1}'.format(len(v), len(v[0])))

    #S = np.zeros((len(u), len(v)), dtype=np.float64)
    #S[:len(s), :len(s)] = np.diag(s)
    #print(np.allclose(X, np.dot(u, np.dot(S, v))))

    #print(np.allclose(np.dot(u, u.T), np.eye(N * N)))
    #print(np.allclose(np.dot(u, u.T), np.eye(N * N)))

    #i = 0
    #eigvec = u[:,i]
    #eigval = s[i]**2
    #print(np.allclose(np.dot(np.dot(X, X.T), eigvec), eigval * eigvec))

    #i = 0
    #eigvec = v[i]
    #eigval = s[i]**2
    #print(np.allclose(np.dot(np.dot(X.T, X), eigvec), eigval * eigvec))

    #plt.imshow(avg_f.reshape((64,64)), 'gray')
    #plt.show()
    return u, len(s), avg_f.reshape((h, w)), neigh

# Reconstruct an image with a specific number of eigenfaces
# img: img to reconstruct
# u: u from the trainMethod
# num_eigenfaces: the number of eigenfaces
# avg: average face
# h,w: height, widht
def reconstruct_image(img,u,num_eigenfaces,avg,h,w):
    img_zeromean = img - avg
    recon_images = np.zeros((h*w, num_eigenfaces+1))
    rmse = np.zeros(num_eigenfaces + 1)
    nfactor = np.linalg.norm(img)

    for n in range(num_eigenfaces + 1):
        if n == 0:
            recon_img = np.copy(avg)
            recon_images[:, 0] = avg.reshape(h*w)
            rmse[0] = numpy.linalg.norm(img - avg) / nfactor
        else:
            eigenface = u[:, n-1].reshape(h*w)
            recon_img += (np.dot(eigenface, img_zeromean.reshape(h*w)) * eigenface).reshape((h, w))
            recon_images[:, n] = recon_img.reshape((h*w))
            rmse[n] = numpy.linalg.norm(recon_img - img) / nfactor


    plt.imshow(recon_img.reshape((64,64)), 'gray')
    plt.show()
    return "'Daniel Stromer_12.png'"

