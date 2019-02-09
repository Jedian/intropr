import cv2
from FaceRecognition.Eigenfaces import *
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from time import time
import glob
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#online detection: True -> webcam!!!
#offline Detection: False -> Do cross validation
onlineDetection = False

if onlineDetection:
    # image size
    N = 64
    
    #cascade classifier for face detection in images
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    #create database from images in yalefaces folder
    labels,train,num_images = createOnlineDatabase(glob.glob('eigenfaces/*.png'))
    
    #train the database
    u,num_eigenfaces,avg = trainDatabase(labels,train,num_images,N,N)
    
    #gets the video source
    video_capture = cv2.VideoCapture(0)
    
    #run until 'q' is pressed
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()
        # get gray value image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect face with haar cascade
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,  minSize=(N-10, N-10))
        
        # draw rectangle around face
        for (x, y, w, h) in faces:
            
            #extract face from frame and gray value image
            face = frame[y: y + h, x: x + w]
            image = gray[y: y + h, x: x + w]
            
            # equalize histogram
            image = cv2.equalizeHist(image)
            # resize to NxN
            image=cv2.resize(image,(N,N))
    
            #draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            #time measurement
            start = time()
            #predict face and store label as string
            pred = str(reconstruct_image(image,u,num_eigenfaces,avg,N,N))
            
            #creating a label to draw it (Name, and time in ms
            name = str(pred).split("'")[1].split("_")[0]+"_"+str((time()-start)*1000).split(".")[0]+" ms"
            # draw the label to the frame
            cv2.putText(frame, name.split("_")[0], (int(x),int(y+h+25)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1,cv2.LINE_AA)
            cv2.putText(frame, name.split("_")[1], (int(x),int(y+h+50)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1,cv2.LINE_AA)
        
        #show the video
        cv2.imshow('Video', frame)
        #break the loop and release webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release camera
    video_capture.release()  
    cv2.destroyAllWindows() 
    
else:
    # load the dataset and extract dimensions
    lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    data = lfw_dataset['data']
    target = lfw_dataset['target']
    labels = lfw_dataset['target_names']
    num_images, h, w = lfw_dataset.images.shape

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

    X_train2 = np.zeros((h*w, len(X_train)))

    for n in range(len(X_train)):
        X_train2[:,n] = X_train[n].reshape((h*w))

    trainDatabase(y_train, X_train2, len(X_train), h, w)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=labels))
