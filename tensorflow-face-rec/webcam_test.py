from numpy import load
from numpy import expand_dims
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from numpy import asarray
from mtcnn.mtcnn import MTCNN

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# load faces
data = load('dataset.npz')
testX_faces = data['arr_2']

# load face embeddings
data = load('embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)           #### Magic Happens HERE!!!!!******

detector = MTCNN()

# Start webcam
print("[ INFO ] Starting webcam...")
cap = cv2.VideoCapture(0)

while True: 
    #Capture frame-by-frame
    __, frame = cap.read()    # __, is necessary
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            face_box = person['box']
            keypoints = person['keypoints']

            cv2.rectangle(frame,
                        (face_box[0], face_box[1]),
                        (face_box[0]+face_box[2], face_box[1] + face_box[3]),
                        (0,155,255),2)
            
            frame_emb = in_encoder.transform(face_box)

            # prediction for the faces
            samples = expand_dims(frame_emb, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)
            
            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = out_encoder.inverse_transform(yhat_class)
            print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord(' '):
        print("[ INFO ] Ending stream...")
        break

#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()