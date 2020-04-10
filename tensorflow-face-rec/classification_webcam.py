from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from imutils.video import VideoStream
from keras.models import load_model
from statistics import mean
import cv2
from collections import namedtuple
from PIL import Image

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

# load face embeddings
data = load('embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

emb_model = load_model("./model/facenet_keras.h5")

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
model.fit(trainX, trainy)

detector = MTCNN()

# Start webcam
cap = cv2.VideoCapture(0)
<<<<<<< HEAD:tensorflow-face-rec/classification_webcam.py
print("[ INFO ] Starting webcam...")

ct_frame = 0
Person = namedtuple('Person', ['name', 'probability'])
lt_personSample = []
=======

>>>>>>> 5dfbb077e30ffae2284899e7485a4b09759891c8:tensorflow-face-rec/webcam_test.py

while ct_frame < 10: 
    # Capture frame-by-frame
    __, frame = cap.read()    # __, is necessary
    
    # Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            # print(person)
            face_box = person['box']
            x1, y1, width, height = face_box
            keypoints = person['keypoints']

            cv2.rectangle(frame,
                        (face_box[0], face_box[1]),
                        (face_box[0]+face_box[2], face_box[1] + face_box[3]),
                        (0,155,255),2)
            
            pixels = asarray(frame)

            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # Extract face as pixels
            face_pixels = pixels[y1:y2, x1:x2]
            face_img = Image.fromarray(face_pixels)
            face_img = face_img.resize((160, 160))
            face_array = asarray(face_img)

            face_emb = get_embedding(emb_model, face_array)

            # prediction for the faces
            samples = expand_dims(face_emb, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)
            
            # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = out_encoder.inverse_transform(yhat_class)
            print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
            lt_personSample.append(Person(predict_names[0], class_probability))

    else:
        print("No face detected")
        
    #display resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) &0xFF == ord(' '):
        print("[ INFO ] Ending stream...")
        break
    
    ct_frame += 1

# When everything's done, release capture
cap.release()
cv2.destroyAllWindows()

probMean = mean(Person.probability for Person in lt_personSample)
highestFreq = 0
name = ""
for person in lt_personSample:
    freq = sum(p.name == person.name for p in lt_personSample)
    if person.name == name: continue
    if freq > highestFreq:
        highestFreq = freq
        name = person.name

print("Person: {} with {}".format(name, mean(Person.probability for Person in lt_personSample)))
