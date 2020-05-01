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


class FaceRec:
    def __init__(self, cam_index = -1, embedding_file="", facenet_model="", display=False, verbose=False):
        # load face embeddings
        data = load( embedding_file )
        trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

        self.emb_model = load_model( facenet_model )

        # normalize input vectors
        in_encoder = Normalizer(norm='l2')
        trainX = in_encoder.transform(trainX)
        testX = in_encoder.transform(testX)

        # label encode targets
        self.out_encoder = LabelEncoder()
        self.out_encoder.fit(trainy)
        trainy = self.out_encoder.transform(trainy)
        testy = self.out_encoder.transform(testy)

        # fit model
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(trainX, trainy)

        # self.detector = MTCNN()
        self.bool_display = display
        self.bool_verbose = verbose
        self.detector = MTCNN()
        self.cam_index = cam_index

        # Start webcam
        self.cap = cv2.VideoCapture(self.cam_index)
        print("[ INFO ] Starting webcam...")



    def get_embedding(self, model, face_pixels):
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


    def face_cap(self):
        ct_frame = 0
        Person = namedtuple('Person', ['name', 'probability'])
        lt_personSample = []

        while ct_frame < 10: 
            # Capture frame-by-frame
            __, frame = self.cap.read()    # __, is necessary
            try:
                # Use MTCNN to detect faces
                result = self.detector.detect_faces(frame)
            except self.detector.InvalidImage:
                print("Invalid Image")
                pass
            if result != []:
                for person in result:
                    # print(person)
                    face_box = person['box']
                    x1, y1, width, height = face_box
                    keypoints = person['keypoints']

                    if self.bool_display == True:
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

                    face_emb = self.get_embedding(self.emb_model, face_array)

                    # prediction for the faces
                    samples = expand_dims(face_emb, axis=0)
                    yhat_class = self.model.predict(samples)
                    yhat_prob = self.model.predict_proba(samples)
                    
                    # get name
                    class_index = yhat_class[0]
                    class_probability = yhat_prob[0,class_index] * 100
                    predict_names = self.out_encoder.inverse_transform(yhat_class)

                    if self.bool_display or self.bool_verbose == True:
                        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

                    lt_personSample.append(Person(predict_names[0], class_probability))

            else:
                print("No face detected")
                lt_personSample.append(Person("", 0))
            
            if self.bool_display == True:
                #display resulting frame
                cv2.imshow('frame',frame)

            if cv2.waitKey(1) &0xFF == ord(' '):
                print("[ INFO ] Ending stream...")
                break
            
            ct_frame += 1

        # When everything's done, release capture
        self.cap.release()
        cv2.destroyAllWindows()

        name, probMean = self.probability(lt_personSample)
        if self.bool_display or self.bool_verbose == True:
            print(name, probMean)

        return name, probMean


    def probability(self, lt_personSample):
        probMean = mean(person.probability for person in lt_personSample)
        highestFreq = 0
        name = ""
        for person in lt_personSample:
            freq = sum(p.name == person.name for p in lt_personSample)
            if person.name == name: continue
            if freq > highestFreq:
                highestFreq = freq
                name = person.name

        if probMean <= 80:
            return "", probMean
        else:
            return name, probMean

