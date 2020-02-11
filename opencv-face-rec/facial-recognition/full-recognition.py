# Full facial recognition training and execution
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import imutils
import pickle
import cv2
import os


confidence_min = 0.5
dataset_str = "dataset"
embeddings_str = "output/embeddings.pickle"
detector_str = "face_detection_model"
embedding_model_str = "openface_nn4.small2.v1.t7"
recognizer_str = "output/recognizer.pickle"
le_str = "output/le.pickle"
output_image = "images/kaplan0.jpg"


#______________Extract Embeddings_____________________________________________________________
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_str, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_str, "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#load our serialized face embedding model
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model_str)

#grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(dataset_str))

#initialize our list of extracted facial embeddings and 
#corresponding people names
knownEmbeddings = []
knownNames = []

#initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

    # load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

    # construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    # apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

    # ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
		# means our minimum probability test (thus helping filter out
		# weak detections)
		if confidence > confidence_min:
            # compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face
			# embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(embeddings_str, "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] done extracting embeddings.")

#______________Train Model_____________________________________________________________
#load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(embeddings_str, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

#train the model used to accept the 128-d embeddings of the face and then
#produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

#write the actual face recognition model
f = open(recognizer_str, "wb")
f.write(pickle.dumps(recognizer))
f.close()

#write the label encoder
f = open(le_str, "wb")
f.write(pickle.dumps(le))
f.close()

#______________Recognition_____________________________________________________________
#load serialized face detector
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector_str, "deploy.prototxt"])
modelPath = os.path.sep.join([detector_str,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embedding_model_str)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(recognizer_str, "rb").read())
le = pickle.loads(open(le_str, "rb").read())

# load the image, resize it to have a width of 600 pixels (while
# maintaining the aspect ratio), and then grab the image dimensions
image = cv2.imread(output_image)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(image, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]
    # filter out weak detections
	if confidence > confidence_min:
		# compute the (x, y)-coordinates of the bounding box for the
		# face
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue
        
        # construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

        # perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		# draw the bounding box of the face along with the associated
		# probability
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
