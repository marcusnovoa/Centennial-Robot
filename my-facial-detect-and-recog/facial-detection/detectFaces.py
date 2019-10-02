# python detectFaces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import packages
import numpy as np
import argparse
import cv2

#construct argument parse and parse command line args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'delpoy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, 
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#load the input image and construct an input blob for the image
#by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0,
    (300,300), (104.0,177.0,123.0))

#pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

#loop through detections
for i in range(0, detections.shape[2]):
    #extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    #filter weak detections by ensuring the confidence is greater than 
    #the minimum confidence
    if confidence > args["confidence"]:
        #compute the (x, y) coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startx, starty, endx, endy) = box.astype("int")

        #draw bounding box of the face allong with the associated prob
        text = "{:.2f}%".format(confidence * 100)
        y = starty - 10 if starty -10 > 10 else starty +10
        cv2.rectangle(image, (startx, starty), (endx, endy), (0, 0, 255), 2)
        cv2.putText(image, text, (startx, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

#show output
cv2.imshow("Output", image)
cv2.waitKey(0)