# python3 detectFacesVideopy --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

#import mecessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

#parse command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, 
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over frames from the video stream
while True:
    #grab the frame from the threaded video stream and resize it to 
    #have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    #grab the frame dimensions and convert to blob
    (h, w) = frame.shape[:2]
    resize = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resize, 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    
    #pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        #extract the confidence (i.e., probability) associated with the
        #prediction
        confidence = detections[0,0,i,2]

        #filter weak detections by ensuring the confidence is 
        #greater than the minimum prob
        if confidence < args["confidence"]:
            continue

        #compute the (x,y) coordinates of the bounding box for the object
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startx, starty, endx, endy) = box.astype("int")

        #draw the bound box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = starty - 10 if starty - 10 > 10 else starty + 10

        cv2.rectangle(frame, (startx, starty), (endx, endy), (0,0,255), 2)
        cv2.putText(frame, text, (startx, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
    
    #show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if 'q' is pressed, break from loop
    if key == ord('q'):
        break

#clean up
cv2.destroyAllWindows()
vs.stop()