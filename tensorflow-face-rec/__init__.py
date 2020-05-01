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
