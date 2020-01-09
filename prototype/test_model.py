import numpy as np
import pickle
import cv2, os
import matplotlib.pyplot as plt
from glob import glob
from keras import optimizers
import sqlite3, pyttsx3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from threading import Thread
from keras import backend as K
K.set_image_dim_ordering('tf')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')

def get_image_size():
	img = cv2.imread('gestures/01/100.jpg', 0)
	return img.shape

def get_num_of_classes():
	return len(glob('gestures/*'))+1

image_x, image_y = get_image_size()



def test():
	with open("test_images", "rb") as f:
		test_images = np.array(pickle.load(f))
	with open("test_labels", "rb") as f:
		test_labels = np.array(pickle.load(f), dtype=np.int32)


	test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
	test_labels = np_utils.to_categorical(test_labels)

	print(test_labels.shape)

	model.summary()
	scores = model.evaluate(test_images, test_labels, verbose=0)
	print("Test dataset")
	print("CNN Error: %.2f%%" % (100-scores[1]*100))
	print("Accuracy: %.2f%%" % (scores[1]*100))
	

test()
K.clear_session();
