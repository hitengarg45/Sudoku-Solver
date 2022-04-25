# This is the entry point. Run this file!
# You don't need to run digitRecognition.py to train the Convolutional Neural Network (CNN).
# I have trained the CNN on my computer and saved the architecture in digitRecognition.h5

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import RealTimeSudokuSolver

############################################
widthImg = 640
heightImg = 480
brightness = 150
############################################

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

# Loading model (Load weights and configuration separately to speed up model.predict)
input_shape = (28, 28, 1)
num_classes = 9
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Load weights from pre-trained model. This model is trained in digitRecognition.py
model.load_weights("digitRecognition.h5")

# Let's turn on webcam
cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, brightness)

oldSudoku = None
while True:
    # Read the frame
    success, img = cap.read()
    cv2.imshow("Webcam", img)
    if success == True:
        #Solved Sudoku Frame
        imgSudoku = RealTimeSudokuSolver.recognizeAndSolve(img, model, oldSudoku)

        # Print the 'solved' image
        showImage(imgSudoku, "Real Time Sudoku Solver", widthImg, heightImg)

        # Hit q if you want to stop the camera
        if cv2.waitKey(1) == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()