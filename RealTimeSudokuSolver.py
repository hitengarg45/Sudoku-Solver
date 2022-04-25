import cv2
import numpy as np
import math
from scipy import ndimage
import copy
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import sudokuSolver
from StackImages import stackImages

###############################################
frameWidth = 640
frameHeight = 480
SIZE = 9
digitPicSize = 28
###############################################


# Write solution on "image"
def writeSolutionOnImage(img, grid, useGrid):
    # Write grid on image
    width = img.shape[1] // 9
    height = img.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if useGrid[i][j] != 0:  # If user fill this cell
                continue  # Move on
            text = str(grid[i][j])
            offSetx = width // 15
            offSety = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (textHeight, textWidth), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)

            fontScale = 0.6 * min(width, height) / max(textHeight, textWidth)
            textHeight *= fontScale
            textWidth *= fontScale
            bottomLeftCornerx = width * j + math.floor((width - textWidth) / 2) + offSetx
            bottomLeftCornery = height * (i + 1) - math.floor((height - textHeight) / 2) + offSety
            img = cv2.putText(img, text, (bottomLeftCornerx, bottomLeftCornery),
                              font, fontScale, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    return img


# This function is used for seperating the digit from noise in "crop_image"
# The Sudoku board will be chopped into 9x9 small square image,
# each of those image is a "crop_image"
def largestConnectedComponent(img):
    img = img.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[:, -1]

    if(len(sizes) <= 1):
        imgBlank = np.zeros(img.shape)
        imgBlank.fill(255)
        return imgBlank

    maxLabel = 1
    # Start from component 1 (not 0) because we want to leave out the background
    maxSize = sizes[1]

    for i in range(2, nb_components):
        if sizes[i] > maxSize:
            maxLabel = i
            maxSize = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == maxLabel] = 0
    return img2


# Calculate how to centralize the image using its center of mass
def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty


# Shift the image using what get_best_shift returns
def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


# Prepare and normalize the image to get ready for digit recognition
def prepare(img_array):
    new_array = img_array.reshape(-1, 28, 28, 1)
    new_array = new_array.astype('float32')
    new_array /= 255
    return new_array


def twoMatricesAreEqual(matrix1, matrix2, row, col):
    for i in range(row):
        for j in range(col):
            if matrix1[i][j] != matrix2[i][j]:
                return False
    return True


#Get Binary Image
def getBinaryImage(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgAdaptiveThresh = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    # imgAdaptiveThresh = cv2.bitwise_not(imgAdaptiveThresh)
    # _, imgAdaptiveThresh = cv2.threshold(imgAdaptiveThresh, 150, 255, cv2.THRESH_BINARY_INV)

    return imgAdaptiveThresh


#Detect Contours and corners for sudoku grid
def getContours(img, imgContour):
    biggest = np.array([])
    maxArea = 0

    #For pycharm
    #contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #For spyder
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 20000:
            cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return imgContour, biggest


#Reorder the corner points for warp prespective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    #get index of smallest value and biggest value
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


#Get warp prespective
def getWarpPrespective(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (frameWidth, frameHeight))

    return imgOutput, matrix


def getWarpThres(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgAdaptiveThresh = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    imgAdaptiveThresh = cv2.bitwise_not(imgAdaptiveThresh)
    _, imgAdaptiveThresh = cv2.threshold(imgAdaptiveThresh, 150, 255, cv2.THRESH_BINARY)

    return imgAdaptiveThresh


#Dividing into 9x9 grid and solving
def divideAndExtractAndSolve(img, imgWarp, imgOriginal, prespectiveTransformedMatrix, model, oldSudoku):
    #Count to check if digits are recognized properly
    count = 0

    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)

    cropWidth = img.shape[1] // 9
    cropHeight = img.shape[0] // 9

    offsetWidth = math.floor(cropWidth / 10)   # Offset is used to get rid of the boundaries
    offsetHeight = math.floor(cropHeight / 10)

    #Divide
    for i in range(SIZE):
        for j in range(SIZE):
            imgCrop = img[cropHeight * i + offsetHeight:cropHeight * (i + 1) - offsetHeight,
                          cropWidth * j + offsetWidth:cropWidth * (j + 1) - offsetWidth]
            #Fisrt test
            #cv2.imshow(str(i) + "_" + str(j), imgCrop)

            imgCrop = cv2.bitwise_not(imgCrop)
            imgCrop = largestConnectedComponent(imgCrop)

            #Resize
            imgCrop = cv2.resize(imgCrop, (digitPicSize, digitPicSize))

            #If white cell, set grid[i][j] = 0 and move to next image
            #Detecting white cell

            #Criteria-1 => Too little black pixels
            if imgCrop.sum() >= digitPicSize**2*255 - digitPicSize * 1 * 255:
                grid[i][j] = 0
                continue

            #Criteria - 2 => Huge White area in center
            centerWidth = imgCrop.shape[1] // 2
            centerHeight = imgCrop.shape[0] // 2
            xStart = centerHeight // 2
            xEnd = centerHeight // 2 + centerHeight
            yStart = centerWidth // 2
            yEnd = centerWidth // 2 + centerWidth
            imgCropCenterRegion = imgCrop[xStart:xEnd, yStart:yEnd]

            if imgCropCenterRegion.sum() >= centerWidth * centerHeight * 255 - 255:
                grid[i][j] = 0
                continue

            #Now Image crop will contain a number
            rows, cols = imgCrop.shape

            #Apply Binary Threshold to make digits more clear
            _, imgCrop = cv2.threshold(imgCrop, 200, 255, cv2.THRESH_BINARY)
            imgCrop = imgCrop.astype(np.uint8)

            #Centralize the image according to center of mass
            imgCrop = cv2.bitwise_not(imgCrop)
            shiftx, shifty = getBestShift(imgCrop)
            shifted = shift(imgCrop, shiftx, shifty)
            imgCrop = shifted

            imgCrop = cv2.bitwise_not(imgCrop)

            #Increment count if digit block
            count = count + 1

            # Up to this point crop_image is good and clean!
            #cv2.imshow(str(i)+str(j), imgCrop)

            # Convert to proper format to recognize
            imgCrop = prepare(imgCrop)

            #Recognize digits
            #Model trained in digitRecognition.py
            prediction = model.predict([imgCrop])

            # 1 2 3 4 5 6 7 8 9 starts from 0, so add 1
            grid[i][j] = np.argmax(prediction[0]) + 1

    userGrid = copy.deepcopy(grid)

    #Solve sudoku after digit recognition on sudoku board

    #If this sudoku board is same as previous board
    #No need to solve again
    if (not oldSudoku is None) and twoMatricesAreEqual(oldSudoku, grid, 9, 9):
        if(sudokuSolver.all_board_non_zero(grid)):
            imgWarp = writeSolutionOnImage(imgWarp, oldSudoku, userGrid)
    #If different board
    else:
        sudokuSolver.solve_sudoku(grid)  # Solve it
        if (sudokuSolver.all_board_non_zero(grid)):  # If we got a solution
            imgWarp = writeSolutionOnImage(imgWarp, grid, userGrid)
            oldSudoku = copy.deepcopy(grid)  # Keep the old solution

    # Apply inverse perspective transform and paste the solutions on top of the orginal image
    imgSolved = cv2.warpPerspective(imgWarp, prespectiveTransformedMatrix, (imgOriginal.shape[1], imgOriginal.shape[0])
                                    , flags=cv2.WARP_INVERSE_MAP)
    imgResult = np.where(imgSolved.sum(axis=-1, keepdims=True) != 0, imgSolved, imgOriginal)

    #Print count of Digits
    print("Digits count: " + str(count))
    return imgResult

def recognizeAndSolve(img, model, oldSudoku):
    #Clone Image for later use
    imgClone = np.copy(img)

    imgBinary = getBinaryImage(img)
    imgContour = img.copy()
    imgContour, biggest = getContours(imgBinary, imgContour)

    if biggest.size != 0:
        imgWarped, prespectiveTransformedMatrix = getWarpPrespective(img, biggest)
        imgWarpThresh = getWarpThres(imgWarped)

        #imgArray = ([img, imgContour], [imgWarped, imgWarpThresh])
        #imgStack = stackImages(0.6, imgArray)
        #cv2.imshow("WebCam1", imgStack)

        imgSolved = divideAndExtractAndSolve(imgWarpThresh, imgWarped, img, prespectiveTransformedMatrix, model, oldSudoku)
        return imgSolved

    else:
        return img