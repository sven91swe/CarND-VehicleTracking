import cv2
import os
from datetime import datetime
import matplotlib.image as mpimg
from undistort import undistort
from udacityCode import find_cars, draw_boxes


pathToTestImages = "../test_images"
outputFolder = "../output_images"

listTestImages = os.listdir(pathToTestImages)

image = cv2.imread("../camera_cal/calibration1.jpg")
undistortedImage = undistort(image)
cv2.imwrite(outputFolder + "/" + "undist-chessboard.jpg", undistortedImage)



#This dict describes the different scales that will be used and also on which subset of the image it will be used.
scales = {0.5: {'y':(410, 455), 'x': (300, 1280-300)},
          0.75: {'y':(410, 474), 'x': (200, 1280-200)},
          1: {'y':(400, 485), 'x': (0, 1280)},
          1.5: {'y':(390, 656), 'x': (0, 1280)},
          2: {'y':(380, 656), 'x': (0, 1280)},
          2.5: {'y':(380, 700), 'x': (0, 1280)}}


for imageName in listTestImages:
    image = mpimg.imread(pathToTestImages + "/" + imageName)
    image = undistort(image)
    print(imageName)
    start = datetime.now()
    listOfAllBoxes = []
    for scale in scales:
        foundCars, allSquares, listOfBoxes = find_cars(image,
                                                       scales[scale]['y'][0],
                                                       scales[scale]['y'][1],
                                                       scales[scale]['x'][0],
                                                       scales[scale]['x'][1],
                                                       scale)
        listOfAllBoxes += listOfBoxes
        print(scale, "  --  ", len(listOfBoxes))
        foundCarsBGR = cv2.cvtColor(foundCars, cv2.COLOR_RGB2BGR)
        allSquaresBGR = cv2.cvtColor(allSquares, cv2.COLOR_RGB2BGR)
        cv2.imwrite(outputFolder + "/" + "foundCars%1.2f-" % (scale) + imageName, foundCarsBGR)
        cv2.imwrite(outputFolder + "/" + "allSquares%1.2f-" % (scale) + imageName, allSquaresBGR)

    allFoundCars = draw_boxes(image, listOfAllBoxes)
    allFoundCarsBGR = cv2.cvtColor(allFoundCars, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outputFolder + "/" + "allFoundCars-" + imageName, allFoundCarsBGR)

    print(datetime.now() - start)

