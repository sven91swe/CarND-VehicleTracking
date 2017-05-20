from undistort import undistort
from udacityCode import find_cars, add_heat, draw_labeled_bboxes, apply_threshold, draw_boxes
import numpy as np
from scipy.ndimage.measurements import label

scales = {0.5: {'y': (410, 455), 'x': (300, 1280-300)},
          0.75: {'y': (410, 455), 'x': (200, 1280-200)},
          1: {'y': (400, 485), 'x': (0, 1280)},
          1.5: {'y': (390, 656), 'x': (0, 1280)},
          2: {'y': (380, 656), 'x': (0, 1280)},
          2.5: {'y': (380, 700), 'x': (0, 1280)}}

heatmap = np.zeros((720, 1280))

threshold = 1
historyFactor = 4

def pipeline(image):
    image = undistort(image)
    listOfAllBoxes = []
    for scale in scales:
        foundCars, allSquares, listOfBoxes = find_cars(image,
                                                       scales[scale]['y'][0],
                                                       scales[scale]['y'][1],
                                                       scales[scale]['x'][0],
                                                       scales[scale]['x'][1],
                                                       scale)
        listOfAllBoxes += listOfBoxes


    global heatmap
    heatmap *= (historyFactor-1)
    heatmap = add_heat(heatmap, listOfAllBoxes)
    heatmap /= historyFactor

    labels = label(apply_threshold(heatmap, threshold))

    allFoundCars = draw_labeled_bboxes(image, labels)

    return allFoundCars


def pipelineSingleFrameDetection(image):
    image = undistort(image)
    listOfAllBoxes = []
    for scale in scales:
        foundCars, allSquares, listOfBoxes = find_cars(image,
                                                       scales[scale]['y'][0],
                                                       scales[scale]['y'][1],
                                                       scales[scale]['x'][0],
                                                       scales[scale]['x'][1],
                                                       scale)
        listOfAllBoxes += listOfBoxes

    allFoundCars = draw_boxes(image, listOfAllBoxes)

    return allFoundCars