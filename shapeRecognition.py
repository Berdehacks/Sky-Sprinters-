
import cv2
import numpy as np


def shapeRecognition(img):

    # img = img[inputRoiP1[1]: inputRoiP2[1], inputRoiP1[0]: inputRoiP2[0]]

    # converting image into grayscale imag
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 140, 180, cv2.THRESH_OTSU)

    cv2.imshow("roi", threshold)

    # threshold = cv2.bitwise_not(threshold)

    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # EDITAR TRESHOLD DE FIGURAS
    for contour in contours[1:]:
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function

    # finding center point of shape
    M = cv2.moments(contour)

    # putting shape name at center of each shape
    if len(approx) == 3:

        return "triangle"

    elif len(approx) == 4:

        return "trapezoid"

    elif len(approx) == 6:

        return "hexagon"

    else:
        return "circle"

    return
