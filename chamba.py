import numpy as np
from djitellopy import tello
import cv2

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

me.takeoff()
desired_height = 1  # in centimeters
me.send_rc_control(0, 0, desired_height, 0)

cap = cv2.VideoCapture(1)

# Thresholding HSV values for light blue
hsvVals = [90, 80, 100, 120, 255, 255]

sensors = 3
threshold = 0.2
width, height = 480, 360
senstivity = 3
weights = [-25, -15, 0, 15, 25]
fSpeed = 20
curve = 0

def thresholding(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def getContours(imgThres, img):
    cx = 0
    contours, hieracrhy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2
        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return cx

def getSensorOutput(imgThres, sensors):
    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (img.shape[1] // sensors) * img.shape[0]
    senOut = []
    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)
    return senOut

def sendCommands(senOut, cx, width):
    global curve
    ## TRANSLATION
    lr = (cx - width // 2) // senstivity
    lr = int(np.clip(lr, -10, 10))
    if 2 > lr > -2:
        lr = 0

    ## Rotation (YAW)
    path_orientation = (width // 2 - cx) / (width // 2)  # Calculate path orientation

    if senOut == [1, 0, 0]: 
        curve = int(weights[0])
    elif senOut == [1, 1, 0]:
        curve = int(weights[1])
    elif senOut == [0, 1, 0]:
        curve = int(weights[2])
    elif senOut == [0, 1, 1]:
        curve = int(weights[3])
    elif senOut == [0, 0, 1]:
        curve = int(weights[4])
    elif senOut == [0, 0, 0]:
        curve = int(weights[2])
    elif senOut == [1, 1, 1]:
        curve = int(weights[2])
    elif senOut == [1, 0, 1]:
        curve = int(weights[2])

    # Adjust YAW based on path orientation
    curve = int(curve - path_orientation * 50)  # Convert curve to an integer

    me.send_rc_control(lr, fSpeed, 0, curve)

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)
    imgThres = thresholding(img)
    cx = getContours(imgThres, img)
    senOut = getSensorOutput(imgThres, sensors)
    sendCommands(senOut, cx, width)
    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    cv2.waitKey(1)
