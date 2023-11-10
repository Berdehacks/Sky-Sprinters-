import numpy as np
from djitellopy import tello
import cv2

drone = tello.Tello()
drone.connect()

print(drone.get_battery())
drone.streamon()
cap = cv2.VideoCapture(1)

width, height = 480, 360

qr = cv2.QRCodeDetector()


while 1:
    img = drone.get_frame_read().framew
    img = cv2.resize(img, (width, height))
    ret_qr, points = qr.detect(img)

    print(ret_qr)
    # cv2.circle(img,)

    cv2.imshow("Output", img)
