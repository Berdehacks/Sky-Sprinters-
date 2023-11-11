import cv2
import numpy as np
from djitellopy import tello
from qrdet import qrdet
from qreader import QReader

# me = tello.Tello()
# me.connect()
# print(me.get_battery(), "%")
# me.streamon()

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(0)

width, height = 800, 600

qreader = QReader()

while True:
    ret, img = cap.read()
    # img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))

    total = qreader.detect_and_decode(image=img, return_detections=True)
    print(total)
    # if ret_qr:
    #     for point in points:
    #         for corner in point:
    #             img = cv2.circle(img, (int(corner[0]), int(corner[1])), 4, (255, 0, 0), -1)

    #             # if int(corner[0]) < left_up[0] and int(corner[1]) > left_up[1]:
    #             #     left_up = (int(corner[0]), int(corner[1]))

    #             # if int(corner[0]) > right_down[0] and int(corner[1]) < right_down[1]:
    #             #     right_down = (int(corner[0]), int(corner[1]))

    #         # frame = cv2.rectangle(frame, left_up, right_down, (255, 0, 0), -1)

    #             # center_coordinates = (int(corner[0]), int(corner[1]))

    #             # frame = cv2.circle(frame, center_coordinates, 4, (255, 0, 0), 1)

    #             # # Display the resulting frame
    #             # cv2.imshow('frame', frame)

    cv2.imshow('frame', img)

    print("#######################################")
    print(total)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()
