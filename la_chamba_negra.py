import numpy as np
from djitellopy import tello
import cv2

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

me = tello.Tello()
me.connect()
print(me.get_battery(), "%")
me.streamon()

# me.takeoff()
# desired_height = 1  # in centimeters
# me.send_rc_control(0, 0, desired_height, 0)

cap = cv2.VideoCapture(1)

# Thresholding HSV values for light blue
hsvVals = [90, 80, 100, 120, 255, 255]

sensors = 3
threshold = 0.2
width, height = 800, 600
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
        # cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
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

    # me.send_rc_control(lr, fSpeed, 0, curve)

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))

    points_to_follow = []

    imgThres = thresholding(img)
    cx = getContours(imgThres, img)

    initial_angle = 90

    current_row_index = img.shape[0] - 1
    rows_to_skip = 30

    # get beginning point
    for i in range(10):
        ranges = []
        current_row = imgThres[current_row_index]
        filtered_array = np.where(current_row == 255)
        currently_consuming_range = False
        curr_largest_range = 0
        range_used = range(0, 0)
        curr_starting = 0

        for x_i in range(len(current_row)):
            if current_row[x_i] == 255:
                if not currently_consuming_range:
                    currently_consuming_range = True
                    curr_starting = x_i

            else:
                if currently_consuming_range:
                    # curr_largest_range = curr_starting - x_i
                    # range_used = range(curr_starting, x_i)

                    ranges.append(range(curr_starting, x_i))
                    currently_consuming_range = False

        if len(ranges) != 0:
            print(f"points to follow {points_to_follow}")
            if len(points_to_follow) == 0:
                print(ranges)
                valid_range = range(0, 0)
                max_range_distance = max([i.stop - i.start for i in ranges])

                for r in ranges:
                    if r.stop - r.start == max_range_distance:
                        valid_range = r
                        break

                center_x = (valid_range.start + valid_range.stop) // 2

                cv2.circle(img, (center_x, current_row_index), 5, (0, 255, 0), -1)
                points_to_follow.append((center_x, current_row_index))

                # we are moving up in the image Y coord
                current_row_index = current_row_index - rows_to_skip
                current_row = imgThres[current_row_index]
                continue

            else:
                one_is_valid = False

                print(f"ranges: {ranges}")
                for r in ranges:
                    print(f"checking range {r}")
                    center_x = (r.start + r.stop) // 2
                    pos = np.array([center_x, current_row_index])

                    # in case this is NOT the first point we are finding
                    angle_found = angle_between(np.array([points_to_follow[-1][0], points_to_follow[-1][1]]), pos)
                    if angle_found < 5:
                        cv2.circle(img, (center_x, current_row_index), 5, (0, 255, 0), -1)
                        points_to_follow.append(pos)
                        one_is_valid = True

                        # we are moving up in the image Y coord
                        current_row_index = current_row_index - rows_to_skip
                        current_row = imgThres[current_row_index]
                        continue

                # if no valid points was found, just jump to the next row
                if not one_is_valid:
                    current_row_index = current_row_index - rows_to_skip
                    current_row = imgThres[current_row_index]


    # scale_percent = 120 # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)

    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    cv2.waitKey(1)
