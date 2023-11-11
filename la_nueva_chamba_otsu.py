import numpy as np
from djitellopy import tello
import cv2


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


me = tello.Tello()
me.connect()
# print(me.get_battery(), "%")
me.streamon()


me.takeoff()
desired_height = -20  # in centimeters
me.send_rc_control(0, 0, desired_height, 0)

cap = cv2.VideoCapture(1)

# # Thresholding HSV values for light blue
# hsvVals = [90, 80, 100, 140, 255, 255]
# # hsvVals = [215, 90, 60, 130, 255, 255]
# last_distance = 0


# Define maximum speed and maximum YAW angle change
MS = 30  # maximum speed
MY = 30  # maximum YAW angle change


def thresholding(img):
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    # upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    # mask = cv2.inRange(hsv, lower, upper)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 140, 180, cv2.THRESH_OTSU)

    return threshold


def getContours(imgThres, img):
    cx = 0
    contours, hieracrhy = cv2.findContours(
        imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2
        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
        # cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return cx

# Function to get the drone's heading angle from optical flow


def get_drone_heading_angle(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract horizontal and vertical components of flow
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # Calculate the average flow direction
    avg_flow_x = np.mean(flow_x)
    avg_flow_y = np.mean(flow_y)

    # Calculate the heading angle
    heading_angle_rad = np.arctan2(avg_flow_y, avg_flow_x)
    heading_angle_deg = np.degrees(heading_angle_rad)

    # Ensure the angle is in the range [0, 360)
    heading_angle_deg = (heading_angle_deg + 360) % 360

    return heading_angle_deg


# Initial frame
prev_frame = me.get_frame_read().frame

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))

    points_to_follow = []

    imgThres = thresholding(img)
    cx = getContours(imgThres, img)

    initial_angle = 90

    current_row_index = img.shape[0] - 1
    rows_to_skip = 30

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
                    ranges.append(range(curr_starting, x_i))
                    currently_consuming_range = False

        if len(ranges) != 0:
            if len(points_to_follow) == 0:
                valid_range = range(0, 0)
                max_range_distance = max([i.stop - i.start for i in ranges])

                for r in ranges:
                    if r.stop - r.start == max_range_distance:
                        valid_range = r
                        break

                center_x = (valid_range.start + valid_range.stop) // 2

                cv2.circle(img, (center_x, current_row_index),
                           5, (0, 255, 0), -1)
                points_to_follow.append((center_x, current_row_index))

                current_row_index = current_row_index - rows_to_skip
                current_row = imgThres[current_row_index]
                continue

            else:
                one_is_valid = False

                for r in ranges:
                    center_x = (r.start + r.stop) // 2
                    pos = np.array([center_x, current_row_index])

                    angle_found = angle_between(
                        np.array([points_to_follow[-1][0], points_to_follow[-1][1]]), pos)
                    if angle_found < 5:
                        cv2.circle(img, (center_x, current_row_index),
                                   5, (0, 255, 0), -1)
                        points_to_follow.append(pos)
                        one_is_valid = True

                        current_row_index = current_row_index - rows_to_skip
                        current_row = imgThres[current_row_index]
                        continue

                if not one_is_valid:
                    current_row_index = current_row_index - rows_to_skip
                    current_row = imgThres[current_row_index]

        chosen_point_index = 6
        pure_pursuit_row = img.shape[0] - (rows_to_skip * 5)
        rotation_speed = 0

        if len(points_to_follow) > chosen_point_index:
            img = cv2.circle(
                img, points_to_follow[chosen_point_index], 10, (0, 0, 255), -1)
            img = cv2.circle(
                img, (width // 2, pure_pursuit_row), 10, (0, 255, 0), -1)

            center = np.array([width // 2, pure_pursuit_row])
            distance = points_to_follow[chosen_point_index][0] - center[0]

            k = 50

            if np.abs(distance) > 20:
                rotation_speed = (distance / (width // 2)) * k

        # alignment_movement = 0
        # alignment_movement_speed = 0.1
        # alignment_delta_movement_speed = 0.25
        # distance = 0
        # max_acceptable_error = 100
        # proportional_speed = 0

        # # print(last_distance)
        # if len(points_to_follow) > 1:
        #     x_coordinates = np.array([angle_between(points_to_follow[i - 1], points_to_follow[i]) for i in range(1, len(points_to_follow))])
        #     x_std_dev = np.average(x_coordinates)
        #     print(x_std_dev)
        #     if x_std_dev > 0.1:
        #         proportional_speed = x_std_dev
        # else:
        #     x_std_dev = 0

        # min_starting_speed = 5
        # speed = min_starting_speed + MS * proportional_speed
        # # print(x_std_dev)

        # if len(points_to_follow) != 0:
        #     distance = points_to_follow[0][0] - (width // 2)
        #     delta_distance = last_distance - distance
        #     # print(delta_distance)

        #     if abs(distance) > max_acceptable_error:
        #         alignment_movement = alignment_movement_speed * distance + alignment_delta_movement_speed * delta_distance

        #         speed = 0
        #         x_std_dev = 0

        # last_distance = distance

        # if len(points_to_follow) > 1:
        #     current_frame = me.get_frame_read().frame
        #     drone_heading_angle = get_drone_heading_angle(prev_frame, current_frame) # Calculate the heading angle based on your drone's orientation data
        #     path_heading_angle = angle_between(points_to_follow[0], points_to_follow[-1])
        #     perpendicular_yaw = path_heading_angle - drone_heading_angle

        #     perpendicular_yaw = (perpendicular_yaw + 360) % 360

        #     if perpendicular_yaw >= 180:
        #         perpendicular_yaw -= 360

        #     yaw = MY * (perpendicular_yaw / 180)

        # me.send_rc_control(int(alignment_movement), int(speed), 0, int(x_std_dev * 0.9))
        me.send_rc_control(0, 35, 0, int(rotation_speed))

        cv2.imshow("Output", img)
        # cv2.imshow("Path", imgThres)
        cv2.waitKey(1)
