import numpy as np
from djitellopy import tello
import time
import cv2

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

me = tello.Tello()
me.connect()
print(me.get_battery(), "%")
me.streamon()

last_points_to_consider = 3
last_few_points_found = np.zeros((last_points_to_consider, 2))
points_not_found = last_points_to_consider

me.takeoff()
desired_height = -10  # in centimeters
me.send_rc_control(0, 0, desired_height, 0)

cap = cv2.VideoCapture(1)

# Thresholding HSV values for light blue

hsvVals = [90, 80, 100, 140, 255, 255]
#hsvVals = [215, 90, 60, 130, 255, 255]
#hsvVals = [100, 50, 50, 120, 255, 255]  #4pm
#hsvVals = [140, 40, 40, 135, 255, 255]

last_distance = 0

past_values_to_take = 3
differentials_values = np.zeros(past_values_to_take)
global_counter = 0
sensors = 3
threshold = 0.2
width, height = 800, 600
senstivity = 3
weights = [-25, -15, 0, 15, 25]
fSpeed = 20
curve = 0

# Define maximum speed and maximum YAW angle change
MS = 30  # maximum speed
MY = 25  # maximum YAW angle change

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

# Function to get the drone's heading angle from optical flow
def get_drone_heading_angle(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

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

def canny(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Initial frame
prev_frame = me.get_frame_read().frame

while True:
    print("processing")

    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))

    points_to_follow = []

    imgThres = thresholding(img)
    # Apply Canny edge detection to the masked image
    edges = canny(img)
    cx = getContours(edges, img)

    initial_angle = 90

    current_row_index = img.shape[0] - 1
    rows_to_skip = 40

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
                    if r.stop - r.start < 50:
                        if r.stop - r.start == max_range_distance:
                            valid_range = r
                            break

                center_x = (valid_range.start + valid_range.stop) // 2

                cv2.circle(img, (center_x, current_row_index), 5, (0, 255, 0), -1)
                points_to_follow.append((center_x, current_row_index))

                current_row_index = current_row_index - rows_to_skip
                current_row = imgThres[current_row_index]
                continue

            else:
                one_is_valid = False

                for r in ranges:
                    center_x = (r.start + r.stop) // 2
                    pos = np.array([center_x, current_row_index])

                    angle_found = angle_between(np.array([points_to_follow[-1][0], points_to_follow[-1][1]]), pos)
                    if angle_found < 13:
                        cv2.circle(img, (center_x, current_row_index), 5, (0, 255, 0), -1)
                        points_to_follow.append(pos)
                        one_is_valid = True

                        current_row_index = current_row_index - rows_to_skip
                        current_row = imgThres[current_row_index]
                        continue

                if not one_is_valid:
                    current_row_index = current_row_index - rows_to_skip
                    current_row = imgThres[current_row_index]

    movement_speed = 27
    rotation_speed = 0
    chosen_point_index = 4

    if len(points_to_follow) > chosen_point_index: # si hay un punto valido en este frame, deberiamos agregarlo
        last_few_points_found[global_counter] = points_to_follow[chosen_point_index]
        points_not_found = points_not_found + 1 if points_not_found > 5 else 5
    else:
        last_few_points_found[global_counter] = np.array([width // 2, 0])
        points_not_found = points_not_found - 1 if points_not_found < 0 else 0

    # no necesitamos un promedio de los dos ejes, solo el x, deberiamos cambiarlo
    average_point_to_follow = np.mean(last_few_points_found, axis=0)

    # si tenemos un promedio valido, hacer la aproximacion de rotacion
    if points_not_found > 0:
        print("average x value is", average_point_to_follow[0])
        pure_pursuit_row = img.shape[0] - (rows_to_skip * 5)

        img = cv2.circle(img, (int(average_point_to_follow[0]), pure_pursuit_row), 10, (0, 0, 255), -1)
        img = cv2.circle(img, (width // 2, pure_pursuit_row), 10, (0, 255, 0), -1)

        center = np.array([width // 2, pure_pursuit_row])
        distance = average_point_to_follow[0] - center[0]
        print("distance", distance)

        differentials_values[global_counter] = distance

        k = 0.25
        ki = 0.06

        if np.abs(distance) > 20: # el error es demasiado grande entonces debemos aproximar una rotacion
            rotation_speed = (distance / (width // 2)) * k + sum(differentials_values) * ki
            # si rotamos, aun queremos movernos un poco, no estoy seguro si el 1 / distance esta totalmente correcto, segun yo si 
            # porque queremos que la velocidad sea inversamente proporcional al error
            # movement_speed = (1 / (distance / (width // 2))) * 25

    else:
        print("no points found, searching")
        movement_speed = 0
        rotation_speed = 40

    # esto esta mal, deberian ser dos counters, no uno para ambas variables
    global_counter += 1
    global_counter %= past_values_to_take

    me.send_rc_control(0, int(movement_speed), 0, int(rotation_speed))

    cv2.imshow("Output", img)
    # cv2.imshow("Path", imgThres)
    cv2.waitKey(1)
    time.sleep(0.1)
