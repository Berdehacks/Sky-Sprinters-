import shapeRecognition
import cv2


# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    shape = shapeRecognition.shapeRecognition(frame, (0, 0), (800, 800))

    cropped = frame[0:400, 0:400]
    cv2.imshow('frame', cropped)

    print(shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
