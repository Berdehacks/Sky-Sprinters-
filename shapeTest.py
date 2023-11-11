import shapeRecognition
import cv2


# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    shape = shapeRecognition.shapeRecognition(frame, (0, 0), (400, 400))
    print(shape)
