# import the opencv library
import cv2


# define a video capture object
vid = cv2.VideoCapture(0)
qr = cv2.QRCodeDetector()


while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    ret_qr, points = qr.detect(frame)

    print(points)

    for corner in points[0]:

        frame = cv2.circle(frame, center_coordinates, radius, color, thickness)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

[[[ 427.        348.      ]
  [ 208.        368.      ]
  [ 111.624916 -107.36884 ]
  [ 637.        102.      ]]]
