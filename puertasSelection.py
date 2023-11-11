import shapeRecognition
import numpy as np


p1 = (209, 299)
p2 = (1306, 799)
p3 = (293, 759)
p4 = (1398, 300)


def selction(img, p1, p2, p3, p4):

    Q1 = p1 - p2
    Q2 = p3 - p4

    normalized_q1 = Q1 / np.linalg.norm(Q1)
    normalized_q2 = Q1 / np.linalg.norm(Q2)

    return
