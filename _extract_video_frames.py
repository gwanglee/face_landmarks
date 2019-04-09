import os
import cv2

SRC_DIR='/Users/gglee/Data/Landmark/test_video'
SRC_FILENAME='landmark_test.mov'

BASENAME = os.path.splitext(SRC_FILENAME)[0]
DST_DIR=os.path.join(SRC_DIR, BASENAME)
os.mkdir(DST_DIR)

cap = cv2.VideoCapture(os.path.join(SRC_DIR, SRC_FILENAME))
cnt = 0

while True:
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    else:
        cv2.imwrite(os.path.join(DST_DIR, '%06d.jpg' % cnt), frame)
        cv2.imshow('frame', frame)
        cnt += 1