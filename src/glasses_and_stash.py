import numpy as np
import cv2

from utils import CFEVideoConf, image_resize

cap = cv2.VideoCapture(0)

save_path           = 'saved-media/glasses_and_stash.mp4'
frames_per_seconds  = 24
config              = CFEVideoConf(cap, filepath=save_path, res='720p')
out                 = cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)
face_cascade        = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade        = cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')
nose_cascade        = cv2.CascadeClassifier('cascades/third-party/Nose18x15.xml')
glasses             = cv2.imread("images/fun/glasses.png", -1)
mustache            = cv2.imread('images/fun/mustache.png',-1)

while(True):
    ret, frame = cap.read()     # Capture frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)



    # Display the resulting frame
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()