import os
import numpy as np
import cv2

filename = 'video.avi'      # .avi or .mp4
frames_per_seconds = 24.0
res = '480p'        # 720p or 1080p

# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Grab resolution dimensions and set video capture to it
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)      # Change the current caputre device to the resulting resolution
    return width, height

# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

cap = cv2.VideoCapture(0)       # Select Default Camera
out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_seconds, get_dims(cap, res))       #dims = width, height

while True:
    ret, frame = cap.read()     # Capture every frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame', frame)  # Display the resulting frame (imgshow)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()       # When everything is done, release the capture
out.release()
cv2.destroyAllWindows()