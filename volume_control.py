from math import hypot

import cv2
import mediapipe as mp
import numpy as np

from util import getAudioDevice

"""
    Volume control by hand (Tip of thumb + index finger)
"""

# https://google.github.io/mediapipe/images/mobile/hand_landmarks.png

# Order: Blue, Green, Red
DOT_COLOR = (0, 0, 255)
CONNECTION_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 0)

EXIT_KEY = 'q'

# Don't change
THUMB_TIP = 4
INDEX_TIP = 8

def main():
    mpDrawing = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands

    volume = getAudioDevice()
    volMin, volMax = volume.GetVolumeRange()[::2]

    cap = cv2.VideoCapture(0)

    with mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:

        while (cap.isOpened()):
            success, frame = cap.read()

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            image, results = detectHands(hands, frame)

            l = []
            landmarks = results.multi_hand_landmarks
            if (landmarks):

                for handLandmark in landmarks:

                    for id, mark in enumerate(handLandmark.landmark):
                        # Draws the detection on the image
                        mpDrawing.draw_landmarks(
                            image, handLandmark, mpHands.HAND_CONNECTIONS,
                            mpDrawing.DrawingSpec(color=DOT_COLOR, thickness=2, circle_radius=4),
                            mpDrawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2, circle_radius=2)
                        )

                        height, width, ignored = image.shape
                        cx, cy = int(mark.x * width), int(mark.y * height)

                        l.append(
                            {
                                'id': id,
                                'cx': cx,
                                'cy': cy
                            }
                        )

            if (l != []):
                vol = setVolume(l, volume, volMin, volMax)
                drawLine(image, l)
                addText(image, l, vol)

            cv2.imshow('Hand tracking', image)

            # Exit if user pressed 'q'
            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break

    print("--- End of Program ---")

def addText(image, landmarks, vol):
    x1, y1 = landmarks[THUMB_TIP]['cx'], landmarks[THUMB_TIP]['cy']
    x2, y2 = landmarks[INDEX_TIP]['cx'], landmarks[INDEX_TIP]['cy']

    midPoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    cv2.putText(image, f'Volume: {vol:.2f}', midPoint, cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 1, cv2.LINE_AA)

def drawLine(image, landmarks):
    """
        Draw line between thumb's tip and index finger's tip
    """
    x1, y1 = landmarks[THUMB_TIP]['cx'], landmarks[THUMB_TIP]['cy']
    x2, y2 = landmarks[INDEX_TIP]['cx'], landmarks[INDEX_TIP]['cy']

    # cv2.line(image, start_point, end_point, color, thickness)
    cv2.line(image, (x1, y1), (x2, y2), LINE_COLOR, 3)

def setVolume(landmarks, volume, volMin, volMax):
    
    x1, y1 = landmarks[THUMB_TIP]['cx'], landmarks[THUMB_TIP]['cy']
    x2, y2 = landmarks[INDEX_TIP]['cx'], landmarks[INDEX_TIP]['cy']


    distance = hypot(x2 - x1, y2 - y1)
    v = np.interp(
        distance,
        [15, 220],
        [volMin, volMax]
    )

    """
    # This we don't check the value of v, it will cause error when setting the volume
    if ((v <= volMin) or (v >= volMax)):
        print('[WARNING] Boundary reached! Please adjust distance between the hand and camera!')
        return
    """

    print(f'volMin = {volMin} | volMax = {volMax} | v = {v}')
    try:
        volume.SetMasterVolumeLevel(v, None)
    except:
        # pass
        print('[WARNING] Boundary reached! Please adjust distance between the hand and camera!')
    
    return v

def detectHands(hands, frame):
    # Recoloring the frame read from webcam
    # Default 'image' get from cv2 is in BGR
    # Convert it to RGB is for mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # Change it back to BGR after the process
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

if (__name__ == '__main__'):
    main()