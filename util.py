from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def getAudioDevice():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    return volume

def formatLandmarks(landmarks, image, mpDrawing, mpHands, dotColor, connectionColor):
    l = []

    for handLandmark in landmarks:

        for id, mark in enumerate(handLandmark.landmark):
            # Draws the detection on the image
            mpDrawing.draw_landmarks(
                image, handLandmark, mpHands.HAND_CONNECTIONS,
                mpDrawing.DrawingSpec(color=dotColor, thickness=2, circle_radius=4),
                mpDrawing.DrawingSpec(color=connectionColor, thickness=2, circle_radius=2)
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
    
    return l