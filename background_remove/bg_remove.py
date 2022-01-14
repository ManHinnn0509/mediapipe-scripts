import cv2
import numpy as np
import mediapipe as mp

# This script isn't perfect

EXIT_KEY = 'q'

def main():
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfieSegmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    bgPath = "./img/bg/bg.png"
    bg = readImage(bgPath)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        height, width, channel = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bg = cv2.resize(bg, (width, height))

        # ???
        results = selfieSegmentation.process(frame)
        mask = results.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.7

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        resultFrame = np.where(condition, frame, bg)
        
        cv2.imshow("Title", resultFrame)

        key = cv2.waitKey(1)
        if (key == ord(EXIT_KEY)):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def readImage(p: str):
    try:
        img = cv2.imread(p)
        return img

    except:
        return None

if (__name__ == '__main__'):
    main()