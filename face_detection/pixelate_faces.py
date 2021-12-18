import cv2
import mediapipe as mp

EXIT_KEY = 'q'

def main():
    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    with mpFaceDetector.FaceDetection(min_detection_confidence=0.7) as faceDetection:
        
        while (cap.isOpened()):
            success, frame = cap.read()

            # Camera's width & height
            width  = int(cap.get(3))
            height = int(cap.get(4))

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert the color for the program to process
            # Since cv2 uses BGR and mediapipe uses RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(image)

            # Convert it back for displaying after processing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if (results.detections):
                for id, detection in enumerate(results.detections):
                    # Draw the box around the face with built-in function
                    # mpDraw.draw_detection(image, detection)

                    # The box around the face
                    box = detection.location_data.relative_bounding_box
                    
                    # Scale the box with current cap's width and height
                    newBox = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)

                    # Pixelate faces
                    image = pixelateFaces(image, newBox)

                    # Draw the scaled box with cv2.rectangle
                    cv2.rectangle(image, newBox, (0, 255, 0), 1)

                    cv2.putText(
                        image, f'Score: {detection.score[0]:.2f}',
                        (newBox[0], newBox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

            cv2.imshow('Pixelate faces', image)

            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break

def pixelateFaces(img, newBox):

    topLeft = (newBox[0], newBox[1])
    rightBottom = (newBox[2], newBox[3])

    x, y = topLeft[0], topLeft[1]
    w, h = rightBottom[0], rightBottom[1]
    
    # Cut out the target part
    ROI = img[y:y + h, x:x + w]
    roiH, roiW, _ = ROI.shape

    # Pixelate it by resizing it to NxN
    PIXELATE_SIDE = 8
    temp = cv2.resize(
        ROI,
        (PIXELATE_SIDE, PIXELATE_SIDE),
        interpolation=cv2.INTER_LINEAR
    )

    # Resize it back to the original size
    ROI = cv2.resize(temp, (roiH, roiW), interpolation=cv2.INTER_NEAREST)

    # Put it back into the original image
    try:
        img[y:y + h, x:x + w] = ROI
    
    except ValueError:
        pass

    # Why a try except & pass?
    # -
    # The following error will be thrown when the face is not INSIDE the image
    # E.g In the image's border
    """
    Traceback (most recent call last):
    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/pixelate_faces.py", line 84, in <module>
        main()
    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/pixelate_faces.py", line 43, in main
        image = pixelateFaces(image, newBox)
    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/pixelate_faces.py", line 79, in pixelateFaces
        img[y:y + h, x:x + w] = ROI
    ValueError: could not broadcast input array from shape (183,182,3) into shape (182,183,3)
    [ WARN:0] global D:\a\opencv-python\opencv-python\opencv\modules\videoio\src\cap_msmf.cpp (438) `anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback
    """

    return img

if (__name__ == '__main__'):
    main()