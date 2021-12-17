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

                    # Blur before drawing the box
                    image = blurPart(image, newBox)

                    # Draw the scaled box with cv2.rectangle
                    cv2.rectangle(image, newBox, (0, 255, 0), 1)

                    cv2.putText(
                        image, f'Score: {detection.score[0]:.2f}',
                        (newBox[0], newBox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

            cv2.imshow('Face blurring', image)

            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break

# There seems to be having a little issue when getting target area
def blurPart(img, newBox):

    topLeft = (newBox[0], newBox[1])
    rightBottom = (newBox[2], newBox[3])

    x, y = topLeft[0], topLeft[1]
    w, h = rightBottom[0], rightBottom[1]
    
    # Cut out the target part
    ROI = img[y:y + h, x:x + w]
    blur = cv2.GaussianBlur(ROI, (0, 0), sigmaX=20, sigmaY=20)
    
    # Put it back in
    img[y:y + h, x:x + w] = blur

    return img

if (__name__ == '__main__'):
    main()