import cv2 as cv
import mediapipe as mp
import numpy as np
import pygame

# pygame and cube set up
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cube")
clock = pygame.time.Clock()

vertices = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
], dtype=float)

faces = [
    (0,1,2,3),  # back
    (4,5,6,7),  # front
    (0,3,7,4),  # left
    (1,2,6,5),  # right
    (3,2,6,7),  # top
    (0,1,5,4),  # bottom
]


faceColors = [
    (255, 80,  80),   # back   - red
    (80,  255, 80),   # front  - green
    (80,  80,  255),  # left   - blue
    (255, 255, 80),   # right  - yellow
    (255, 80,  255),  # top    - magenta
    (80,  255, 255),  # bottom - cyan
]

# transform matrices
def rotateX(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rotateY(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def project(vertex, fov=500, viewerDistance=5):
    z = viewerDistance + vertex[2]
    x = int(vertex[0] * fov / z + WIDTH / 2)
    y = int(-vertex[1] * fov / z + HEIGHT / 2)
    return (x, y)

def getHandOrientation(handLandmarks):
    wrist = handLandmarks[0]
    midMcp = handLandmarks[9]
    dx = midMcp.x - wrist.x
    dy = midMcp.y - wrist.y
    return dy, dx  # rawX, rawY

def isHandOpen(handLandmarks):
    fingers = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]
    wrist = handLandmarks[0]
    extended = 0
    for tipId, pipId in fingers:
        tip = handLandmarks[tipId]
        pip = handLandmarks[pipId]
        tipDist = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
        pipDist = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
        if tipDist > pipDist:
            extended += 1
    return extended >= 3

# mediapipe setup
modelPath = "models/hand_landmarker.task"
baseOptions = mp.tasks.BaseOptions(model_asset_path=modelPath)
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=baseOptions,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# STOLEN FROM GOOGLE IPYNB
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    return annotated_image
# END STOLEN

capture = cv.VideoCapture(0)

SENSITIVITY = 10

# X axis controlled by left hand
angleX, baseAngleX, refX, wasOpenLeft = 0, 0, None, False
# Y axis controlled by right hand
angleY, baseAngleY, refY, wasOpenRight = 0, 0, None, False

# mother of all helper 
def processHand(hand, handedness, angleX, baseAngleX, refX, wasOpen, angleY, baseAngleY, refY, wasOpenRight):
    rawX, rawY = getHandOrientation(hand)
    handOpen = isHandOpen(hand)
    
    if handedness == "Left":
        if handOpen:
            if not wasOpen:
                refX = rawX
            angleX = baseAngleX + (rawX - refX) * SENSITIVITY
        else:
            if wasOpen:
                baseAngleX = angleX
            refX = None
        wasOpen = handOpen

    elif handedness == "Right":
        if handOpen:
            if not wasOpenRight:
                refY = rawY
            angleY = baseAngleY + (rawY - refY) * -SENSITIVITY
        else:
            if wasOpenRight:
                baseAngleY = angleY
            refY = None
        wasOpenRight = handOpen

    return angleX, baseAngleX, refX, wasOpen, angleY, baseAngleY, refY, wasOpenRight

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        clock.tick(60)
        screen.fill((0, 0, 0))

        _, frame = capture.read()
        frame = cv.flip(frame, 1)

        # mediapipe only handles rgb 
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(capture.get(cv.CAP_PROP_POS_MSEC))
        results = landmarker.detect_for_video(mpImage, timestamp)

        # handles the webcam feed and hands
        if results.hand_landmarks:
            frame = draw_landmarks_on_image(frame, results)
            for idx, hand in enumerate(results.hand_landmarks):
                handedness = results.handedness[idx][0].category_name
                angleX, baseAngleX, refX, wasOpenLeft, angleY, baseAngleY, refY, wasOpenRight = processHand(
                    hand, handedness,
                    angleX, baseAngleX, refX, wasOpenLeft,
                    angleY, baseAngleY, refY, wasOpenRight
                )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        rotated = vertices @ rotateX(angleX).T @ rotateY(angleY).T
        projected = [project(v) for v in rotated]

        faceDepths = []
        for i, face in enumerate(faces):
            avgZ = sum(rotated[v][2] for v in face) / 4
            faceDepths.append((avgZ, i))
        faceDepths.sort(reverse=True)  # furthest first

        # handles the pygame display 
        for _, i in faceDepths:
            face = faces[i]
            points = [projected[v] for v in face]
            pygame.draw.polygon(screen, faceColors[i], points)
            pygame.draw.polygon(screen, (0,0,0), points, 2)  # black outline

        pygame.display.flip()

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow('Cam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

pygame.quit()
capture.release()
cv.destroyAllWindows()