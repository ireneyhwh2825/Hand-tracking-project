import cv2
import mediapipe as mp
import time

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# MediaPipe hand setup
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Drawing utility
mpDraw = mp.solutions.drawing_utils

# FPS variables
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    # Skip loop if frame not captured
    if not success or img is None:
        print("Camera frame not captured")
        continue

    # Convert image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(imgRGB)

    # If hands detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Print landmark coordinates
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                #if id == 4:
                cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    # Display FPS
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Show image
    cv2.imshow("Image", img)

    # Exit with ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
#cv2.destroyAllWindows()