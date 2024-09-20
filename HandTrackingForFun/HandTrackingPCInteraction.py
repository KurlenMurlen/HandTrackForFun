import cv2
import mediapipe as mp
import time
import math
import pyautogui
import numpy as np

# Screen dimensions for the mapping purposes of the code
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands(False)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
dragging = False  # Creating a state of dragging, to be able to use the mouse more accordingly
double_clicking = False  # Adding another statement to be able to double click more effectively without activating the dragging motion
previous_pinky_y = 0  # To track the vertical position of the pinky finger so you can scroll up and down

while True:
    success, img = cap.read()

    # Mirrorring the image for better visualization
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Coordinates for the specific markings, focusing on the index finger
            landmarks = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))
                # Draws a bigger circle at the index finger tip for visualization
                if id == 8:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

            # Drawing the landmarks
            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

            # Control mouse with the index finger (id=8)
            if len(landmarks) >= 21:
                index_tip = landmarks[8]

                # Mapping the hand position in parallel to the screen size
                screen_x = np.interp(index_tip[0], (0, w), (0, screen_width))
                screen_y = np.interp(index_tip[1], (0, h), (0, screen_height))

                # Moving the mouse accordingly to the movements defined just now (51-52)
                pyautogui.moveTo(screen_x, screen_y)

                # Checking for the pinch as gesture
                thumb_tip = landmarks[4]
                distance_thumb_index = math.sqrt((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2)

                # If the distance is small, consider it a pinch (drag)
                if distance_thumb_index < 30:
                    if not dragging:  # Start dragging if not already dragging
                        pyautogui.mouseDown()
                        dragging = True
                else:
                    if dragging:  # Stop dragging when the fingers are apart 
                        pyautogui.mouseUp()
                        dragging = False

                # Checking for ring finger and thumb touch for double-click
                ring_tip = landmarks[16]
                distance_thumb_ring = math.sqrt((thumb_tip[0] - ring_tip[0]) ** 2 + (thumb_tip[1] - ring_tip[1]) ** 2)

                if distance_thumb_ring < 30:
                    if not double_clicking:  # Perform double-click if not already done
                        pyautogui.doubleClick()
                        double_clicking = True
                else:
                    double_clicking = False  # Reset state if fingers are apart

                # Check for pinky movement to control scrolling
                pinky_tip = landmarks[20]
                stationary_fingers = True
                
                # Ensure the other fingers are relatively stationary
                for fingertip_id in [4, 8, 12, 16]:  # Thumb, index, middle, ring tips
                    _, fingertip_y = landmarks[fingertip_id]
                    if abs(fingertip_y - landmarks[fingertip_id][1]) > 20:  # Tolerance to detect stationary fingers
                        stationary_fingers = False
                        break

                # If other fingers are stationary, detect pinky movement
                if stationary_fingers:
                    current_pinky_y = pinky_tip[1]  # Get pinky's y-coordinate
                    if previous_pinky_y != 0:
                        # Determine direction of movement
                        if current_pinky_y < previous_pinky_y - 20:  # Moving up
                            pyautogui.scroll(5)
                        elif current_pinky_y > previous_pinky_y + 20:  # Moving down
                            pyautogui.scroll(-5)
                    previous_pinky_y = current_pinky_y
                else:
                    previous_pinky_y = 0  # Reset if other fingers are not stationary

    # FPS calculation to display on camera (on my notebook it gets a tad lil bit laggy, I accept suggestions to make it run smoother(the notebook sucks))
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Displaying the fps
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Showing the picture
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window, shutting down the program
cap.release()
cv2.destroyAllWindows()

