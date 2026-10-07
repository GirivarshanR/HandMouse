
import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

screen_width, screen_height = pyautogui.size()

print(f"Screen: {screen_width}x{screen_height}")
print("Move your index finger to control the cursor.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                index_finger = hand_landmarks.landmark[8]

                x = index_finger.x
                y = index_finger.y

                mouse_x = max(
                    1, min(int(x * screen_width), screen_width - 2)
                )
                mouse_y = max(
                    1, min(int(y * screen_height), screen_height - 2)
                )

                # Move the actual Windows cursor
                pyautogui.moveTo(mouse_x, mouse_y)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("HandMouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()
