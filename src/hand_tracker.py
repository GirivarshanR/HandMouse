import cv2
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(
    "http://192.168.31.91:8080/video"
)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

screen_width, screen_height = pyautogui.size()

# -------------------------
# Cursor settings
# -------------------------

smoothing = 0.25

previous_x = screen_width // 2
previous_y = screen_height // 2

# Index base (#5) tracking area
x_min = 0.30
x_max = 0.70

y_min = 0.30
y_max = 0.70

# -------------------------
# Gesture settings
# -------------------------

is_left_pinching = False
is_right_pinching = False

pinch_threshold = 0.05
release_threshold = 0.08

print(f"Screen: {screen_width}x{screen_height}")
print("Move index finger base to control cursor.")
print("Thumb + index = left click.")
print("Thumb + middle = right click.")
print("Press Q to quit.")

try:
    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # -------------------------
                # Get landmarks
                # -------------------------

                index_base = hand_landmarks.landmark[5]
                thumb = hand_landmarks.landmark[4]
                index_finger = hand_landmarks.landmark[8]
                middle_finger = hand_landmarks.landmark[12]

                # -------------------------
                # Cursor movement
                # -------------------------

                x = index_base.x
                y = index_base.y

                x = (x - x_min) / (x_max - x_min)
                y = (y - y_min) / (y_max - y_min)

                x = max(0.0, min(x, 1.0))
                y = max(0.0, min(y, 1.0))

                target_x = int(
                    x * (screen_width - 1)
                )

                target_y = int(
                    y * (screen_height - 1)
                )

                current_x = previous_x + (
                    target_x - previous_x
                ) * smoothing

                current_y = previous_y + (
                    target_y - previous_y
                ) * smoothing

                previous_x = current_x
                previous_y = current_y

                pyautogui.moveTo(
                    int(current_x),
                    int(current_y)
                )

                # -------------------------
                # Calculate distances
                # -------------------------

                index_dx = thumb.x - index_finger.x
                index_dy = thumb.y - index_finger.y

                middle_dx = thumb.x - middle_finger.x
                middle_dy = thumb.y - middle_finger.y

                index_distance = (
                    index_dx * index_dx
                    + index_dy * index_dy
                ) ** 0.5

                middle_distance = (
                    middle_dx * middle_dx
                    + middle_dy * middle_dy
                ) ** 0.5

                # -------------------------
                # Left click
                # -------------------------

                if (
                    index_distance < pinch_threshold
                    and index_distance < middle_distance
                    and not is_left_pinching
                ):
                    pyautogui.click(button="left")
                    is_left_pinching = True

                # -------------------------
                # Right click
                # -------------------------

                elif (
                    middle_distance < pinch_threshold
                    and middle_distance < index_distance
                    and not is_right_pinching
                ):
                    pyautogui.click(button="right")
                    is_right_pinching = True

                # -------------------------
                # Release states
                # -------------------------

                if index_distance > release_threshold:
                    is_left_pinching = False

                if middle_distance > release_threshold:
                    is_right_pinching = False

                # -------------------------
                # Draw landmarks
                # -------------------------

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        else:

            is_left_pinching = False
            is_right_pinching = False

        cv2.imshow(
            "HandMouse",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    camera.release()
    hands.close()
    cv2.destroyAllWindows()