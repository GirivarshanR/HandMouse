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
# Cursor smoothing
# -------------------------

smoothing = 0.25

previous_x = screen_width // 2
previous_y = screen_height // 2

# -------------------------
# Cursor tracking area
# -------------------------
# The index base (#5) movement inside this
# camera range is mapped to the entire screen.

x_min = 0.30
x_max = 0.70

y_min = 0.30
y_max = 0.70

# -------------------------
# Pinch settings
# -------------------------

is_pinching = False

pinch_threshold = 0.05
release_threshold = 0.08

print(f"Screen: {screen_width}x{screen_height}")
print("Move the base of your index finger to control the cursor.")
print("Pinch your thumb and index finger to click.")
print("Press Q to quit.")

try:
    while True:

        success, frame = camera.read()

        if not success:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
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
                index_tip = hand_landmarks.landmark[8]
                thumb = hand_landmarks.landmark[4]

                # -------------------------
                # Use index base (#5)
                # as cursor tracker
                # -------------------------

                x = index_base.x
                y = index_base.y

                # -------------------------
                # Map camera range
                # to entire screen
                # -------------------------

                x = (x - x_min) / (x_max - x_min)
                y = (y - y_min) / (y_max - y_min)

                # Keep inside 0-1
                x = max(0.0, min(x, 1.0))
                y = max(0.0, min(y, 1.0))

                # Convert to screen coordinates
                target_x = int(
                    x * (screen_width - 1)
                )

                target_y = int(
                    y * (screen_height - 1)
                )

                # -------------------------
                # Cursor smoothing
                # -------------------------

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
                # Pinch detection
                # -------------------------

                dx = thumb.x - index_tip.x
                dy = thumb.y - index_tip.y

                distance = (
                    dx * dx + dy * dy
                ) ** 0.5

                if (
                    distance < pinch_threshold
                    and not is_pinching
                ):
                    pyautogui.click()
                    is_pinching = True

                elif distance > release_threshold:
                    is_pinching = False

                # -------------------------
                # Draw landmarks
                # -------------------------

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        else:
            is_pinching = False

        # -------------------------
        # Display camera
        # -------------------------

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