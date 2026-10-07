import cv2
import mediapipe as mp
import pyautogui
import time

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
is_dragging = False

pinch_start_time = 0

# Position when pinch begins
click_x = 0
click_y = 0

pinch_threshold = 0.035
release_threshold = 0.065

drag_hold_time = 0.5

print(f"Screen: {screen_width}x{screen_height}")
print("Move index finger base to control cursor.")
print("Quick thumb + index pinch = left click.")
print("Hold thumb + index pinch = drag.")
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

                index_base = hand_landmarks.landmark[13]
                thumb = hand_landmarks.landmark[4]
                index_finger = hand_landmarks.landmark[8]
                middle_finger = hand_landmarks.landmark[12]

                # -------------------------
                # Cursor position
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

                # -------------------------
                # Calculate pinch distances
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
                # Start left pinch
                # -------------------------

                if (
                    not is_left_pinching
                    and not is_right_pinching
                    and index_distance < pinch_threshold
                    and index_distance < middle_distance
                ):

                    is_left_pinching = True

                    pinch_start_time = time.time()

                    # Save where the click started
                    click_x = int(current_x)
                    click_y = int(current_y)

                # -------------------------
                # Left pinch is active
                # -------------------------

                if is_left_pinching:

                    pinch_duration = (
                        time.time() - pinch_start_time
                    )

                    # Become drag after hold time
                    if (
                        pinch_duration >= drag_hold_time
                        and not is_dragging
                    ):

                        pyautogui.mouseDown(
                            button="left"
                        )

                        is_dragging = True

                    # Release pinch
                    if index_distance > release_threshold:

                        if is_dragging:

                            pyautogui.mouseUp(
                                button="left"
                            )

                            is_dragging = False

                        else:

                            # Quick pinch = click
                            pyautogui.click(
                                x=click_x,
                                y=click_y,
                                button="left"
                            )

                        is_left_pinching = False

                # -------------------------
                # Normal cursor movement
                # -------------------------

                pyautogui.moveTo(
                    int(current_x),
                    int(current_y)
                )

                # -------------------------
                # Right click
                # -------------------------

                if (
                    not is_left_pinching
                    and not is_dragging
                    and middle_distance < pinch_threshold
                    and middle_distance < index_distance
                    and not is_right_pinching
                ):

                    pyautogui.click(
                        button="right"
                    )

                    is_right_pinching = True

                # -------------------------
                # Release right pinch
                # -------------------------

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

            # Safety release
            if is_dragging:

                pyautogui.mouseUp(
                    button="left"
                )

            is_left_pinching = False
            is_right_pinching = False
            is_dragging = False

        cv2.imshow(
            "HandMouse",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    if is_dragging:

        pyautogui.mouseUp(
            button="left"
        )

    camera.release()
    hands.close()
    cv2.destroyAllWindows()