import cv2
import mediapipe as mp
import pyautogui
import math
import time

pyautogui.PAUSE = 0

camera = cv2.VideoCapture("http://192.168.31.91:8080/video")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_width, screen_height = pyautogui.size()

previous_x = None
previous_y = None

smoothing = 0.25

x_min = 0.30
x_max = 0.70
y_min = 0.30
y_max = 0.70

# Gestures
left_pinching = False
right_pinching = False
dragging = False

left_pinch_start = None
right_pinch_start = None

pinch_threshold = 0.035
release_threshold = 0.065
drag_hold_time = 0.5

# Scroll
is_scrolling = False

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]
        landmarks = hand.landmark

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Cursor
        cursor = landmarks[13]

        x = (cursor.x - x_min) / (x_max - x_min)
        y = (cursor.y - y_min) / (y_max - y_min)

        x = max(0, min(1, x))
        y = max(0, min(1, y))

        target_x = int(x * (screen_width - 1))
        target_y = int(y * (screen_height - 1))

        if previous_x is None:
            previous_x = target_x
            previous_y = target_y

        smooth_x = previous_x + (target_x - previous_x) * smoothing
        smooth_y = previous_y + (target_y - previous_y) * smoothing

        # Pinch distances
        thumb_tip = landmarks[4]
        index_finger = landmarks[8]
        middle_finger = landmarks[12]

        index_distance = math.sqrt(
            (thumb_tip.x - index_finger.x) ** 2 +
            (thumb_tip.y - index_finger.y) ** 2
        )

        middle_distance = math.sqrt(
            (thumb_tip.x - middle_finger.x) ** 2 +
            (thumb_tip.y - middle_finger.y) ** 2
        )

        # Finger positions
        index_joint = landmarks[6]
        middle_joint = landmarks[10]

        ring_finger = landmarks[16]
        ring_joint = landmarks[14]

        pinky_finger = landmarks[20]
        pinky_joint = landmarks[18]

        index_extended = index_finger.y < index_joint.y
        middle_extended = middle_finger.y < middle_joint.y

        ring_folded = ring_finger.y > ring_joint.y
        pinky_folded = pinky_finger.y > pinky_joint.y

        finger_gap = math.sqrt(
            (index_finger.x - middle_finger.x) ** 2 +
            (index_finger.y - middle_finger.y) ** 2
        )

        thumb_is_clear = (
            index_distance > release_threshold and
            middle_distance > release_threshold
        )

        index_bend = math.sqrt(
            (index_finger.x - index_joint.x) ** 2 +
            (index_finger.y - index_joint.y) ** 2
        )

        middle_bend = math.sqrt(
            (middle_finger.x - middle_joint.x) ** 2 +
            (middle_finger.y - middle_joint.y) ** 2
        )

        # Left click and drag
        if index_distance < pinch_threshold:

            if not left_pinching:
                left_pinching = True
                left_pinch_start = time.time()

        elif index_distance > release_threshold:

            if left_pinching:

                pinch_duration = time.time() - left_pinch_start

                if dragging:

                    pyautogui.mouseUp()
                    dragging = False

                elif pinch_duration < drag_hold_time:

                    pyautogui.click(
                        int(smooth_x),
                        int(smooth_y)
                    )

                left_pinching = False
                left_pinch_start = None

        if left_pinching and not dragging:

            if time.time() - left_pinch_start >= drag_hold_time:

                pyautogui.moveTo(
                    int(smooth_x),
                    int(smooth_y)
                )

                pyautogui.mouseDown()
                dragging = True

        # Right click
        if (
            not left_pinching
            and not dragging
            and middle_distance < pinch_threshold
            and middle_distance < index_distance
        ):

            if not right_pinching:

                pyautogui.click(
                    int(smooth_x),
                    int(smooth_y),
                    button="right"
                )

                right_pinching = True

        elif middle_distance > release_threshold:

            right_pinching = False

        # Scroll
        scroll_gesture = (
            not left_pinching
            and not dragging
            and not right_pinching
            and ring_folded
            and pinky_folded
            and finger_gap < 0.07
            and thumb_is_clear
        )

        if scroll_gesture:

            is_scrolling = True

            if index_extended and middle_extended:

                pyautogui.scroll(1)

            elif index_bend < 0.10 and middle_bend < 0.10:

                pyautogui.scroll(-1)

        else:

            is_scrolling = False

        # Cursor position
        if not is_scrolling:

            pyautogui.moveTo(
                int(smooth_x),
                int(smooth_y)
            )

        previous_x = smooth_x
        previous_y = smooth_y

    else:

        if dragging:
            pyautogui.mouseUp()

        left_pinching = False
        right_pinching = False
        dragging = False

        left_pinch_start = None
        right_pinch_start = None

        is_scrolling = False

    cv2.imshow("HandMouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()