import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("HandMouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()