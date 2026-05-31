import cv2


# ----------------------------
# Configuration
# ----------------------------
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
EXIT_KEY = 27  # ESC key


def load_cascades():
    """
    Load Haar Cascade classifiers for face and eye detection.

    Returns:
        tuple: (face_cascade, eye_cascade)
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade.")

    if eye_cascade.empty():
        raise RuntimeError("Failed to load eye cascade.")

    return face_cascade, eye_cascade


def detect_and_draw_faces(frame, face_cascade, eye_cascade):
    """
    Detect faces and eyes in a frame and draw rectangles around them.

    Args:
        frame: Input BGR image frame.
        face_cascade: Face detector.
        eye_cascade: Eye detector.

    Returns:
        frame: Annotated frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS
    )

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

        # Region of interest (face area)
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Detect eyes inside the detected face
        eyes = eye_cascade.detectMultiScale(face_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 127, 255),
                2
            )

    return frame


def main():
    """
    Main application loop.
    """
    face_cascade, eye_cascade = load_cascades()

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to capture frame.")
                break

            frame = detect_and_draw_faces(
                frame,
                face_cascade,
                eye_cascade
            )

            cv2.imshow("Face and Eye Detection", frame)

            # Exit when ESC is pressed or window is closed
            if cv2.waitKey(1) & 0xFF == EXIT_KEY:
                break
            if cv2.getWindowProperty("Face and Eye Detection", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
