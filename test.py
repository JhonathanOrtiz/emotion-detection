import cv2

# Read video from file with opencv and play it
cap = cv2.VideoCapture(
    "/Users/jhonathanortiz/dev/tesis/emotion-recognizer/videos/VID-20210725-WA0149.mp4"
)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        cap = cv2.VideoCapture(
            "/Users/jhonathanortiz/dev/tesis/emotion-recognizer/videos/VID-20210725-WA0149.mp4"
        )
cap.release()
cv2.destroyAllWindows()
