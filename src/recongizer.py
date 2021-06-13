import cv2

from get_predictions import tf_inference
from utils import constant

face_model_path = constant.FACE_MODEL
print(face_model_path)


cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(face_model_path)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    # nFrame = np.hstack([frame, np.zeros((480, 300, 3), dtype=np.uint8)])
    nFrame = frame
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y : y + h, x : x + w]
        rostro = cv2.resize(rostro, (48, 48), interpolation=cv2.INTER_CUBIC)
        result = tf_inference(rostro)
        print(result)

        cv2.putText(
            frame,
            "{}".format(result),
            (x, y - 5),
            1,
            1.3,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # image = emotionImage(imagePaths[result[0]])
        # nFrame = cv2.hconcat([frame, image])
        nFrame = frame

    cv2.imshow("nFrame", nFrame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
