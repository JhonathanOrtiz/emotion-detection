from tkinter import *
from tkinter import Tk
import json

from PIL import Image, ImageTk
import cv2
import numpy as np
import requests

from utils import constant

faceClassif = cv2.CascadeClassifier(constant.FACE_MODEL)
url = "http://127.0.0.1:8000/predict"
result = ""


def vizualize():
    global cap
    global result
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (480, 300))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            auxFrame = gray.copy()
            faces = faceClassif.detectMultiScale(gray, 1.3, 5)
            im = frame
            for (x, y, w, h) in faces:
                rostro = auxFrame[y : y + h, x : x + w]
                rostro = cv2.resize(rostro, (48, 48), interpolation=cv2.INTER_CUBIC)
                result = predict_emotion(rostro)
                im = draw_face(frame, result, x, y, h, w)
                if Variable.get() == "Imagen":
                    print(result)
                    emoji = get_emoji(result)
                    cv2.imshow("rostro", emoji)
            im = Image.fromarray(im)
            img = ImageTk.PhotoImage(image=im)

            lbl_video.configure(image=img)
            lbl_video.image = img
            lbl_video.after(10, vizualize)

        else:
            cap.release()
    else:
        lbl_video.image = ""
        cap.release()


def predict_emotion(rostro):
    list_img = rostro.tolist()
    data = {"image": list_img}
    try:
        response = response = requests.post(url, json=data).json()
        return response["prediction"]
    except json.JSONDecodeError:
        return "No emotion detected"


def draw_face(frame, result, x, y, h, w):
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
    return frame


def get_emoji(str_class: str) -> np.ndarray:
    image = cv2.imread(
        f"/Users/jhonathanortiz/dev/tesis/emotion-recognizer/emojis/{str_class}.jpeg"
    )
    image = cv2.resize(image, (480, 300))
    return image


def end():
    global cap
    cap.release()
    cv2.destroyAllWindows()


def init_video():
    global cap
    cap = cv2.VideoCapture(0)
    vizualize()


def play_video(result):
    print("here")
    video = cv2.VideoCapture(
        f"/Users/jhonathanortiz/dev/tesis/emotion-recognizer/videos/{result}.mp4"
    )

    while True:
        ret, frame = video.read()
        if ret:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            video = cv2.VideoCapture(
                f"/Users/jhonathanortiz/dev/tesis/emotion-recognizer/videos/{result}.mp4"
            )
    video.release()
    cv2.destroyAllWindows()


def capture():
    global result
    end()
    play_video(result)
    print("here")


root = Tk()
cap = None


f1 = Frame(root)
f1.grid(column=1, row=1)

f3 = Frame(root)
f3.grid(column=2, row=1)

f2 = Frame(root)
f2.grid(column=3, row=1)

Variable = StringVar(f1)
Variable.set("Libre")

top_label = Label(f3, text="Facial Emotion Recognizer")
top_label.grid(
    column=1,
    row=1,
    columnspan=4,
)

init_button = Button(f1, text="Iniciar", width=15, command=init_video)
init_button.grid(column=0, row=1, padx=1, pady=1)

end_button = Button(f1, text="Finalizar", width=15, command=end)
end_button.grid(column=0, row=4, padx=1, pady=1)

cap_button = Button(f2, text="Capturar", width=15, command=capture)
cap_button.grid(column=0, row=1, padx=1, pady=1)

select_eslimulo = OptionMenu(f2, Variable, "Libre", "Imagen", "Colores")
select_eslimulo.grid(column=0, row=3, padx=1, pady=1)

lbl_video = Label(f3)
lbl_video.grid(column=1, row=2, columnspan=4, rowspan=4)
root.mainloop()
