# Model
from torchvision.transforms import transforms

TORCH_MODEL_PARAMS = (
    "/Users/jhonathanortiz/dev/tesis/emotion-recognizer/models/torchmodel_4.pt"
)
FACE_MODEL = "/Users/jhonathanortiz/dev/tesis/emotion-recognizer/models/haarcascade_frontalface_default.xml"

TF_WEIGHTS = (
    "/Users/jhonathanortiz/dev/tesis/emotion-recognizer/models/tf_model_weight.h5"
)

CLASSES_DICT = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "suprised",
}

TORCH_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
