import torch
from torchvision.transforms import transforms
from PIL import Image

import numpy as np

from model import TorchConvNet, load_tf_model
from utils import constant


# TensorFlow stuff
tf_convnet = load_tf_model(constant.TF_WEIGHTS)

# classes
classes = constant.CLASSES_DICT


def torch_inference(img: np.ndarray) -> str:
    """Feed trained CNN with an image and get predictions.

    Args:
        img (np.ndarray): Image to be classified.

    Returns:
        str: Class where given image belong.
    """
    model_params = torch.load(constant.CNN_MODEL_PARAMS)
    torch_convnet = TorchConvNet()
    torch_convnet.state_dict(model_params)
    image = Image.fromarray(img)
    tensor_image = constant.TORCH_TRANSFORMS(image)
    tensor_image = torch.unsqueeze(tensor_image, dim=0)

    with torch.no_grad():
        torch_convnet.eval()
        probs = torch_convnet(tensor_image)
        pred = torch.argmax(probs).item()
        str_class = classes.get(pred)
        return str_class


def tf_inference(x: np.ndarray) -> str:
    """Feed trained CNN with an image and get predictions.

    Args:
        x (np.ndarray): Image to be classified.

    Returns:
        str: Class where given image belong.
    """

    x = x / 255.0
    batch_image = x[np.newaxis, :, :, np.newaxis]
    prob_preds = tf_convnet.predict(batch_image)
    pred = np.argmax(prob_preds)
    return classes.get(pred)
