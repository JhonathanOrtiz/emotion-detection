""" Convulutional Neural Network Architectures"""
from torch import nn

from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential


class TorchConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_1 = nn.Sequential(
            nn.Conv2d(3, 64, (5, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
        )

        self.pool_1 = nn.MaxPool2d((2, 2))
        self.drop_1 = nn.Dropout(0.3)

        self.cnn_2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
        )

        self.pool_2 = nn.MaxPool2d((2, 2))
        self.drop_2 = nn.Dropout(0.4)

        self.cnn_3 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
        )

        self.pool_3 = nn.MaxPool2d((2, 2))
        self.drop_3 = nn.Dropout(0.5)

        self.fc_1 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        x = self.cnn_2(x)
        x = self.drop_2(self.pool_2(x))
        x = self.cnn_3(x)
        x = self.drop_3(self.pool_3(x))
        x = x.view((x.shape[0], -1))
        x = self.fc_1(x)
        return x


def load_tf_model(path=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding="same", input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(Dense(7, activation="softmax"))

    model.load_weights(path)
    return model
