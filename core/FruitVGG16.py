"""
FruitVGG16.py
"""
from abc import ABC

from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.models import Model

from core.FruitBasic import FruitBasic


class FruitVGG16(FruitBasic, ABC):

    def compile(self, train_ds: any, class_num: int) -> any:
        """
        Compile Model
        :return:
        """
        class_names = train_ds.class_names
        class_units = len(class_names)
        base_model = VGG16(weights="imagenet", include_top=False)
        inputs = Input(self.input_shape)
        x = self.train_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(class_units, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model
