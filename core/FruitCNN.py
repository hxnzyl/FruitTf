"""
FruitCNN.py
"""
from abc import ABC

from tensorflow.python.keras.layers import Rescaling, Conv2D, MaxPooling2D, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v1 import SGD
from tensorflow.python.layers.core import Flatten

from core.FruitBasic import FruitBasic


class FruitCNN(FruitBasic, ABC):

    def compile(self, train_ds: any, class_num: int) -> Sequential:
        """
        Create Model
        :param class_num:
        :return:
        """
        # 搭建模型
        model = Sequential([
            # 对模型做归一化
            Rescaling(1. / 255, input_shape=self.input_shape),
            # 卷积层，该卷积层的输出为16
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            # 卷积层，该卷积层的输出为32
            Conv2D(32, (3, 3), activation='relu'),
            # 添加池化层
            MaxPooling2D(2, 2),
            # Add another convolution
            # 卷积层，输出为64
            Conv2D(64, (3, 3), activation='relu'),
            # 池化层
            MaxPooling2D(2, 2),
            # Conv2D(128, (3, 3), activation='relu'),
            # MaxPooling2D(2, 2),
            # 转化为一维
            Flatten(),
            # Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(class_num, activation='softmax')
        ])
        # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
        opt = SGD(learning_rate=0.01)
        # opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # 返回模型
        return model
