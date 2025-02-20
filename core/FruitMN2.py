"""
FruitMN2.py
"""

from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from core.FruitBasic import FruitBasic


class FruitMN(FruitBasic):
    learning_rate: float = 0.001
    base_model: Model = None

    def compile(self, train_ds: any, class_units: int) -> any:
        """
        Compile Model
        :return:
        """
        # 构建 MobileNet 模型
        base_model = MobileNetV2(input_shape=self.input_shape, weights="imagenet", include_top=False)
        # 特征提取程序将每个 image_batch.shape 图像转换为 5x5x1280 的特征块
        image_batch, label_batch = next(iter(train_ds))
        feature_batch = base_model(image_batch)
        # 将模型的主干参数进行冻结
        base_model.trainable = False
        # base_model.summary()
        # 将特征转换成每个图像一个向量（包含1280个元素）
        global_average_layer = GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        # 用Dense层将这些特征转换成每个图像一个预测
        prediction_layer = Dense(class_units, activation="softmax")
        prediction_layer(feature_batch_average)
        # 用Keras函数式API将数据扩充、重新缩放、base_model和特征提取程序层链接在一起来构建模型
        inputs = Input(self.input_shape)  # 统一输入尺寸
        # x = self.train_pretreatment(inputs)
        x = self.train_augmentation(inputs)  # 数据增强
        x = preprocess_input(x)  # 输入预处理
        x = base_model(x, training=False)  # 由于我们的模型包含 BatchNormalization 层，因此使用 training = False
        x = global_average_layer(x)  # 转换为每个图像一个向量
        x = Dropout(0.2)(x)  # 使用Dropout
        outputs = prediction_layer(x)  # 预测输出值
        model = Model(inputs, outputs)
        # 编译模型
        # 输出层会做normalization(softmax)
        # BinaryCrossentropy(from_logits=True)
        model.compile(Adam(learning_rate=self.learning_rate), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        model.summary()
        # 权重和偏差
        # print(len(model.trainable_variables))
        self.base_model = base_model
        return model

    def reconcile(self, model: any) -> any:
        # 解冻模型的顶层
        self.base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.base_model.layers))
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        # base_model.summary()
        # 重新编译
        # 在训练一个大得多的模型并且想要重新调整预训练权重时使用较低的学习率。
        model.compile(RMSprop(learning_rate=self.learning_rate / 10), loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.summary()
        return model
