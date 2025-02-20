"""
FruitBasic.py
"""
from abc import abstractmethod
from os.path import join
from time import time

# import cv2
from numpy import argmax, expand_dims
from numpy.ma.core import concatenate
from sklearn.metrics import classification_report
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import RandomFlip, RandomRotation
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

from config import WEIGHTS_PATH, FRUIT_LABEL_LIST, FRUIT_IMAGE_HEIGHT, FRUIT_IMAGE_WIDTH, TRAIN_FRUIT_PATH, \
    TEST_FRUIT_PATH, MODELS_PATH
from utils.dataset import train_test_dataset_from_directory
from utils.report import create_accuracy_and_loss, show_images_dataset_to_grid


class FruitBasic:

    def __init__(self, gene_data: bool = True):
        self.model = None
        self.gene_data = gene_data
        self.model_name = self.__class__.__name__.replace("Fruit", "")
        self.num_channels = 3
        self.input_shape = (FRUIT_IMAGE_HEIGHT, FRUIT_IMAGE_WIDTH, self.num_channels)
        self.image_size = (FRUIT_IMAGE_HEIGHT, FRUIT_IMAGE_WIDTH)
        self.batch_size = 16
        self.test_size = 0.2
        # 训练预处理
        # self.train_pretreatment = Sequential([
        #     Rescaling(1. / 127.5, offset=-1, input_shape=self.input_shape)
        # ])
        # 训练增强层
        self.train_augmentation = Sequential([
            RandomFlip('horizontal_and_vertical', input_shape=self.input_shape),
            RandomRotation(0.2),
            # RandomZoom((-0.5, 0.5), (-0.5, 0.5)),
        ])

    @abstractmethod
    def compile(self, train_ds: any, class_units: any) -> any:
        pass

    @abstractmethod
    def reconcile(self, model: any) -> any:
        pass

    def get_train_val_ds(self) -> tuple[any, any, any]:
        """
        Get train and val dataset
        :return:
        """
        # load train dataset
        train_dir = TRAIN_FRUIT_PATH
        print("Load train dataset: " + train_dir)
        train_ds = image_dataset_from_directory(
            train_dir, subset="training", label_mode='categorical', validation_split=0.2,
            batch_size=self.batch_size, image_size=self.image_size, seed=42
        )
        class_names = train_ds.class_names
        train_ds = train_ds.cache().shuffle(self.batch_size * 8, 42).prefetch(buffer_size=AUTOTUNE)
        val_ds = image_dataset_from_directory(
            train_dir, subset="validation", label_mode='categorical', validation_split=0.2,
            batch_size=self.batch_size, image_size=self.image_size, seed=42
        )
        val_ds = val_ds.cache().shuffle(self.batch_size * 8, 42).prefetch(buffer_size=AUTOTUNE)
        return train_ds, val_ds, class_names

    def get_train_test_ds(self) -> tuple[any, any, any, any]:
        """
        Get train and test dataset
        :return:
        """
        train_dir = TRAIN_FRUIT_PATH
        print("Load train dataset: " + train_dir)
        return train_test_dataset_from_directory(train_dir, self.test_size)

    def get_test_ds(self):
        """
        Get test dataset
        :return:
        """
        test_dir = TEST_FRUIT_PATH
        print("Load test dataset: " + test_dir)
        test_ds = image_dataset_from_directory(
            test_dir, label_mode='categorical', batch_size=self.batch_size,
            image_size=self.image_size, seed=42
        )
        class_names = test_ds.class_names
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
        return test_ds, class_names

    def train(self, epochs: int, reconcile: bool = True):
        """
        Train Dataset
        :param epochs:
        :param reconcile:
        :return:
        """
        # 加载数据集与验证集
        begin_time = time()
        train_ds, val_ds, class_names = self.get_train_val_ds()
        print("Train classes: ", class_names)
        # 编译模型
        model = self.compile(train_ds, len(class_names))
        # 指明训练的轮数epoch，开始训练
        model_file = join(WEIGHTS_PATH, self.model_name + ".h5")
        callbacks = [
            # 只保存最佳模型
            ModelCheckpoint(monitor="val_loss", filepath=model_file, save_best_only=True, save_weights_only=True),
            # 当评价指标不在提升时，减少学习率
            ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.1, min_lr=1e-6),
            # 早停，如果验证集loss在5个epoch内没有改善则提前停止训练，停止时恢复最佳权重
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        ]
        trains = model.fit(
            train_ds, epochs=epochs, workers=3, batch_size=self.batch_size,
            validation_data=val_ds, callbacks=callbacks
        )
        results = {"accuracy": trains.history['accuracy'], "val_accuracy": trains.history['val_accuracy'],
                   "loss": trains.history['loss'], "val_loss": trains.history['val_loss']}
        if reconcile:
            print("--------------------------- Reconcile ---------------------------")
            model = self.reconcile(model)
            if model:
                epochs += 10
                trains = model.fit(
                    train_ds, initial_epoch=trains.epoch[-1], epochs=epochs, workers=3, batch_size=self.batch_size,
                    validation_data=val_ds, callbacks=callbacks
                )
                for key in results:
                    results[key] += trains.history[key]
        # 保存成 saved
        model_file = join(MODELS_PATH, self.model_name + ".h5")
        model.save(model_file, save_format="tf")
        # 创建交叉图
        create_accuracy_and_loss(self.model_name, results)
        # 记录结束时间
        print('Train runtime: ', time() - begin_time)
        print("Train results: ", results)

    def predict(self, img_path: str):
        """
        Predict Image
        :return:
        """
        # Load model
        model = self.model if self.model else load_model(MODELS_PATH + self.model_name + ".h5")
        # Load image
        img = load_img(img_path, target_size=self.image_size)
        x = img_to_array(img)
        x = expand_dims(x, 0)
        # Predict image
        predictions = model.predict(x)
        # 输出预测结果
        return predictions

    def predict_max(self, img_path: str) -> int:
        pres = self.predict(img_path)
        return argmax(pres, axis=1)[0]

    def predict_max_label(self, img_path: str) -> str:
        prei = self.predict_max(img_path)
        return FRUIT_LABEL_LIST[prei]

    def evaluate(self):
        """
        Test Dataset
        :return:
        """
        # 加载测试集
        test_ds, class_names = self.get_test_ds()
        # 加载模型
        model = self.model if self.model else load_model(MODELS_PATH + self.model_name + ".h5")
        # model.summary()
        # 测试
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test loss: {test_loss:.3f}")
        print(f"Test accuracy: {test_acc:.3f}")
        # 预测
        # as_numpy_iterator()为分批次batch操作
        image_batch, label_batch = test_ds.as_numpy_iterator().next()
        predictions = model.predict_on_batch(image_batch).argmax(axis=1)
        print('Predictions:\n', predictions)
        print('Labels:\n', label_batch)
        # Show images
        show_images_dataset_to_grid(image_batch, predictions, class_names)

    def report(self):
        """
        Report Dataset
        :return:
        """
        # 加载测试集
        test_ds, class_names = self.get_test_ds()
        # 加载模型
        model = self.model if self.model else load_model(MODELS_PATH + self.model_name + ".h5")
        # 获取预测结果
        y_pred = model.predict(test_ds)
        # 获取测试集的真实标签
        y_true = concatenate([y for x, y in test_ds], 0)
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        report = classification_report(y_true, y_pred)
        print("Test directory: " + TEST_FRUIT_PATH)
        print("Classification report: ")
        print(report)
