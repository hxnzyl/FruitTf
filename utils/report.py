from datetime import datetime
from os import environ
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from config import RESULTS_PATH

# fix: KMP already initialized
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def create_heat_map(model_name: str, model_obj: Any, test_ds: Any):
    """
    创建热力图
    :param model_name:
    :param model_obj:
    :param test_ds:
    :return:
    """
    # 基本配置
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 对模型分开进行推理
    val_rel = []
    val_pre = []
    class_names = test_ds.class_names
    class_size = len(test_ds.class_names)
    for images, reals in test_ds:
        pre = model_obj.predict(images)
        rel_max = np.argmax(reals.numpy(), axis=1)
        pre_max = np.argmax(pre, axis=1)
        # 将推理对应的标签取出
        for i in rel_max:
            val_rel.append(i)
        for i in pre_max:
            val_pre.append(i)
    heatmap = np.zeros((class_size, class_size))
    for label, pre in zip(val_rel, val_pre):
        heatmap[label][pre] = heatmap[label][pre] + 1
    # 创建画布
    heatmap_sum = np.sum(heatmap, axis=1).reshape(-1, 1)
    heatmap_dat = heatmap / heatmap_sum
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_dat, cmap="OrRd")
    # 修改标签
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    # x轴标签过长，需要旋转一下
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    # 添加每个热力块的具体数值
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            # print(FRUIT_CLASS[i], FRUIT_CLASS[j])
            ax.text(j, i, round(heatmap_dat[i, j], 2), ha="center", va="center", color="black")
    ax.set_xlabel("Predict")
    ax.set_ylabel("Actual")
    ax.set_title(model_name + " Heat Map")
    fig.tight_layout()
    plt.colorbar(im)
    # 保存到本地
    filename = RESULTS_PATH + model_name + '_heat_map_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
    plt.savefig(filename, dpi=200)
    print("Heat Map Save to: " + filename)


def create_accuracy_and_loss(model_name: str, history: Any):
    """
    Create accuracy and loss
    :param model_name:
    :param history:
    :return:
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # 保存图片
    filename = RESULTS_PATH + model_name + '_result_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
    plt.savefig(filename, dpi=200)
    print("Accuracy And Loss Save to: " + filename)


def show_images_dataset_to_grid(images: Any, labels: Any, class_names: list[str]):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()
