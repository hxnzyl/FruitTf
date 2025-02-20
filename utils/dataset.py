from os import listdir, makedirs
from os.path import exists, join
from time import time

from keras.applications.imagenet_utils import preprocess_input
from numpy import array, zeros
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from config import FRUIT_IMAGE_HEIGHT, FRUIT_IMAGE_WIDTH


def gene_train_dataset_from_directory(target_dir: str, source_dir: str):
    """
    增加测试集
    :return:
    """
    # 开始增强
    begin_time = time()

    datagen = ImageDataGenerator(
        rotation_range=40,  # 随机旋转角度范围
        width_shift_range=0.2,  # 随机水平平移范围(相对于图片宽度)
        height_shift_range=0.2,  # 随机竖直平移范围(相对于图片高度)
        shear_range=0.2,  # 随机裁剪
        zoom_range=0.2,  # 随机缩放
        horizontal_flip=True,  # 随机水平翻转
        vertical_flip=True,  # 随机竖直翻转
        fill_mode='nearest'  # 填充模式
    )

    # fro parent dir
    for subdir in listdir(target_dir):
        if not exists(join(source_dir, subdir)):
            makedirs(join(source_dir, subdir))
        # for class dirs
        l = 0
        for file in listdir(join(target_dir, subdir)):
            img = load_img(join(target_dir, subdir, file))
            x = img_to_array(img)
            # 将图片转化为4D张量(batch_size, height, width, channels)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(
                    x, batch_size=1, save_to_dir=join(source_dir, subdir),
                    save_prefix=file[:-4], save_format='jpg'):
                i += 1
                # 控制每张图片生成4张新图像
                if i > 3:
                    break
            l += 1
            print(f'GENE Dataset: {subdir} {l}')

    run_time = time() - begin_time
    print('GENE Runtime:', run_time, "s")


def train_test_dataset_from_directory(directory: str, test_size: float = 0.2) -> tuple[any, any, any, any]:
    """
    Train test split
    :param directory:
    :param test_size:
    :return:
    """
    label_list = listdir(directory)
    label_size = len(label_list)
    # 加载图像数据和标签
    img_list = []
    idx_list = []
    for index, label in enumerate(label_list):
        sub_dir = join(directory, label)
        for sub_file in listdir(sub_dir):
            img_path = join(sub_dir, sub_file)
            img_file = load_img(img_path, target_size=(FRUIT_IMAGE_HEIGHT, FRUIT_IMAGE_WIDTH))
            img_file = img_to_array(img_file)
            img_file = preprocess_input(img_file)
            img_list.append(img_file)
            idx_list.append(index)
    img_list = array(img_list)
    idx_list = array(idx_list)
    # 独热编码标签
    hot_labels = zeros((len(idx_list), label_size))
    for index, label in enumerate(idx_list):
        hot_labels[index, label] = 1
    # 划分训练集和测试集（同时也是验证集）
    # X_train, X_test, y_train, y_test
    return train_test_split(img_list, hot_labels, test_size=test_size, random_state=42)
