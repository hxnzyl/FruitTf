import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:  # 检查是否至少存在一个GPU设备。
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

"""################################################################
# 函数功能：用于配置 GPU 内存管理。
# 函数说明：tf.config.experimental.set_memory_growth(device, enable)
# 参数说明：
#         device      要配置的物理设备对象（如:'GPU'、'CPU'、'TPU'）
#         enable      内存增长模式的启用状态（bool类型）。
#                     （1）若为True, 启用按需分配内存的模式。即 TensorFlow 将在需要时动态增加 GPU 内存，而不是一次性分配整个 GPU 的内存空间。
#                     （2）若为False，禁用按需分配内存的模式。即 TensorFlow 会一次性分配整个 GPU 的内存空间。
################################################################"""
