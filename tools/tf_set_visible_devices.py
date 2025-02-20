import tensorflow as tf

# 获取CPU和GPU设备列表
devices = tf.config.experimental.list_physical_devices('CPU') + tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(devices[0], 'CPU')  # 设置CPU设备为可见
tf.config.experimental.set_visible_devices(devices[1], 'GPU')  # 设置GPU设备为可见

"""################################################################
# 函数功能：用于设置指定类型的物理设备为可见状态，从而选择使用的设备。
# 函数说明：tf.config.experimental.set_visible_devices(devices, device_type)
# 参数说明：
#         devices         要设置为可见的物理设备列表。
#         device_type:    要设置为可见的设备类型。可以是 'CPU'、'GPU' 或者 'TPU'。
################################################################"""

"""################################################################
# 函数功能：获取系统中的物理设备列表，可以指定设备类型。
# 函数说明：tf.config.experimental.list_physical_devices(device)
# 参数说明：
#         device      设备类型。
#             （1）若不输入，则默认列举所有设备
#             （2）若输入指定类型，则提取指定类型的所有设备。如:'GPU'、'CPU'、'TPU'。
################################################################"""
