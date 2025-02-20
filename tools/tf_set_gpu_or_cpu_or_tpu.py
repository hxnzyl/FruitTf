import os

# （1）设置环境变量 CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# （2）测试 os.environ['CUDA_VISIBLE_DEVICES']
try:
    cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    print("os.environ['CUDA_VISIBLE_DEVICES']     =", cuda_visible_devices)
except KeyError:
    print("环境变量 CUDA_VISIBLE_DEVICES 未设置")

# （3）测试 os.environ.get('CUDA_VISIBLE_DEVICES')
cuda_visible_devices_get = os.environ.get('CUDA_VISIBLE_DEVICES')
if cuda_visible_devices_get is not None:
    print("os.environ.get('CUDA_VISIBLE_DEVICES') =", cuda_visible_devices_get)
else:
    print("环境变量 CUDA_VISIBLE_DEVICES 未设置")

"""###########################################################################################################
# 函数功能：获取环境变量 CUDA_VISIBLE_DEVICES 的值。
# 函数说明：os.environ['CUDA_VISIBLE_DEVICES']
#           CPU:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'       # 表示禁用所有GPU设备，TensorFlow使用CPU设备。
#           GPU:        os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # 表示（单GPU）TensorFlow可以看到并使用的GPU设备的编号: 0。
#                       os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'    # 表示（多GPU）TensorFlow可以看到并使用的GPU设备的编号: 0、1、2。
# 返回一个包含指定设备编号的字符串，表示 TensorFlow 可以看到的 GPU 设备列表。
###########################################################################################################"""
