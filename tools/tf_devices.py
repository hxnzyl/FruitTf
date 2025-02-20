from tensorflow.python.framework.config import list_physical_devices

# 在当前系统中，获取 TensorFlow 支持的所有物理设备
devices = list_physical_devices()
# 打印每个设备的信息（类型 + 名称）
for device in devices:
    print(device, "设备类型:", device.device_type, "设备名称:", device.name)

"""
PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU') 设备类型: CPU 设备名称: /physical_device:CPU:0
PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU') 设备类型: GPU 设备名称: /physical_device:GPU:0
"""
