from tensorflow.python.framework.test_util import is_gpu_available

if is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU")
