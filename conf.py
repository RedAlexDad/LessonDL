import sys
import ctypes
import subprocess
import tensorflow as tf

# Вывести информацию о системе Ubuntu
ubuntu_info = subprocess.run(['lsb_release', '-a'], capture_output=True, text=True)
print("Ubuntu information:")
print(ubuntu_info.stdout)

# Вывести информацию о архитектуре и версии ядра
architecture_info = subprocess.run(['uname', '-m'], capture_output=True, text=True)
kernel_info = subprocess.run(['uname', '-r'], capture_output=True, text=True)
print("\nSystem information:")
print("Architecture:", architecture_info.stdout.strip())
print("Kernel version:", kernel_info.stdout.strip())

# Вывести остальную информацию
print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)
print("CUDA support:", tf.test.is_built_with_cuda())
import tensorflow as tf


# Получение информации о версии CUDA
def get_cuda_version():
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if line.startswith('Cuda compilation tools'):
                cuda_version = line.split()[-1]
                return cuda_version
        return "CUDA not found"
    except Exception as e:
        return f"Error retrieving CUDA version: {e}"


# Пример использования
cuda_version = get_cuda_version()
print("CUDA version:", cuda_version)


# Версия cuDNN
def get_cudnn_version():
    try:
        # Загрузка библиотеки cuDNN
        libcudnn = ctypes.CDLL("libcudnn.so")
        # Функция cuDNN для получения версии
        cudnnGetVersion = libcudnn.cudnnGetVersion
        # Вызов функции и возвращение версии
        return cudnnGetVersion()
    except Exception as e:
        return f"Error getting cuDNN version: {e}"

# Вывод версии cuDNN
print("cuDNN version:", get_cudnn_version())

print("GPU devices available:", tf.config.experimental.list_physical_devices('GPU'))
