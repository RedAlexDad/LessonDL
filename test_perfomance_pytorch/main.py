import ctypes
import numpy as np

# Загрузка библиотеки
my_lib = ctypes.cdll.LoadLibrary('./my_lib.so')

# Определение типа аргумента функции
my_lib.calculate_determinant.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), ctypes.c_int]

def calculate_determinant(matrix):
    # Вызов функции из библиотеки
    determinant = my_lib.calculate_determinant(matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), matrix.shape[0])
    return determinant

# Пример использования
matrix = np.random.rand(3, 3).astype(np.float32)
determinant = calculate_determinant(matrix)
print("Определитель матрицы:", determinant)
