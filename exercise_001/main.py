import ctypes

# Загрузите разделяемую библиотеку C/C++
lib = ctypes.CDLL("./cmake-build-debug/libLessonDL.so")

# Определите типы аргументов и возвращаемого значения для функции C
lib.sum_array.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.sum_array.restype = ctypes.c_float

# Создайте массив Python
array = [1.0, 2.0, 3.0, 4.0]

# Преобразуйте массив Python в массив C
array_c = (ctypes.c_float * len(array))(*array)

# Вызовите функцию C
sum = lib.sum_array(array_c, len(array))

# Выведите результат
print(f"Сумма элементов массива: {sum}")