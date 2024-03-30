import ctypes
import time

lib = ctypes.CDLL('./build/libLessonDL.so')

lib.fibonacci_cpp.argtype = [ctypes.c_int]
lib.fibonacci_cpp.restype = ctypes.c_int





def fibonacci_python(n):
    if n <= 1:
        return n
    else:
        return fibonacci_python(n - 1) + fibonacci_python(n - 2)


def measure_python_performance(n):
    start_time = time.time()
    result = fibonacci_python(n)
    end_time = time.time()
    time_measure = end_time - start_time

    return result, time_measure

def fibonacci_cpp(n):
    start_time = time.time()
    result = lib.fibonacci_cpp(n)
    end_time = time.time()
    time_measure = end_time - start_time

    return result, time_measure



if __name__ == "__main__":
    n = 20

    if n > 30:
        raise ValueError('n должно быть не больше 30')

    result_python, python_time = measure_python_performance(n)
    result_cpp, cpp_time = fibonacci_cpp(n)

    print('Размер фибоначчи: ', n)
    print(f'Результат вызова функции на C/C++: {result_cpp}, время: {cpp_time}')
    print(f'Результат вызова функции на Python: {result_python}, время: {python_time}')
    print(f'C/C++ быстрее Python в разы: {python_time/cpp_time}')