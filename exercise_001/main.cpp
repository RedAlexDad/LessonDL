#include <iostream>

// Функция для вычисления суммы элементов массива
extern "C" {
float sum_array(float *array, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum;
}
}
// Можно скомпилировать код C/C++ в разделяемую библиотеку:
// g++ -c -fPIC example.cpp -o example.o
// g++ -shared -o libexample.so example.o

int main() {

    return 0;
}