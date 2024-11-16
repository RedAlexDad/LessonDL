#include <CL/sycl.hpp>
#include <iostream>

int main() {
    // Создаем SYCL очередь
    cl::sycl::queue queue;

    const int N = 1024;
    int data[N];

    // Инициализация массива
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }

    // Параллельное вычисление с использованием SYCL
    {
        cl::sycl::buffer<int, 1> buf(data, cl::sycl::range<1>(N));

        queue.submit([&](cl::sycl::handler &cgh) {
            auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);

            cgh.parallel_for<class hello_world>(cl::sycl::range<1>(N), [=](cl::sycl::id<1> idx) {
                acc[idx] *= 2; // Удвоение каждого элемента
            });
        });
    }

    // Вывод результатов
    for (int i = 0; i < 10; ++i) { // Вывод первых 10 элементов
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
