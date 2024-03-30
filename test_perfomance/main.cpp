// main.cpp
#include <iostream>
#include <ctime>
#include <csignal> // Для использования raise и SIGABRT

extern "C" {
int fibonacci_cpp(int n) {
    if (n <= 1)
        return n;
    else
        return fibonacci_cpp(n - 1) + fibonacci_cpp(n - 2);
}
}

double measure_cpp_performance(int n) {
    clock_t start_time = clock();
    std::cout << "Число фибоначчи" << ": " << fibonacci_cpp(n) << std::endl;
    clock_t end_time = clock();
    return double(end_time - start_time) / CLOCKS_PER_SEC;
}

int main() {
    int n = 20;

    if (n > 30) {
        std::cerr
                << "Предупреждение: Значение n больше 30. Это может вызвать длительное время выполнения алгоритма Фибоначчи."
                << std::endl;
        raise(SIGABRT);
        // Можно также выбросить исключение или вызвать функцию raise(SIGABRT) для завершения программы
    }

    double cpp_time = measure_cpp_performance(n);
    std::cout << "Время выполнения функции для n = " << n << ": " << cpp_time << " секунд" << std::endl;

    return 0;
}