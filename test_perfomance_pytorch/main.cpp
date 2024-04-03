#include <iostream>
#include <chrono>
#include <tuple>
#include <torch/torch.h>

extern "C" {
std::tuple<double, double> calculate_determinant(int N) {
    // Создаем случайную матрицу с помощью PyTorch
    auto matrix = torch::randn({N, N});

    // Начало замера времени
    auto start = std::chrono::steady_clock::now();

    // Вычисляем определитель матрицы
    double determinant = torch::linalg::det(matrix).item<double>();

    // Конец замера времени
    auto end = std::chrono::steady_clock::now();

    // Вычисляем время выполнения
    double execution_time = std::chrono::duration<double>(end - start).count();

    // Возвращаем результат и время выполнения в виде кортежа
    return std::make_tuple(determinant, execution_time);
}
}

int main() {
    int N = 1000;
    double determinant, execution_time;

    // Вызываем функцию для вычисления определителя матрицы и времени выполнения
    auto result = calculate_determinant(N);

    // Выводим результаты
    std::cout << "Определитель матрицы: " << std::get<0>(result) << std::endl;
    std::cout << "Время выполнения на CPU: " << std::get<1>(result) << " секунд." << std::endl;

    return 0;
}
/*
 * Ваш код успешно выполнен.sdadas
 * Он создает случайную матрицу размером 1000x1000 с помощью PyTorch, вычисляет ее определитель и выводит результат, а также время выполнения.
 * В данном случае определитель матрицы оказался бесконечным (inf), что, вероятно, связано с тем, что матрица является вырожденной.
 *
 * Это время выполнения зависит от производительности вашего компьютера и может отличаться на других машинах.
 */