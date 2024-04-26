#include <chrono>
#include <torch/torch.h>
#include <iostream>

// Функция для вычисления определителя на CPU
std::chrono::duration<double> computeDeterminantCPU(torch::Tensor &matrix) {
    // Начало замера времени
    auto start = std::chrono::steady_clock::now();

    // Вычисляем определитель матрицы на CPU
    auto determinant_cpu = torch::linalg::det(matrix);

    // Конец замера времени
    auto end = std::chrono::steady_clock::now();

    // Возвращаем время выполнения на CPU
    return std::chrono::duration<double>(end - start);
}

// Функция для вычисления определителя на GPU (CUDA)
std::chrono::duration<double> computeDeterminantCUDA(torch::Tensor &matrix) {
    // Переключаемся на устройство CUDA
    torch::Device device(torch::kCUDA);
    matrix = matrix.to(device);

    // Начало замера времени
    auto start = std::chrono::steady_clock::now();

    // Вычисляем определитель матрицы на CUDA
    auto determinant_cuda = torch::linalg::det(matrix);

    // Конец замера времени
    auto end = std::chrono::steady_clock::now();

    // Переключаемся обратно на CPU для вывода результата
    determinant_cuda = determinant_cuda.to(torch::kCPU);

    // Возвращаем время выполнения на CUDA
    return std::chrono::duration<double>(end - start);
}

// Функция для сравнения производительности на CPU и CUDA
void computeDeterminantComparison(int N) {
    // Создаем случайную матрицу с помощью PyTorch
    auto matrix = torch::randn({N, N});

    // Вычисляем определитель на CPU
    auto elapsed_seconds_cpu = computeDeterminantCPU(matrix);

    // Вычисляем определитель на CUDA
    auto elapsed_seconds_cuda = computeDeterminantCUDA(matrix);

    // Выводим результаты
    std::cout << "Размерность матрицы: " << N << std::endl;
    std::cout << "Определитель матрицы на CPU: " << elapsed_seconds_cpu.count() << " секунд." << std::endl;
    std::cout << "Определитель матрицы на CUDA: " << elapsed_seconds_cuda.count() << " секунд." << std::endl;

    // Вычисляем во сколько раз быстрее операция на CUDA по сравнению с CPU
    double speedup = elapsed_seconds_cpu.count() / elapsed_seconds_cuda.count();
    std::cout << "Операция на CUDA быстрее, чем на CPU, в " << speedup << " раз(а)." << std::endl;
}

int main(int argc, char* argv[]) {
    // Проверяем, был ли передан аргумент через командную строку
    if (argc > 1) {
        // Преобразуем аргумент из строки в целое число
        int N = std::stoi(argv[1]);
        // Вызываем функцию computeDeterminantComparison с переданным значением размера матрицы
        computeDeterminantComparison(N);
    } else {
        // Если аргумент не был передан, выводим сообщение об использовании программы
        std::cout << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
    }

    return 0;
}

/*
 * Данные после работы программы:
 *
 * (base) redalexdad@redalexdad-Nitro-AN515-44:~/Документы/GitHub/LessonDL/cpp/cmake-build-debug$ ./exercise_002_main 100
 * Размерность матрицы: 100
 * Определитель матрицы на CPU: 0.00883802 секунд.
 * Определитель матрицы на CUDA: 0.152505 секунд.
 * Операция на CUDA быстрее, чем на CPU, в 0.0579525 раз(а).
 *
 * (base) redalexdad@redalexdad-Nitro-AN515-44:~/Документы/GitHub/LessonDL/cpp/cmake-build-debug$ ./exercise_002_main 1000
 * Размерность матрицы: 1000
 * Определитель матрицы на CPU: 0.00987784 секунд.
 * Определитель матрицы на CUDA: 0.162866 секунд.
 * Операция на CUDA быстрее, чем на CPU, в 0.0606501 раз(а).
 *
 * (base) redalexdad@redalexdad-Nitro-AN515-44:~/Документы/GitHub/LessonDL/cpp/cmake-build-debug$ ./exercise_002_main 10000
 * Размерность матрицы: 10000
 * Определитель матрицы на CPU: 2.51953 секунд.
 * Определитель матрицы на CUDA: 0.617398 секунд.
 * Операция на CUDA быстрее, чем на CPU, в 4.08089 раз(а).
 *
 * (base) redalexdad@redalexdad-Nitro-AN515-44:~/Документы/GitHub/LessonDL/cpp/cmake-build-debug$ ./exercise_002_main 20000
 * Размерность матрицы: 20000
 * Определитель матрицы на CPU: 18.9653 секунд.
 * Определитель матрицы на CUDA: 3.27793 секунд.
 * Операция на CUDA быстрее, чем на CPU, в 5.78576 раз(а).
 *
 * (base) redalexdad@redalexdad-Nitro-AN515-44:~/Документы/GitHub/LessonDL/cpp/cmake-build-debug$ ./exercise_002_main 50000
 * terminate called after throwing an instance of 'c10::OutOfMemoryError'
 * what():  CUDA out of memory. Tried to allocate 9.31 GiB. GPU
 * Аварийный останов (образ памяти сброшен на диск)
 */