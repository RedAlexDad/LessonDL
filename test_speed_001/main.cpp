#include <iostream>
#include <vector>
#include <chrono>
// Определим конкретную версию
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/cl2.hpp>

int main() {
    const int arraySize = 1000000000;
    std::vector<int> a(arraySize, 1);
    std::vector<int> b(arraySize, 2);
    std::vector<int> result(arraySize);

    // КАНОНИЧЕСКИЙ КОД

    // Измеряем время выполнения канонического кода на ЦП
    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < arraySize; ++i) {
        result[i] = a[i] + b[i];
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;

    // std::cout << "CPU Result: ";
    // for (int i = 0; i < arraySize; ++i) {
    //     std::cout << result[i] << " ";
    // }
    // std::cout << std::endl;

    std::cout << "CPU Time on CPU: " << duration_cpu.count() << " seconds" << std::endl;

    // Теперь сравним с OpenCL

    // Измеряем время выполнения OpenCL на ЦП
    auto start_opencl = std::chrono::high_resolution_clock::now();

    // Получаем доступные платформы
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found." << std::endl;
        return 1;
    }

    // Выбираем первую платформу
    cl::Platform platform = platforms.front();

    // Получаем доступные устройства на выбранной платформе (например, GPU)
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        std::cerr << "No GPU devices found." << std::endl;
        return 1;
    }

    // Создаем контекст
    cl::Context context(devices);

    // Создаем очередь команд
    // CPU
    // cl::CommandQueue queue(context, devices.front());
    // GPU
    cl::CommandQueue queue(context, devices.front(), CL_QUEUE_PROFILING_ENABLE);
    // cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);


    // Загружаем исходный код ядра
    std::string kernelSource = R"(
        __kernel void vectorAdd(__global const int* a, __global const int* b, __global int* result) {
            int i = get_global_id(0);
            result[i] = a[i] + b[i];
        }
    )";

    // Создаем программу из исходного кода
    cl::Program::Sources sources{1, {kernelSource.c_str(), kernelSource.length() + 1}};
    cl::Program program(context, sources);

    // Собираем программу
    program.build(devices);

    // Создаем ядро из программы
    cl::Kernel kernel(program, "vectorAdd");

    // Создаем буферы для передачи данных между хостом и устройством
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * arraySize, a.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * arraySize, b.data());
    cl::Buffer bufferResult(context, CL_MEM_WRITE_ONLY, sizeof(int) * arraySize);

    // Устанавливаем аргументы ядра
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferResult);

    // Запускаем ядро
    cl::Event event;
    // Установите размер рабочей группы (поэкспериментируйте, чтобы найти оптимальное значение)
    const size_t workGroupSize = 512;
    // const size_t workGroupSize = 256;
    // const size_t workGroupSize = 128;
    // const size_t workGroupSize = 64;

    // Поставьте ядро в очередь с заданным размером рабочей группы
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(arraySize), cl::NDRange(workGroupSize));

    // queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(arraySize), cl::NullRange, nullptr, &event);

    // Ждем завершения выполнения
    event.wait();

    // Читаем результат обратно на хост
    queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, sizeof(int) * arraySize, result.data());

    auto end_opencl = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_opencl = end_opencl - start_opencl;

    // Выводим результат
    // std::cout << "OpenCL Result: ";
    // for (int i = 0; i < arraySize; ++i) {
    //     std::cout << result[i] << " ";
    // }
    // std::cout << std::endl;

    std::cout << "OpenCL Time on GPU: " << duration_opencl.count() << " seconds" << std::endl;

    return 0;

    // const int arraySize = 100000000000;
    // CPU Time on CPU: 6.67702 seconds
    // OpenCL Time on GPU: 2.77877 seconds
}
