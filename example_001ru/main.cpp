#include <iostream>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else

#include <CL/cl2.hpp>

#endif

#define NUM_GLOBAL_WITEMS 1024

void compareResults(double CPUtime, double GPUtime, int trial) {
    double time_ratio = (CPUtime / GPUtime);
    std::cout << "ВЕРСИЯ " << trial << " -----------" << std::endl;
    std::cout << "Время на ЦП: " << CPUtime << std::endl;
    std::cout << "Время на ГП: " << GPUtime << std::endl;
    std::cout << "ГП быстрее в ";
    if (time_ratio > 1)
        std::cout << time_ratio << " раз!" << std::endl;
    else
        std::cout << (1 / time_ratio) << " раз медленнее :(" << std::endl;
}

double timeAddVectorsCPU(int n, int k) {
    // складывает два вектора размером n k раз, возвращает общее время выполнения
    std::clock_t start;
    double duration;

    int A[n], B[n], C[n];
    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = n - i;
        C[i] = 0;
    }

    start = std::clock();
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++)
            C[j] = A[j] + B[j];
    }

    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    return duration;
}

void warmup(cl::Context &context, cl::CommandQueue &queue,
            cl::Kernel &add, int A[], int B[], int n) {
    int C[n];
    // выделяем память
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);

    // отправляем команды записи в очередь
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * n, B);

    // ЗАПУСКАЕМ ЯДРО
    add.setArg(1, buffer_B);
    add.setArg(0, buffer_A);
    add.setArg(2, buffer_C);
    for (int i = 0; i < 5; i++)
        queue.enqueueNDRangeKernel(add, cl::NullRange, cl::NDRange(NUM_GLOBAL_WITEMS), cl::NDRange(32));

    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * n, C);
    queue.finish();
}

int main(int argc, char *argv[]) {
    bool verbose;
    if (argc == 1 || std::strcmp(argv[1], "0") == 0)
        verbose = true;
    else
        verbose = false;

    const int n = 8 * 32 * 128;             // размер векторов
    // const int n = 8 * 32 * 512;             // размер векторов
    // const int n = 8 * 32 * 1024;             // размер векторов
    const int k = 10000;                    // количество итераций цикла
    // const int NUM_GLOBAL_WITEMS = 1024; // количество потоков

    // получаем все платформы (драйверы), например, NVIDIA
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << " Не найдено платформ. Проверьте установку OpenCL!\n";
        exit(1);
    }
    cl::Platform default_platform = all_platforms[0];

    // получаем устройство по умолчанию (ЦП, ГП) на выбранной платформе
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " Не найдено устройств. Проверьте установку OpenCL!\n";
        exit(1);
    }

    // используем устройство[1], так как это ГП; устройство[0] - ЦП (НЕ ФАКТ, см вывод на консольное окно)
    cl::Device default_device = all_devices[0];
    std::cout << "Выбранная платформа: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Использованное устройство: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context({default_device});
    cl::Program::Sources sources;

    // вычисляет для каждого элемента; C = A + B
    std::string kernel_code = R"(
       void kernel add_looped_1(global const int* v1, global const int* v2, global int* v3,
                                 const int n, const int k) {
           int ID, NUM_GLOBAL_WITEMS, ratio, start, stop;
           ID = get_global_id(0);
           NUM_GLOBAL_WITEMS = get_global_size(0);

           ratio = (n / NUM_GLOBAL_WITEMS); // элементов на поток
           start = ratio * ID;
           stop  = ratio * (ID+1);

           int i, j;
           for (i=0; i<k; i++) {
               for (j=start; j<stop; j++) {
                   v3[j] = v1[j] + v2[j];
               }
           }
       }
    )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Ошибка при построении: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)
                  << std::endl;
        exit(1);
    }

    // запускаем CPU-код
    float CPUtime = timeAddVectorsCPU(n, k);

    // устанавливаем ядра и векторы для GPU-кода
    cl::CommandQueue queue(context, default_device);
    cl::Kernel add_looped_1 = cl::Kernel(program, "add_looped_1");

    // конструируем векторы
    int A[n], B[n], C[n];
    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = n - i - 1;
    }

    // прогрев...
    warmup(context, queue, add_looped_1, A, B, n);
    queue.finish();

    std::clock_t start_time;

    // ВЕРСИЯ 1 ==========================================
    // запускаем таймер
    double GPUtime1;
    start_time = std::clock();

    // выделяем память
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);

    // отправляем команды записи в очередь
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * n, B);

    // ЗАПУСКАЕМ ЯДРО
    add_looped_1.setArg(0, buffer_A);
    add_looped_1.setArg(1, buffer_B);
    add_looped_1.setArg(2, buffer_C);
    add_looped_1.setArg(3, n);
    add_looped_1.setArg(4, k);
    queue.enqueueNDRangeKernel(add_looped_1, cl::NullRange,  // ядро, смещение
                               cl::NDRange(NUM_GLOBAL_WITEMS), // глобальное количество рабочих элементов
                               cl::NDRange(128));               // локальное количество (на группу)

    // считываем результат с ГП сюда; включая ради замера времени
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * n, C);
    queue.finish();
    GPUtime1 = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;

    // сравниваем!
    const int NUM_VERSIONS = 1;
    double GPUtimes[NUM_VERSIONS] = {GPUtime1};
    if (verbose) {
        for (int i = 0; i < NUM_VERSIONS; i++)
            compareResults(CPUtime, GPUtimes[i], i + 1);
    } else {
        std::cout << CPUtime << ",";
        for (int i = 0; i < NUM_VERSIONS - 1; i++)
            std::cout << GPUtimes[i] << ",";
        std::cout << GPUtimes[NUM_VERSIONS - 1] << std::endl;
    }
    return 0;
}
