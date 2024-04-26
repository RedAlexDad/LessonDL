#include <iostream>
#include <algorithm>
#include <iterator>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

using namespace std;
using namespace cl;

// Функция для вычисления факториала
int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

// Функция для получения первой найденной OpenCL платформы
Platform getPlatform() {
    std::vector<Platform> all_platforms;
    Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        cout << "Платформы не найдены. Проверьте установку OpenCL!\n";
        exit(1);
    }
    return all_platforms[0];
}

// Функция для получения устройства с заданным индексом на выбранной платформе
Device getDevice(Platform platform, int i, bool display = false) {
    std::vector<Device> all_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        cout << "Устройства не найдены. Проверьте установку OpenCL!\n";
        exit(1);
    }

    if (display) {
        for (int j = 0; j < all_devices.size(); j++)
            printf("Устройство %d: %s\n", j, all_devices[j].getInfo<CL_DEVICE_NAME>().c_str());
    }
    return all_devices[i];
}

int main() {
    const int n = 1024;    // Размер векторов
    const int c_max = 5;   // Максимальное значение для итераций
    const int coeff = factorial(c_max);

    int A[n], B[n], C[n];     // A - начальный вектор, B - результат, C - ожидаемый результат
    for (int i = 0; i < n; i++) {
        A[i] = i;
        C[i] = coeff * i;
    }
    Platform default_platform = getPlatform();
    cout << "Выбранная платформа: " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;
    Device default_device = getDevice(default_platform, 0);
    cout << "Выбранное устройство: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;
    Context context({default_device});
    Program::Sources sources;

    // OpenCL ядро, умножающее каждый элемент массива на константу c
    std::string kernel_code = R"(
        void kernel multiply_by(global int* A, const int c) {
            A[get_global_id(0)] = c * A[get_global_id(0)];
        }
    )";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        cout << "Ошибка компиляции: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * n, A);

    Kernel multiply_by = Kernel(program, "multiply_by");
    multiply_by.setArg(0, buffer_A);

    // Выполнение ядра для каждого значения c
    for (int c = 2; c <= c_max; c++) {
        multiply_by.setArg(1, c);
        queue.enqueueNDRangeKernel(multiply_by, NullRange, NDRange(n), NDRange(32));
    }

    queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * n, B);

    // Проверка на равенство результатов с ожидаемым результатом
    if (std::equal(std::begin(B), std::end(B), std::begin(C)))
        cout << "Массивы равны!" << endl;
    else
        cout << "Ой, массивы не равны!" << endl;

    return 0;
}
