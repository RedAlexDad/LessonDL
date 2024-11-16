#include <iostream>
#include <CL/cl.h>

int main() {
    // Инициализация переменных
    cl_uint platformCount;
    cl_platform_id* platforms;

    // Получение количества доступных платформ
    clGetPlatformIDs(0, nullptr, &platformCount);

    // Выделение памяти под платформы
    platforms = new cl_platform_id[platformCount]; 

    // Получение списка платформ
    clGetPlatformIDs(platformCount, platforms, nullptr);

    // Печать информации о каждой платформе
    for (cl_uint i = 0; i < platformCount; i++) {
        char platformName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);

        std::cout << "Platform " << i + 1 << ": " << platformName << std::endl;

        // Получение количества устройств в платформе
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 
                       0, nullptr, &deviceCount);

        // Выделение памяти под устройства
        cl_device_id* devices = new cl_device_id[deviceCount];

        // Получение списка устройств
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 
                       deviceCount, devices, nullptr);

        // Печать информации о каждом устройстве
        for (cl_uint j = 0; j < deviceCount; j++) {
            char deviceName[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);

            std::cout << "  Device " << j + 1 << ": " << deviceName << std::endl;
        }

        // Очищение
        delete[] devices;
    }

    // Очищение
    delete[] platforms;

    return 0;
}
