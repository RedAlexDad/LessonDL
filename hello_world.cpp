// Стандартные включения языка C
#include <stdio.h>

// Включения для OpenCL
#include <CL/cl2.hpp>

int main() {
    cl_int CL_err = CL_SUCCESS;
    cl_uint numPlatforms = 0;

    // Получаем количество доступных OpenCL-платформ
    CL_err = clGetPlatformIDs(0, NULL, &numPlatforms);

    if (CL_err == CL_SUCCESS) {
        printf("%u платформ(а) найдено\n", numPlatforms);

        if (numPlatforms > 0) {
            // Получаем информацию о каждой платформе
            cl_platform_id *platforms = new cl_platform_id[numPlatforms];
            CL_err = clGetPlatformIDs(numPlatforms, platforms, NULL);

            for (cl_uint i = 0; i < numPlatforms; ++i) {
                printf("\nИнформация о платформе #%u:\n", i + 1);

                // Получаем имя платформы
                size_t platformNameSize = 0;
                CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &platformNameSize);
                char *platformName = new char[platformNameSize];
                CL_err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platformNameSize, platformName, NULL);
                printf("  Имя платформы: %s\n", platformName);
                delete[] platformName;

                // Получаем количество устройств на платформе
                cl_uint numDevices = 0;
                CL_err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
                printf("  Устройств на платформе: %u\n", numDevices);

                if (numDevices > 0) {
                    // Получаем информацию о каждом устройстве
                    cl_device_id *devices = new cl_device_id[numDevices];
                    CL_err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

                    for (cl_uint j = 0; j < numDevices; ++j) {
                        printf("    Информация об устройстве #%u:\n", j + 1);

                        // Получаем имя устройства
                        size_t deviceNameSize = 0;
                        CL_err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &deviceNameSize);
                        char *deviceName = new char[deviceNameSize];
                        CL_err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, deviceNameSize, deviceName, NULL);
                        printf("      Имя устройства: %s\n", deviceName);
                        delete[] deviceName;
                    }

                    delete[] devices;
                }
            }

            delete[] platforms;
        }
    } else {
        printf("Ошибка при вызове clGetPlatformIDs(%i)\n", CL_err);
    }
    return 0;
}
