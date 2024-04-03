#include <iostream>
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/cl2.hpp>

int main() {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    // Выбор платформы и устройства OpenCL
    cl::Platform default_platform;
    cl::Device default_device;

    // Перебираем платформы
    for (const auto &platform: all_platforms) {
        // Выбираем первую платформу
        default_platform = platform;

        // Получаем устройства для выбранной платформы
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        // Если устройства найдены, выбираем первое
        if (!devices.empty()) {
            default_device = devices[0];
            break;
        }
    }

    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

    return 0;
}