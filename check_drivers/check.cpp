#include <iostream>
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/cl2.hpp>


int main(){
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << "No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    std::cout << "Available OpenCL platforms:\n";
    for (size_t i = 0; i < all_platforms.size(); ++i) {
        std::cout << "Platform " << i << ": " << all_platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
    }

    // Явно выбираем платформу OpenCL (передайте индекс нужной платформы)
    cl::Platform default_platform = all_platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);

    if (all_devices.size() == 0) {
        std::cout << " No GPU devices found. Check your hardware or OpenCL installation!\n";
        exit(1);
    }

    cl::Device default_device = all_devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    return 0;
}