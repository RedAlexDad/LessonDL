#include <CL/sycl.hpp>
#include <iostream>

int main() {
    try {
        auto platforms = cl::sycl::platform::get_platforms();
        for (const auto& platform : platforms) {
            std::cout << "Platform: " << platform.get_info<cl::sycl::info::platform::name>() << "\n";

            auto devices = platform.get_devices();
            for (const auto& device : devices) {
                std::cout << "  Device: "<< device.get_info<cl::sycl::info::device::name>() << "\n";
            }
        }
    } catch (cl::sycl::exception const& e) {
        std::cout << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
