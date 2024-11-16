#include <CL/sycl.hpp>
#include <iostream>

class hello_world;

int main(int, char**) {
    auto device_selector = cl::sycl::default_selector_v;
    
    cl::sycl::queue queue(device_selector);

    std::cout << "Running on: "
              << queue.get_device().get_info<cl::sycl::info::device::name>()
              << "\n";
    
    queue.submit([&] (cl::sycl::handler& cgh) {
        auto os = cl::sycl::stream{128, 128, cgh};
        cgh.single_task<hello_world>([=]() { 
            os << "Hello World! (on device)\n"; 
        });
    });
    
    return 0;
}