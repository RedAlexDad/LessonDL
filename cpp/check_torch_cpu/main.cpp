#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Device device(torch::kCPU);
    torch::Tensor tensor = torch::eye(3, torch::TensorOptions().device(device));
    std::cout << tensor << std::endl;
}
