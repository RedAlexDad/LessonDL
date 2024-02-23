#include <torch/torch.h>
#include <iostream>

int main() {
    // Создание тензора
    torch::Tensor tensor = torch::randn({2, 3});

    // Вывод размера тензора
    std::cout << "Tensor size: " << tensor.sizes() << std::endl;

    return 0;
}
