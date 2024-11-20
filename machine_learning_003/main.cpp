#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

// Установка констант
const float LEARNING_RATE = 0.01f;
const int EPOCHS = 10'000;
const int INPUT_SIZE = 1;
const int HIDDEN_SIZE = 2;
const int OUTPUT_SIZE = 1;

// Функция сигмоида
float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

// Производная сигмоиды
float sigmoid_derivative(float z) {
    return z * (1 - z);
}

// Функция для вывода информации об устройстве
void print_device_info(const sycl::device& device) {
    std::cout << "Selected device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Device vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Device version: " << device.get_info<sycl::info::device::version>() << std::endl;
}

int main() {
    // Переменная для выбора устройства: true для GPU, false для CPU
    bool use_gpu = true;

    sycl::queue q;
        
    // Создание очереди с выбранным устройством
    if (use_gpu) {
        q = sycl::queue(sycl::gpu_selector{});
    } else {
        q = sycl::queue(sycl::cpu_selector{});
    }

    // Вывод информации об устройстве
    print_device_info(q.get_device());

    // Пример обучающих данных
    std::vector<std::pair<float, int>> data = {
        {0.5, 0}, {1.5, 0}, {3.0, 1}, {4.0, 1}, {5.5, 1}
    };

    // Инициализация весов и биасов случайным образом
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

    std::vector<float> weights_hidden(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> bias_hidden(HIDDEN_SIZE);
    std::vector<float> weights_output(HIDDEN_SIZE * OUTPUT_SIZE);
    float bias_output = dis(gen);

    for (auto& weight : weights_hidden) weight = dis(gen);
    for (auto& bias : bias_hidden) bias = dis(gen);
    for (auto& weight : weights_output) weight = dis(gen);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_rmse = 0.0f;

        for (const auto& point : data) {
            float x = point.first;
            int y = point.second;

            // Прямое прохождение
            float hidden_layer[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                hidden_layer[i] = sigmoid(x * weights_hidden[i] + bias_hidden[i]);
            }

            float output = sigmoid(std::inner_product(hidden_layer, hidden_layer + HIDDEN_SIZE, weights_output.begin(), bias_output));

            // Ошибка и RMSE
            float error = y - output;
            total_rmse += error * error;

            // Обратное распространение
            float d_output = error * sigmoid_derivative(output);

            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                weights_output[i] += LEARNING_RATE * d_output * hidden_layer[i];
            }
            bias_output += LEARNING_RATE * d_output;

            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                float d_hidden = d_output * weights_output[i] * sigmoid_derivative(hidden_layer[i]);
                weights_hidden[i] += LEARNING_RATE * d_hidden * x;
                bias_hidden[i] += LEARNING_RATE * d_hidden;
            }
        }

        total_rmse = std::sqrt(total_rmse / data.size());
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " - Total RMSE: " << total_rmse << std::endl;
        }
    }

    return 0;
}
