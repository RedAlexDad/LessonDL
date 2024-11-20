#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

// Установка констант
const float LEARNING_RATE = 0.01f;  // Скорость обучения
const float MAX_UPDATE = 0.1f;
const int EPOCHS = 100'000;  // Уменьшено для быстроты тестов

float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

int main() {
    sycl::queue q;
    std::vector<std::pair<float, int>> data = {
        {0.5, 0}, {1.5, 0}, {3.0, 1}, {4.0, 1}, {5.5, 1}
    };

    // Случайная инициализация весов
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

    float weight = dis(gen);
    float bias = dis(gen);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float temp_w = 0.0f, temp_b = 0.0f;
        sycl::buffer<float, 1> param_buf(&weight, sycl::range<1>(1));
        sycl::buffer<float, 1> bias_buf(&bias, sycl::range<1>(1));
        sycl::buffer<float, 1> temp_w_buf(&temp_w, sycl::range<1>(1));
        sycl::buffer<float, 1> temp_b_buf(&temp_b, sycl::range<1>(1));

        q.submit([&](sycl::handler &h) {
            auto param = param_buf.get_access<sycl::access::mode::read>(h);
            auto bias = bias_buf.get_access<sycl::access::mode::read>(h);
            auto temp_w = temp_w_buf.get_access<sycl::access::mode::write>(h);
            auto temp_b = temp_b_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for<class logistic_regression>(sycl::range<1>(data.size()), [=](sycl::id<1> idx) {
                float x = data[idx].first;
                int y = data[idx].second;
                float prediction = sigmoid(param[0] * x + bias[0]);
                float error = y - prediction;

                auto temp_w_atomic = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(temp_w[0]);
                auto temp_b_atomic = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(temp_b[0]);

                float update_w = LEARNING_RATE * error * x;
                float update_b = LEARNING_RATE * error;

                temp_w_atomic.fetch_add(update_w);
                temp_b_atomic.fetch_add(update_b);
            });
        }).wait();

        weight += std::clamp(temp_w_buf.get_access<sycl::access::mode::read>()[0], -MAX_UPDATE, MAX_UPDATE);
        bias += std::clamp(temp_b_buf.get_access<sycl::access::mode::read>()[0], -MAX_UPDATE, MAX_UPDATE);

        // Вычисление RMSE для текущей эпохи
        float rmse = 0.0f;
        for (const auto &point : data) {
            float x = point.first;
            int y = point.second;
            float prediction = sigmoid(weight * x + bias);
            float error = y - prediction;
            rmse += error * error;
        }
        rmse = std::sqrt(rmse / data.size());

        if (epoch % 100 == 0) {
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Epoch " << epoch << " - RMSE: " << rmse << std::endl;
        }
    }

    float final_rmse = 0.0f;
    for (const auto &point : data) {
        float x = point.first;
        int y = point.second;
        float prediction = sigmoid(weight * x + bias);
        float error = y - prediction;
        final_rmse += error * error;
    }
    final_rmse = std::sqrt(final_rmse / data.size());

    std::cout << "Final Model - Weight: " << weight << ", Bias: " << bias << std::endl;
    std::cout << "Final RMSE: " << final_rmse << std::endl;

    return 0;
}
