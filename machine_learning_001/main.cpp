#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <limits>

const float LEARNING_RATE = 0.1f;  // Уменьшена скорость обучения
const float MAX_UPDATE = 1.0f;  // Ограничение на максимальное изменение веса

int main() {
    sycl::queue q;
    std::vector<std::pair<float, float>> data = {{1, 2}, {2, 3}, {3, 5}, {4, 4}, {5, 5}};
    float weights[2] = {0, 0};

    for (int epoch = 0; epoch < 1000; ++epoch) {
        float temp_w0 = 0.0f, temp_w1 = 0.0f;

        sycl::buffer<float, 1> weight_buf(weights, sycl::range<1>(2));
        sycl::buffer<float, 1> temp_w_buf(sycl::range<1>(2));  // Буфер для временных весов

        q.submit([&](sycl::handler &h) {
            auto w = weight_buf.get_access<sycl::access::mode::read>(h);
            auto temp_w = temp_w_buf.get_access<sycl::access::mode::read_write>(h);

            h.parallel_for<class gradient_descent>(sycl::range<1>(data.size()), [=](sycl::id<1> idx) {
                float x = data[idx].first;
                float y = data[idx].second;
                float prediction = w[0] + w[1] * x;
                float loss = y - prediction;

                // Используем атомарные обновления с помощью atomic_ref
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
                    temp_w0_atomic(temp_w[0]);
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>
                    temp_w1_atomic(temp_w[1]);

                float update_w0 = LEARNING_RATE * loss;
                float update_w1 = LEARNING_RATE * loss * x;

                // Ограничиваем обновления
                update_w0 = sycl::clamp(update_w0, -MAX_UPDATE, MAX_UPDATE);
                update_w1 = sycl::clamp(update_w1, -MAX_UPDATE, MAX_UPDATE);

                temp_w0_atomic.fetch_add(update_w0);
                temp_w1_atomic.fetch_add(update_w1);
            });
        }).wait();

        // Чтение накопленных изменений и обновление весов
        {
            auto temp_w_read = temp_w_buf.get_access<sycl::access::mode::read>();
            weights[0] += temp_w_read[0];
            weights[1] += temp_w_read[1];
        }

        // Вычисление ошибки
        float error = 0.0f;
        for (const auto &point : data) {
            float x = point.first;
            float y = point.second;
            float prediction = weights[0] + weights[1] * x;
            error += (y - prediction) * (y - prediction);
        }
        error /= data.size();

        std::cout.precision(6);
        if (epoch % 100 == 0 || epoch == 999) {
            std::cout << "Epoch " << epoch << " - Error: " << error << std::endl;
        }
    }

    std::cout.precision(6);
    std::cout << "Final weights: [" << weights[0] << ", " << weights[1] << "]" << std::endl;
    return 0;
}

/*
(base) redalexdad@redalexdad-Nitro-AN515-44:~/GitHub/LessonDL/build$ /home/redalexdad/GitHub/LessonDL/build/machine_learning_001_main
Epoch 0 - Error: 112.447
Epoch 100 - Error: 50.5993
Epoch 200 - Error: 50.1527
Epoch 300 - Error: 51
Epoch 400 - Error: 50.5037
Epoch 500 - Error: 50.3806
Epoch 600 - Error: 50.1742
Epoch 700 - Error: 49.4549
Epoch 800 - Error: 50.1159
Epoch 900 - Error: 50.8354
Epoch 999 - Error: 59.0712
Final weights: [1.57547, -1.57587]
*/