#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// Функция сигмоиды
float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Функция потерь (бинарная кросс-энтропия)
float binary_cross_entropy(float y_true, float y_pred) {
    return -(y_true * std::log(y_pred) + (1 - y_true) * std::log(1 - y_pred));
}

// Функция для вычисления RMSE
float calculate_rmse(const std::vector<float>& y_true, const std::vector<float>& y_pred) {
    float sum_squared_error = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        sum_squared_error += std::pow(y_true[i] - y_pred[i], 2);
    }
    return std::sqrt(sum_squared_error / y_true.size());
}

int main() {
    // Пример данных
    std::vector<float> X = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<float> y = {0.0, 0.0, 1.0, 1.0, 1.0};

    // Инициализация параметров модели
    float weight = 0.0;
    float bias = 0.0;
    float learning_rate = 0.01;
    int epochs = 1000;

    // Создаем очередь для выполнения на устройстве по умолчанию
    sycl::queue q;

    // Создаем буферы для данных
    sycl::buffer<float, 1> buffer_X(X.data(), sycl::range<1>(X.size()));
    sycl::buffer<float, 1> buffer_y(y.data(), sycl::range<1>(y.size()));
    sycl::buffer<float, 1> buffer_y_pred(y.size());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        q.submit([&](sycl::handler& h) {
            // Получаем доступ к буферам
            auto accessor_X = buffer_X.get_access<sycl::access::mode::read>(h);
            auto accessor_y = buffer_y.get_access<sycl::access::mode::read>(h);
            auto accessor_y_pred = buffer_y_pred.get_access<sycl::access::mode::write>(h);

            // Параллельное выполнение
            h.parallel_for(sycl::range<1>(X.size()), [=](sycl::id<1> i) {
                float linear_model = weight * accessor_X[i] + bias;
                float prediction = sigmoid(linear_model);
                accessor_y_pred[i] = prediction;
            });
        });

        // Обновление параметров модели
        // float gradient_weight = 0.0;
        // float gradient_bias = 0.0;
        // for (std::size_t i = 0; i < X.size(); ++i) {
        //     float prediction = sigmoid(weight * X[i] + bias);
        //     float error = prediction - y[i];
        //     gradient_weight += error * X[i];
        //     gradient_bias += error;
        // }
        // gradient_weight /= X.size();
        // gradient_bias /= X.size();

        // weight -= learning_rate * gradient_weight;
        // bias -= learning_rate * gradient_bias;
    }

    // Вычисление предсказаний
    // std::vector<float> y_pred(y.size());
    // q.submit([&](sycl::handler& h) {
    //     auto accessor_y_pred = buffer_y_pred.get_access<sycl::access::mode::read>(h);
    //     h.parallel_for(sycl::range<1>(y.size()), [=](sycl::id<1> i) {
    //         y_pred[i] = accessor_y_pred[i];
    //     });
    // }).wait();

    // Вычисление RMSE
    // float rmse = calculate_rmse(y, y_pred);
    // std::cout << "RMSE: " << rmse << std::endl;

    return 0;
}
