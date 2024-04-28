#include <torch/torch.h>

// Определение простой модели
struct SimpleModel : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    SimpleModel(int input_size, int output_size) {
        fc = register_module("fc", torch::nn::Linear(input_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = fc(x);
        return x;
    }
};

// Функция для обучения модели
void train_model(SimpleModel& model, torch::Device device, torch::data::DataLoader& train_loader, torch::optim::Optimizer& optimizer, int num_epochs) {
    model.to(device);

    model.train();
    float learning_rate = 0.01;

    // Обучение
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0;
        int total_samples = 0;

        for (auto& batch : train_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            optimizer.zero_grad();

            auto outputs = model.forward(data);
            auto loss = torch::mse_loss(outputs, targets);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();
            total_samples += data.size(0);
        }

        epoch_loss /= total_samples;
        std::cout << "Epoch: " << epoch << ", Loss: " << epoch_loss << std::endl;
    }
}

int main() {
    // Пример данных (замените на свои реальные данные)
    auto data = torch::randn({100, 10});
    auto targets = torch::randn({100, 1});
    auto dataset = torch::data::datasets::TensorDataset(data, targets);
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(16).workers(2));

    // Создание и обучение модели
    SimpleModel model(10, 1);
    torch::Device device(torch::kCPU);

    torch::optim::SGD optimizer(model.parameters(), 0.01);

    train_model(model, device, *train_loader, optimizer, 10);

    return 0;
}
