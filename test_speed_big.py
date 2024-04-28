import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from tabulate import tabulate

INPUT_SIZE = 500
OUTPUT_SIZE = 100

# Пример данных (замените на свои реальные данные)
class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
        self.data = torch.randn(size, input_size)
        self.targets = torch.randn(size, output_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# Определение более сложной модели
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Функция для обучения модели
def train_model(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


# Функция для предсказания с использованием обученной модели
def predict(model, device, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
    return predictions


# Создание и обучение модели на CPU
model_cpu = ComplexModel()
device_cpu = torch.device("cpu")
model_cpu.to(device_cpu)
train_dataset_cpu = RandomDataset()
test_dataset_cpu = RandomDataset()
train_loader_cpu = torch.utils.data.DataLoader(train_dataset_cpu, batch_size=64, shuffle=True)
test_loader_cpu = torch.utils.data.DataLoader(test_dataset_cpu, batch_size=64)

criterion_cpu = nn.MSELoss()
optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(5):
    train_loss_cpu = train_model(model_cpu, device_cpu, train_loader_cpu, optimizer_cpu, criterion_cpu)
    print(f"Epoch {epoch + 1}/5, Loss: {train_loss_cpu:.4f}")
end_time = time.time()
cpu_training_time = end_time - start_time

start_time = time.time()
predictions_cpu = predict(model_cpu, device_cpu, test_loader_cpu)
end_time = time.time()
cpu_prediction_time = end_time - start_time

# Создание и обучение модели на GPU
model_gpu = ComplexModel()
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_gpu.to(device_gpu)
train_dataset_gpu = RandomDataset()
test_dataset_gpu = RandomDataset()
train_loader_gpu = torch.utils.data.DataLoader(train_dataset_gpu, batch_size=64, shuffle=True)
test_loader_gpu = torch.utils.data.DataLoader(test_dataset_gpu, batch_size=64)

criterion_gpu = nn.MSELoss()
optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(5):
    train_loss_gpu = train_model(model_gpu, device_gpu, train_loader_gpu, optimizer_gpu, criterion_gpu)
    print(f"Epoch {epoch + 1}/5, Loss: {train_loss_gpu:.4f}")
end_time = time.time()
gpu_training_time = end_time - start_time

start_time = time.time()
predictions_gpu = predict(model_gpu, device_gpu, test_loader_gpu)
end_time = time.time()
gpu_prediction_time = end_time - start_time

# Рассчет процентного соотношения
training_ratio_gpu = cpu_training_time / gpu_training_time * 100
prediction_ratio_gpu = cpu_prediction_time / gpu_prediction_time * 100

# Сохранение результатов в табличной форме с использованием tabulate
results = pd.DataFrame({
    'Metric': ['Training Time', 'Prediction Time'],
    'CPU': [cpu_training_time, cpu_prediction_time],
    'GPU': [gpu_training_time, gpu_prediction_time]
})

# Вывод результатов с использованием tabulate
table = tabulate(results, headers='keys', tablefmt='grid', showindex=False)
print(table)

# Рассчет процентного соотношения
training_ratio = cpu_training_time / gpu_training_time * 100
prediction_ratio = cpu_prediction_time / gpu_prediction_time * 100

# Определение, кто быстрее
training_speedup_indicator = "GPU" if training_ratio > 100 else "CPU"
prediction_speedup_indicator = "GPU" if prediction_ratio > 100 else "CPU"

# Создание таблицы для процентных соотношений
ratios_table = pd.DataFrame({
    'Metric': ['Training Speedup', 'Prediction Speedup'],
    'Speedup Ratio (%)': [training_ratio, prediction_ratio],
    'What fast?': [training_speedup_indicator, prediction_speedup_indicator]
})

# Вывод таблицы с использованием tabulate
ratios_output = tabulate(ratios_table, headers='keys', tablefmt='grid', showindex=False)
print(ratios_output)
