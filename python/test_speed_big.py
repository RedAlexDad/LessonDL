# НЕ УБИРАЙТЕ ЭТУ ЯЧЕЙКУ, ИНАЧНЕ БУДЕТ НЕПРАВИЛЬНО ИНИЦИАЛИЗИРОВАНО ОКРУЖЕНИЕ, ЧТО И ВЫВЕДЕТ ОШИБКУ ВЕРСИИ ptxas!!!
import os

os.environ['PATH'] = '/usr/local/cuda-12.3/bin:' + os.environ['PATH']

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import time
from tabulate import tabulate


# Создание сложной модели
def create_complex_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


# Загрузка данных CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормализация данных
x_train, x_test = x_train / 255.0, x_test / 255.0

# # Создание модели на CPU
print('Создание модели на CPU')
model_cpu = create_complex_model()

# Обучение на CPU
start_time = time.time()
model_cpu.fit(x_train, y_train, epochs=5)
end_time = time.time()
cpu_training_time = end_time - start_time

# Оценка на тестовых данных
start_time = time.time()
test_loss, test_acc = model_cpu.evaluate(x_test, y_test, verbose=2)
end_time = time.time()
cpu_evaluation_time = end_time - start_time

# Создание модели на GPU
print('Создание модели на GPU')
with tf.device('/device:GPU:0'):
    model_gpu = create_complex_model()

# Обучение на GPU
start_time = time.time()
model_gpu.fit(x_train, y_train, epochs=5)
end_time = time.time()
gpu_training_time = end_time - start_time

# Оценка на тестовых данных
start_time = time.time()
test_loss, test_acc = model_gpu.evaluate(x_test, y_test, verbose=2)
end_time = time.time()
gpu_evaluation_time = end_time - start_time

# Расчет процентного соотношения
print('Расчет процентного соотношения')
cpu_performance_ratio = (cpu_training_time + cpu_evaluation_time) / (gpu_training_time + gpu_evaluation_time) * 100

# Табличный вывод
table = [
    ["Device", "Training Time (s)", "Evaluation Time (s)", "Total Time (s)"],
    ["CPU", f"{cpu_training_time:.2f}", f"{cpu_evaluation_time:.2f}", f"{cpu_training_time + cpu_evaluation_time:.2f}"],
    ["GPU", f"{gpu_training_time:.2f}", f"{gpu_evaluation_time:.2f}", f"{gpu_training_time + gpu_evaluation_time:.2f}"],
    ["Performance Ratio", "", "", f"{cpu_performance_ratio:.2f}%"]
]

print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))