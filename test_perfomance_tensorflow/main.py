import tensorflow as tf
import time

# Генерация случайных данных
input_data = tf.random.uniform((10000, 1000))

# Создание модели с некоторыми операциями
def model(input_data):
    x = tf.keras.layers.Dense(512, activation='relu')(input_data)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    return x

# Замер времени выполнения
start_time = time.time()

# Выполнение модели
output = model(input_data)

# Вывод времени выполнения
print("Время выполнения:", time.time() - start_time, "секунд")