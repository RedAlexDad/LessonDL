import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Вывод информации о доступных устройствах
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU devices available.")
else:
    print("GPU devices:")
    for device in physical_devices:
        print(f"- {device.name} ({device.device_type})")

# Загрузка датасета
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание простой модели и выполнение вычислений на GPU
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(784,), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели на GPU
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Оценка модели на GPU
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
