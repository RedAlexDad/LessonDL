import ctypes

lib = ctypes.CDLL('./build/libLessonDL.so')

# Определение функций из библиотеки на C/C++
lib.train_model.restype = ctypes.c_int
lib.train_model.argtypes = [ctypes.c_char_p, ctypes.c_int]

lib.predict.restype = ctypes.c_float
lib.predict.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

# Функции для взаимодействия с машинным обучением на C/C++
def train_model(model_path, epochs):
    return lib.train_model(model_path.encode(), epochs)

def predict(data):
    data_array = (ctypes.c_float * len(data))(*data)
    return lib.predict(data_array, len(data))


# Пример использования
if __name__ == "__main__":
    model_path = "model.pth"
    epochs = 20
    train_model(model_path, epochs)

    data = [0.5, 0.6, 0.2, 0.8]
    prediction = predict(data)
    print("Prediction:", prediction)