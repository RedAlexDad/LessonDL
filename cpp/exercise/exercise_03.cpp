#include <iostream>
#include <tensorflow/c/c_api.h>

int main() {
    // Инициализация TensorFlow
    TF_Status* status = TF_NewStatus();
    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* session_options = TF_NewSessionOptions();

    // Параметры
    const int n_samples = 1000;
    const float sigma = 1.0;
    const int n = 6;

    // Создание константных узлов для параметров
    TF_Output const_nodes[3];
    const_nodes[0] = {TF_GraphOperationByName(graph, "n_samples"), 0};
    const_nodes[1] = {TF_GraphOperationByName(graph, "sigma"), 0};
    const_nodes[2] = {TF_GraphOperationByName(graph, "n"), 0};

    // Установка значений параметров
    TF_Tensor* const_values[3];
    const_values[0] = TF_AllocateTensor(TF_INT32, nullptr, 0, sizeof(int));
    const_values[1] = TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(float));
    const_values[2] = TF_AllocateTensor(TF_INT32, nullptr, 0, sizeof(int));

    *static_cast<int*>(TF_TensorData(const_values[0])) = n_samples;
    *static_cast<float*>(TF_TensorData(const_values[1])) = sigma;
    *static_cast<int*>(TF_TensorData(const_values[2])) = n;

    // Входные тензоры для x_t и y_t
    TF_Output input_nodes[2];
    input_nodes[0] = {TF_GraphOperationByName(graph, "x_t"), 0};
    input_nodes[1] = {TF_GraphOperationByName(graph, "y_t"), 0};

    // Создание сессии
    TF_Session* session = TF_NewSession(graph, session_options, status);

    // Запуск сессии
    TF_SessionRun(session, nullptr, const_nodes, const_values, 3, nullptr, nullptr, 0, nullptr, 0, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Ошибка при запуске сессии: " << TF_Message(status) << std::endl;
        TF_DeleteGraph(graph);
        TF_DeleteSessionOptions(session_options);
        TF_DeleteStatus(status);
        return 1;
    }

    // Получение результатов
    TF_Tensor* result_tensor = nullptr;
    TF_SessionRun(session, nullptr, input_nodes, nullptr, 2, nullptr, &result_tensor, 0, nullptr, 0, nullptr, status);

    if (TF_GetCode(status) == TF_OK) {
        std::cout << "Результаты успешно получены." << std::endl;
        // Здесь вы можете обрабатывать результаты, например, выводить их на экран.
    } else {
        std::cerr << "Ошибка при получении результатов: " << TF_Message(status) << std::endl;
    }

    // Освобождение ресурсов
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_options);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    return 0;
}
