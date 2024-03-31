#include <iostream>
#include <tensorflow/c/c_api.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

int main() {
    // Инициализация генератора случайных чисел
    std::srand(std::time(nullptr));

    // Генерация случайных данных
    const int num_elements = 10 * 10;
    float input_data[num_elements];
    for (int i = 0; i < num_elements; ++i) {
        input_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Создание сессии и графа
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Status* status = TF_NewStatus();

// Установка числа потоков для параллельного выполнения операций между узлами (межоперационная параллельность)
    TF_SetConfig(session_opts, "inter_op_parallelism_threads", "1");

// Установка числа потоков для параллельного выполнения операций внутри узлов (внутриоперационная параллельность)
    TF_SetConfig(session_opts, "intra_op_parallelism_threads", "1");

    // Создание графа
    TF_Graph* graph = TF_NewGraph();

    // Создание операций
    TF_Output inputs[1] = {TF_Output{TF_GraphOperationByName(graph, "input"), 0}};
    TF_Output outputs[1] = {TF_Output{TF_GraphOperationByName(graph, "output"), 0}};

    // Создание сессии
    TF_Session* session = TF_NewSession(graph, session_opts, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Ошибка при создании сессии: " << TF_Message(status) << std::endl;
        TF_DeleteSessionOptions(session_opts);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }

    // Замер времени выполнения
    auto start_time = std::chrono::steady_clock::now();

    // Выполнение модели
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, nullptr, 0, input_data, sizeof(input_data), TF_DeleteTensor, nullptr);
    TF_Tensor* output_tensor;
    TF_SessionRun(session, nullptr, inputs, &input_tensor, 1, outputs, &output_tensor, 1, nullptr, 0, nullptr, status);
    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Ошибка при выполнении сессии: " << TF_Message(status) << std::endl;
        TF_DeleteTensor(input_tensor);
        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(session_opts);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 1;
    }

    // Вывод времени выполнения
    auto end_time = std::chrono::steady_clock::now();
    std::cout << "Время выполнения: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0 << " секунд" << std::endl;

    // Очистка памяти
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    return 0;
}
