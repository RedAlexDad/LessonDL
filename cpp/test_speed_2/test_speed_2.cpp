#include <iostream>
#include <tensorflow/c/c_api.h>

// Создание простых данных для обучения
const int data_size = 1000;
float training_data[data_size][1];
int labels[data_size];

void generate_data() {
    for (int i = 0; i < data_size; ++i) {
        training_data[i][0] = static_cast<float>(rand()) / RAND_MAX;
        labels[i] = (training_data[i][0] > 0.5) ? 1 : 0;
    }
}

int main() {
    // Генерация данных
    generate_data();

    // Создание графа TensorFlow
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sessionOptions, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Failed to create TensorFlow session: " << TF_Message(status) << std::endl;
        TF_DeleteGraph(graph);
        TF_DeleteSessionOptions(sessionOptions);
        TF_DeleteStatus(status);
        return 1;
    }

    // Создание простой модели с одним слоем
    const int input_size = 1;
    const int output_size = 2;

    TF_Operation* input_operation = TF_Placeholder(graph, TF_FLOAT, TF_Dimensions{data_size, input_size}, "input");
    TF_Operation* weight = TF_Const(graph, TF_Float(1.0f), TF_Dimensions{input_size, output_size}, "weight");
    TF_Operation* bias = TF_Const(graph, TF_Float(0.0f), TF_Dimensions{output_size}, "bias");

    TF_Operation* multiply = TF_MatMul(graph, input_operation, weight, TF_Transpose_a | TF_Transpose_b, "multiply");
    TF_Operation* add = TF_Add(graph, multiply, bias, "add");
    TF_Operation* output = TF_Softmax(graph, add, "output");

    // Определение операций графа для обучения
    TF_Operation* label_placeholder = TF_Placeholder(graph, TF_INT32, TF_Dimensions{data_size}, "labels");
    TF_Operation* cross_entropy = TF_SoftmaxCrossEntropyWithLogits(graph, output, label_placeholder, "cross_entropy");
    TF_Operation* mean_loss = TF_Mean(graph, cross_entropy, TF_Int32(0), "mean_loss");
    TF_Operation* train_step = TF_TrainAdam(graph, mean_loss, 0.01, 0.9, 0.999, 1e-08, 0.01, "train_step");

    // Запуск обучения
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, TF_Dimensions{data_size, input_size}, training_data, sizeof(training_data), [](void* data, size_t, void*) { delete[] static_cast<float*>(data); }, nullptr);
    TF_Tensor* label_tensor = TF_NewTensor(TF_INT32, TF_Dimensions{data_size}, labels, sizeof(labels), nullptr, nullptr);

    TF_Tensor* input_tensors[] = {input_tensor, label_tensor};
    TF_Operation* output_operations[] = {train_step};

    TF_SessionRun(session, nullptr, nullptr, nullptr, 0, input_tensors, output_operations, 1, nullptr, 0, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Training failed: " << TF_Message(status) << std::endl;
    } else {
        std::cout << "Training successful!" << std::endl;
    }

    // Освобождение ресурсов
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(label_tensor);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sessionOptions);
    TF_DeleteStatus(status);

    return 0;
}
