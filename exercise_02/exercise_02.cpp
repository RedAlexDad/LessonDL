// Определим конкретную версию
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/cl2.hpp>
#include <tensorflow/c/c_api.h>
#include <cstdio>

int main() {
    // Использование SYCL на графическом процессоре
    try {
        sycl::queue queue(sycl::gpu_selector{});
        sycl::buffer<float> buffer_a(sycl::range<1>{1});
        sycl::buffer<float> buffer_b(sycl::range<1>{1});
        sycl::buffer<float> buffer_result(sycl::range<1>{1});

        queue.submit([&](sycl::handler &cgh) {
            auto a = buffer_a.get_access<sycl::access::mode::write>(cgh);
            auto b = buffer_b.get_access<sycl::access::mode::write>(cgh);
            auto result = buffer_result.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class vector_add>(sycl::range<1>{1}, [=](sycl::item<1> item) {
                a[item] = 2.0;
                b[item] = 3.0;
                result[item] = a[item] + b[item];
            });
        });
        queue.wait();

        // Передача результата в TensorFlow на CPU
        TF_Status* status = TF_NewStatus();
        TF_Graph* graph = TF_NewGraph();
        TF_SessionOptions* session_options = TF_NewSessionOptions();

        const char* input_node_names[] = {"x", "y"};
        const char* output_node_names[] = {"z"};

        TF_Output inputs[2];
        inputs[0] = {TF_GraphOperationByName(graph, input_node_names[0]), 0};
        inputs[1] = {TF_GraphOperationByName(graph, input_node_names[1]), 0};

        TF_Tensor* input_values[2];
        input_values[0] = TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(float));
        input_values[1] = TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(float));

        *static_cast<float*>(TF_TensorData(input_values[0])) = 2.0;
        *static_cast<float*>(TF_TensorData(input_values[1])) = 3.0;

        TF_Output output = {TF_GraphOperationByName(graph, output_node_names[0]), 0};
        TF_Tensor* output_value = nullptr;

        TF_Session* session = TF_NewSession(graph, session_options, status);
        TF_SessionRun(session, nullptr, inputs, input_values, 2, &output, &output_value, 1, nullptr, 0, nullptr, status);

        // Вывод результата
        printf("Результат сложения: %f\n", *static_cast<float*>(TF_TensorData(output_value)));

        // Освобождение ресурсов
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
        TF_DeleteSessionOptions(session_options);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
    } catch (const sycl::exception &e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
