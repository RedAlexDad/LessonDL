#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <armadillo>
#include <CL/cl.h>

// Функция для проверки ошибок OpenCL
void checkError(cl_int error, const char* message) {
    if (error != CL_SUCCESS) {
        std::cerr << message << ": Error " << error << std::endl;
        std::exit(1);
    }
}

// Функция проверки наличия расширения fp64
bool checkFp64Support(cl_device_id device) {
    char extensions[1024];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, nullptr);
    return strstr(extensions, "cl_khr_fp64") != nullptr;
}

// Функция для выбора устройства GPU
cl_device_id selectGPUDevice() {
    cl_int err;
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    for (auto platform : platforms) {
        cl_uint numDevices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);

            for (auto device : devices) {
                if (checkFp64Support(device)) {
                    return device;
                }
            }
        }
    }

    throw std::runtime_error("No suitable GPU device with double precision support found.");
}

class HybridMemoryPolynomialModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    arma::cx_mat phi_HMP;  // Матрица фич
    int K;    // Порядок нелинейности для полинома памяти
    int M;    // Глубина памяти для полинома памяти
    int K_e;  // Порядок нелинейности для огибающей полинома памяти
    int M_e;  // Глубина памяти для огибающей полинома памяти
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    const char* kernelSource = R"(
    __kernel void calculateFeatures(
        __global const float2* input,
        __global float2* features,
        const unsigned int K,
        const unsigned int M,
        const unsigned int K_e,
        const unsigned int M_e,
        const unsigned int maxTerms,
        const unsigned int n_samples
    ) {
        int n = get_global_id(0);
        
        // Массив для хранения фич
        __local float2 terms[1024]; // Установите подходящий размер (максимальное значение maxTerms)
        
        if (n < n_samples) {
            for (int i = 0; i < maxTerms; ++i) {
                terms[i] = (float2)(0.0f, 0.0f);
            }

            // Вычисление MP терминов
            int idx = 0;
            for (int m = 0; m <= M; ++m) {
                if (n - m >= 0) {
                    float2 data_n_m = input[n - m];
                    float magnitude = sqrt(data_n_m.x * data_n_m.x + data_n_m.y * data_n_m.y);
                    for (int k = 1; k <= K; ++k) {
                        float scale = pow(magnitude, k - 1);
                        terms[idx].x += data_n_m.x * scale;
                        terms[idx].y += data_n_m.y * scale;
                        idx++;
                    }
                }
            }

            // Вычисление Envelope MP терминов
            for (int m = 1; m <= M_e; ++m) {
                if (n - m >= 0) {
                    float2 data_n_m = input[n - m];
                    float2 data_n = input[n];
                    float magnitude = sqrt(data_n_m.x * data_n_m.x + data_n_m.y * data_n_m.y);
                    for (int k = 2; k <= K_e; ++k) {
                        float scale = pow(magnitude, k - 1);
                        terms[idx].x += data_n.x * scale;
                        terms[idx].y += data_n.y * scale;
                        idx++;
                    }
                }
            }

            // Сохранение результатов в выходной буфер
            for (int i = 0; i < maxTerms; ++i) {
                features[n * maxTerms + i] = terms[i];
            }
        }
    }
    )";

    void initializeOpenCL() {
        device = selectGPUDevice();

        cl_int err;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context.");

        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL command queue.");
    }

    void buildAndRunKernel() {
        cl_int err;

        // Создание программы
        cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
        checkError(err, "Failed to create program");

        // Компиляция программы
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t logSize;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "Build log:\n" << log.data() << std::endl;
            std::exit(1);
        }

        // Создание ядра
        cl_kernel kernel = clCreateKernel(program, "calculateFeatures", &err);
        checkError(err, "Failed to create kernel");

        int n_samples = data.n_rows - std::max(M, M_e);
        int maxTerms = (M + 1) * K + M_e * (K_e - 1);
        int sizeInput = n_samples * sizeof(cl_float2);
        int sizeFeatures = n_samples * maxTerms * sizeof(cl_float2);

        // Создание буферов
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeInput, nullptr, &err);
        checkError(err, "Failed to create input buffer");
        cl_mem featuresBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeFeatures, nullptr, &err);
        checkError(err, "Failed to create features buffer");

        // Копирование данных в буфер
        std::vector<cl_float2> hostInput(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            hostInput[i] = {static_cast<float>(data(i, 0).real()), static_cast<float>(data(i, 0).imag())};
        }
        clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, sizeInput, hostInput.data(), 0, nullptr, nullptr);

        // Установка аргументов ядра
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &featuresBuffer);
        clSetKernelArg(kernel, 2, sizeof(unsigned int), &K);
        clSetKernelArg(kernel, 3, sizeof(unsigned int), &M);
        clSetKernelArg(kernel, 4, sizeof(unsigned int), &K_e);
        clSetKernelArg(kernel, 5, sizeof(unsigned int), &M_e);
        clSetKernelArg(kernel, 6, sizeof(unsigned int), &maxTerms);
        clSetKernelArg(kernel, 7, sizeof(unsigned int), &n_samples);

        // Размерность рабочей группы
        size_t globalWorkSize = n_samples;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        checkError(err, "Failed to enqueue kernel");

        // Логирование перед запуском ядра
        // std::cout << "Запуск ядра с n_samples = " << n_samples << ", maxTerms = " << maxTerms << std::endl;

        // Чтение результата
        std::vector<cl_float2> hostFeatures(n_samples * maxTerms);
        clEnqueueReadBuffer(queue, featuresBuffer, CL_TRUE, 0, sizeFeatures, hostFeatures.data(), 0, nullptr, nullptr);

        // std::cout << "Чтение завершено из OpenCL" << std::endl;
        
        // Освобождение ресурсов
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(featuresBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);

        // Обработка данных на хосте
        arma::cx_mat phi_HMP_temp(n_samples, maxTerms);
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < maxTerms; ++j) {
                phi_HMP_temp(i, j) = {hostFeatures[i * maxTerms + j].x, hostFeatures[i * maxTerms + j].y};
            }
        }
        phi_HMP = std::move(phi_HMP_temp);

        // std::cout << "Обучение модели..." << std::endl;

        // Обучение модели
        arma::cx_colvec y_trimmed = data.col(1).subvec(std::max(M, M_e), data.n_rows - 1);
        // std::cout << "Размеры phi_HMP: " << phi_HMP.n_rows << "x" << phi_HMP.n_cols << std::endl;
        // std::cout << "Размеры y_trimmed: " << y_trimmed.n_elem << std::endl;
        
        // std::cout << "phi_HMP: " << phi_HMP << std::endl;
        coefficients = arma::solve(phi_HMP, y_trimmed);

        // if (coefficients.n_elem > 0) {
        //     std::cout << "Коэффициенты успешно вычислены" << std::endl;
        // }
    }

public:
    HybridMemoryPolynomialModel(const std::string& filename, int K, int M, int K_e, int M_e)
        : K(K), M(M), K_e(K_e), M_e(M_e) {
        int rows;
        data = load_data(filename, rows);
        initializeOpenCL();
    }

    arma::cx_mat load_data(const std::string& filename, int& rows) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        std::string line;
        std::vector<std::complex<double>> inputs, outputs;

        while (std::getline(file, line)) {
            if (line.find("Time") != std::string::npos) continue;

            std::stringstream ss(line);
            std::string token;
            double real_in, imag_in;
            int index = 0;

            while (std::getline(ss, token, ',')) {
                std::replace(token.begin(), token.end(), '(', ' ');
                std::replace(token.begin(), token.end(), ')', ' ');
                std::replace(token.begin(), token.end(), 'j', ' ');
                std::stringstream val_stream(token);
                val_stream >> real_in >> imag_in;

                if (index == 1) {
                    inputs.emplace_back(real_in, imag_in);
                } else if (index == 2) {
                    outputs.emplace_back(real_in, imag_in);
                }
                ++index;
            }
        }

        rows = inputs.size();
        arma::cx_mat data(inputs.size(), 2);
        for (size_t i = 0; i < inputs.size(); ++i) {
            data(i, 0) = inputs[i];
            data(i, 1) = outputs[i];
        }

        return data;
    }

    void fit() {
        buildAndRunKernel();
    }

    arma::cx_vec predict() const {
        if (coefficients.empty() || phi_HMP.empty()) {
            throw std::runtime_error("Model is not trained. Call fit() before predict().");
        }
        return phi_HMP * coefficients;
    }

    std::pair<double, double> calculateRMSE(const arma::cx_vec& predictions) const {
        if (predictions.n_elem != data.n_rows - std::max(M, M_e)) {
            throw std::runtime_error("Error: Prediction size does not match the trimmed data size.");
        }

        arma::cx_vec y_true = data.col(1).subvec(std::max(M, M_e), data.n_rows - 1);

        double rmse_real = std::sqrt(arma::mean(arma::square(arma::real(y_true) - arma::real(predictions))));
        double rmse_imag = std::sqrt(arma::mean(arma::square(arma::imag(y_true) - arma::imag(predictions))));

        return {rmse_real, rmse_imag};
    }

    arma::cx_mat getData() const {
        return data;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <filename> <K> <M> <K_e> <M_e>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int M = std::stoi(argv[3]);
    int K_e = std::stoi(argv[4]);
    int M_e = std::stoi(argv[5]);

    try {
        cl_device_id device = selectGPUDevice();

        // Получение и вывод имени устройства
        size_t deviceNameSize;
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize);
        std::vector<char> deviceName(deviceNameSize);
        clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr);

        std::cout << "Selected GPU Device: " << std::string(deviceName.begin(), deviceName.end()) << std::endl;

        // Создание контекста и очереди команд
        cl_int err;
        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create context.");

        cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue.");
        
        // Далее выполняйте OpenCL задачи на этом устройстве

        // Освободить ресурсы
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    try {
        // Создаем модель с загрузкой из файла
        HybridMemoryPolynomialModel model(filename, K, M, K_e, M_e);

        // Обучение
        model.fit();

        // Предсказания
        arma::cx_vec predictions = model.predict();


        auto [rmse_real, rmse_imag] = model.calculateRMSE(predictions);

        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

        // // Выводим предсказания
        // for (int i = 0; i < predictions.n_elem; ++i) {
        //     std::cout << "Prediction[" << i << "] = (" 
        //               << predictions[i].real() << ", " 
        //               << predictions[i].imag() << ")" << std::endl;
        // }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
