#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <CL/cl.h>

// Функция для проверки ошибок OpenCL
void checkError(cl_int error, const char* message) {
    if (error != CL_SUCCESS) {
        std::cerr << message << ": Error " << error << std::endl;
        std::exit(1);
    }
}

class HybridMemoryPolynomialModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    int K;    // Порядок нелинейности для полинома памяти
    int M;    // Глубина памяти для полинома памяти
    int K_e;  // Порядок нелинейности для огибающей полинома памяти
    int M_e;  // Глубина памяти для огибающей полинома памяти

    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    const char* kernelSource = R"(
    __kernel void calculateTerms(
        __global const float2* input,
        __global float2* output,
        const unsigned int K,
        const unsigned int M,
        const unsigned int K_e,
        const unsigned int M_e,
        const unsigned int n_samples
    ) {
        int n = get_global_id(0);
        if (n < n_samples) {
            float2 result = (float2)(0.0f, 0.0f);

            // Вычисляем термины для полинома памяти
            for (int m = 0; m <= M; ++m) {
                if (n - m >= 0) {
                    float2 data_n_m = input[n - m];
                    float magnitude = sqrt(data_n_m.x * data_n_m.x + data_n_m.y * data_n_m.y);

                    for (int k = 1; k <= K; ++k) {
                        float scale = pow(magnitude, k - 1);
                        result.x += data_n_m.x * scale;
                        result.y += data_n_m.y * scale;
                    }
                }
            }

            // Вычисляем термины для огибающей полинома памяти
            for (int m = 1; m <= M_e; ++m) {
                if (n - m >= 0) {
                    float2 data_n_m = input[n - m];
                    float2 data_n = input[n];
                    float magnitude = sqrt(data_n_m.x * data_n_m.x + data_n_m.y * data_n_m.y);

                    for (int k = 2; k <= K_e; ++k) {
                        float scale = pow(magnitude, k - 1);
                        result.x += data_n.x * scale;
                        result.y += data_n.y * scale;
                    }
                }
            }

            output[n] = result;
        }
    }
    )";

    void initializeOpenCL() {
        cl_int err;
        cl_uint platformCount;

        // Получение количества платформ
        clGetPlatformIDs(0, nullptr, &platformCount);
        std::vector<cl_platform_id> platforms(platformCount);
        clGetPlatformIDs(platformCount, platforms.data(), nullptr);

        // Получение устройства (для простоты используем первое доступное устройство)
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
        device = devices[0];

        // Создание контекста
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        checkError(err, "Failed to create context");

        // Создание очереди команд
        queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
        checkError(err, "Failed to create command queue");
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
        cl_kernel kernel = clCreateKernel(program, "calculateTerms", &err);
        checkError(err, "Failed to create kernel");

        // Размер данных
        int n_samples = data.n_rows - std::max(M, M_e);
        int size = n_samples * sizeof(cl_float2);

        std::vector<cl_float2> host_input(n_samples), host_output(n_samples);
        // Преобразование данных в формат, используемый в ядре (пример)
        for (int i = 0; i < n_samples; ++i) {
            host_input[i] = {(float)data(i, 0).real(), (float)data(i, 0).imag()};
        }

        // Создание буферов
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, host_input.data(), &err);
        checkError(err, "Failed to create input buffer");
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, nullptr, &err);
        checkError(err, "Failed to create output buffer");

        // Установка аргументов ядра
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
        clSetKernelArg(kernel, 2, sizeof(unsigned int), &K);
        clSetKernelArg(kernel, 3, sizeof(unsigned int), &M);
        clSetKernelArg(kernel, 4, sizeof(unsigned int), &K_e);
        clSetKernelArg(kernel, 5, sizeof(unsigned int), &M_e);
        clSetKernelArg(kernel, 6, sizeof(unsigned int), &n_samples);

        // Определение размерности рабочей группы
        size_t globalWorkSize = n_samples;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        checkError(err, "Failed to enqueue kernel");

        // Чтение данных обратно на хост
        err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, size, host_output.data(), 0, nullptr, nullptr);
        checkError(err, "Failed to read buffer");

        // Преобразование данных обратно в нужный формат (пример)
        for (int i = 0; i < n_samples; ++i) {
            data(i, 1) = std::complex<double>(host_output[i].s[0], host_output[i].s[1]);
        }

        // Освобождение ресурсов
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }

public:
    HybridMemoryPolynomialModel(const std::string& filename, int K, int M, int K_e, int M_e)
        : K(K), M(M), K_e(K_e), M_e(M_e), coefficients(arma::cx_colvec()) {

        initializeOpenCL(); // инициализация OpenCL

        int rows;
        data = load_data(filename, rows);
        if (rows <= std::max(M, M_e)) {
            throw std::runtime_error("Error: Not enough data samples or incorrect data format.");
        }

        fit(); // Начинаем обучение модели
    }

    ~HybridMemoryPolynomialModel() {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
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
            double real_in, imag_in, real_out, imag_out;
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
        try {
            buildAndRunKernel();  // Предположим, здесь обрабатываем и создаем нужные фичи

            // Армирования проверки наличия рассчитанных данных до продолжения решения
            if (data.n_rows <= std::max(M, M_e)) {
                throw std::runtime_error("Insufficient data after kernel computation.");
            }

            int n_samples = data.n_rows - std::max(M, M_e);
            int max_terms = (M + 1) * K + M_e * (K_e - 1);
            arma::cx_mat phi_HMP(n_samples, max_terms, arma::fill::zeros);
            arma::cx_colvec y_trimmed(n_samples);

            for (int n = std::max(M, M_e); n < data.n_rows; ++n) {
                std::vector<std::complex<long double>> row;
                construct_mp_terms(row, n);
                construct_envmp_terms(row, n);

                for (size_t i = 0; i < row.size(); ++i) {
                    phi_HMP(n - std::max(M, M_e), i) = row[i];
                }

                y_trimmed(n - std::max(M, M_e)) = data(n, 1);
            }

            coefficients = arma::solve(phi_HMP, y_trimmed);

            // Проверка, что коэффициенты были правильно рассчитаны
            if (coefficients.n_elem == 0) {
                throw std::runtime_error("Failed to compute model coefficients.");
            }
        } catch (const std::exception& ex) {
            std::cerr << "Error during fitting process: " << ex.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void construct_mp_terms(std::vector<std::complex<long double>>& terms, int n) {
        for (int m = 0; m <= M; ++m) {
            for (int k = 1; k <= K; ++k) {
                std::complex<long double> term = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                terms.push_back(term);
            }
        }
    }

    void construct_envmp_terms(std::vector<std::complex<long double>>& terms, int n) {
        for (int m = 1; m <= M_e; ++m) {
            for (int k = 2; k <= K_e; ++k) {
                if (n - m >= 0) {
                    std::complex<long double> term = data(n, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                    terms.push_back(term);
                }
            }
        }
    }

    arma::cx_colvec predict() {
        if (coefficients.n_elem == 0) {
            throw std::runtime_error("Model has not been fitted yet.");
        }

        int n_samples = data.n_rows - std::max(M, M_e);
        arma::cx_colvec y_pred(n_samples);
        
        for (int n = std::max(M, M_e); n < data.n_rows; ++n) {
            std::vector<std::complex<long double>> phi_n;
            construct_mp_terms(phi_n, n);
            construct_envmp_terms(phi_n, n);

            arma::cx_rowvec phi_vec(phi_n.size());
            for (size_t i = 0; i < phi_n.size(); ++i) {
                phi_vec(i) = phi_n[i];
            }

            y_pred(n - std::max(M, M_e)) = arma::dot(phi_vec, coefficients);
        }

        return y_pred;
    }

    std::pair<long double, long double> calculate_rmse(const arma::cx_colvec& y_true, const arma::cx_colvec& y_pred) {
        long double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.size());
        long double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.size());

        return {rmse_real, rmse_imag};
    }

    arma::cx_colvec get_output_data() const {
        return data.col(1).tail(data.n_rows - std::max(M, M_e));
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 6) {
            std::cerr << "Usage: " << argv[0] << " <filename> <K> <M> <K_e> <M_e>" << std::endl;
            return 1;
        }

        std::string filename = argv[1];
        int K = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        int K_e = std::stoi(argv[4]);
        int M_e = std::stoi(argv[5]);

        HybridMemoryPolynomialModel model(filename, K, M, K_e, M_e);

        arma::cx_colvec y_pred = model.predict();
        arma::cx_colvec y_true = model.get_output_data();

        auto [rmse_real, rmse_imag] = model.calculate_rmse(y_true, y_pred);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
