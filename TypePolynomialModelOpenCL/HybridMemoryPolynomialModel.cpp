#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>
#include <CL/cl.h>
#include <cmath>
#include <stdexcept>
#include <iomanip>

// Проверка поддержки расширения fp64
bool checkFp64Support(cl_device_id device) {
    char extensions[1024];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, nullptr);
    return strstr(extensions, "cl_khr_fp64") != nullptr;
}

// Выбор устройства GPU
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

// Объявление ядра для вычисления признаков
const char* kernelSource = R"CLC(
__kernel void computeFeatures(
    __global const double2* inputData,
    __global double2* featureData,
    const int K, const int M, const int K_e, const int M_e, const int nSamples) {
    int idx = get_global_id(0);
    int max_terms = (M + 1) * K + M_e * (K_e - 1);

    if (idx < nSamples) {
        int featureIdx = 0;

        // Construct MP terms
        for (int m = 0; m <= M; ++m) {
            if (idx >= m) {
                double2 in = inputData[idx - m];
                double absVal = sqrt(in.x * in.x + in.y * in.y);
                for (int k = 1; k <= K; ++k) {
                    double factor = pow(absVal, k - 1);
                    featureData[idx * max_terms + featureIdx] = (double2)(in.x * factor, in.y * factor);
                    featureIdx++;
                }
            }
        }

        // Construct ENVP MP terms
        for (int m = 1; m <= M_e; ++m) {
            if (idx >= m) {
                double2 current = inputData[idx];
                double2 in = inputData[idx - m];
                double absVal = sqrt(in.x * in.x + in.y * in.y);
                for (int k = 2; k <= K_e; ++k) {
                    double factor = pow(absVal, k - 1);
                    featureData[idx * max_terms + featureIdx] = (double2)(current.x * factor, current.y * factor);
                    featureIdx++;
                }
            }
        }
    }
}
)CLC";

class HybridMemoryPolynomialModel {
private:
    arma::cx_mat data;
    arma::cx_colvec coefficients;
    arma::cx_mat phi_HMP;
    int K, M, K_e, M_e;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    void initializeOpenCL() {
        device = selectGPUDevice();

        // Получение и вывод имени устройства
        size_t deviceNameSize;
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize);
        std::vector<char> deviceName(deviceNameSize);
        clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr);
        std::cout << "Selected GPU Device: " << std::string(deviceName.begin(), deviceName.end()) << std::endl;

        cl_int err;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context.");

        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL command queue.");

        // Создание программы и ядра
        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program.");

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char buffer[2048];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, nullptr);
            std::cerr << "OpenCL Program Build Log: " << buffer << std::endl;
            throw std::runtime_error("Failed to build program.");
        }

        kernel = clCreateKernel(program, "computeFeatures", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create kernel.");
    }

    void buildAndRunKernel() {
        int n_samples = data.n_rows - std::max(M, M_e);
        int max_terms = (M + 1) * K + M_e * (K_e - 1);
        phi_HMP.set_size(n_samples, max_terms);
        phi_HMP.zeros();

        // Проверка размеров
        std::cout << "Number of samples: " << n_samples << ", Max terms: " << max_terms << std::endl;
        if (n_samples == 0 || max_terms == 0) {
            throw std::runtime_error("Invalid matrix dimensions, can't proceed.");
        }

        std::vector<cl_double2> inputVec(data.n_rows);
        std::vector<cl_double2> featureVec(n_samples * max_terms);
        for (size_t i = 0; i < data.n_rows; ++i) {
            inputVec[i].s[0] = std::real(data(i, 0));
            inputVec[i].s[1] = std::imag(data(i, 0));
        }

        cl_int err;
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double2) * inputVec.size(), inputVec.data(), &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create input buffer.");

        cl_mem featureBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double2) * featureVec.size(), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create feature buffer.");

        // Установка аргументов ядра
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &featureBuffer);
        err |= clSetKernelArg(kernel, 2, sizeof(int), &K);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &M);
        err |= clSetKernelArg(kernel, 4, sizeof(int), &K_e);
        err |= clSetKernelArg(kernel, 5, sizeof(int), &M_e);
        err |= clSetKernelArg(kernel, 6, sizeof(int), &n_samples);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel arguments.");

        size_t globalWorkSize = n_samples;

        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue kernel.");

        err = clEnqueueReadBuffer(queue, featureBuffer, CL_TRUE, 0, sizeof(cl_double2) * featureVec.size(), featureVec.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to read buffer.");

        // Заполнение phi_HMP
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < max_terms; ++j) {
                phi_HMP(i, j) = std::complex<double>(featureVec[i * max_terms + j].s[0], featureVec[i * max_terms + j].s[1]);
            }
        }

        std::cout << "phi_HMP matrix dimensions: " << phi_HMP.n_rows << "x" << phi_HMP.n_cols << std::endl;

        coefficients = arma::solve(phi_HMP, data.col(1).subvec(std::max(M, M_e), data.n_rows - 1));

        // Очистка
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(featureBuffer);
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
        if (coefficients.empty()) {
            throw std::runtime_error("Model is not trained. Call fit() before predict().");
        }
        return (phi_HMP * coefficients);
    }

    std::pair<double, double> calculateRMSE(const arma::cx_vec& predictions, const arma::cx_vec& y_true) const {
        if (predictions.n_elem != y_true.n_elem) {
            throw std::runtime_error("Error: Prediction size does not match the true data size.");
        }

        double rmse_real = std::sqrt(arma::mean(arma::square(arma::real(y_true) - arma::real(predictions))));
        double rmse_imag = std::sqrt(arma::mean(arma::square(arma::imag(y_true) - arma::imag(predictions))));

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
        model.fit();

        arma::cx_vec y_pred = model.predict();
        arma::cx_vec y_true = model.get_output_data();

        auto [rmse_real, rmse_imag] = model.calculateRMSE(y_pred, y_true);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
