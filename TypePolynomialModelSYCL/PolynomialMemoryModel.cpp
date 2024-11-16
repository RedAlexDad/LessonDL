#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <armadillo>

class PolynomialMemoryModel {
private:
    arma::cx_mat data;             // Данные
    arma::cx_colvec coefficients;  // Коэффициенты
    int K;                         // Порядок нелинейности
    int M;                         // Глубина памяти

public:
    PolynomialMemoryModel(const std::string& filename, int K, int M)
        : K(K), M(M), coefficients(arma::cx_colvec()) {
        int rows;
        data = load_data(filename, rows);
        if (rows <= M) {
            throw std::runtime_error("Error: Not enough data samples or incorrect data format.");
        }
    }

    arma::cx_mat load_data(const std::string& filename, int& rows) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + filename);
        }

        std::string line;
        std::vector<std::complex<double>> inputs, outputs; // Изменение типа на double

        while (std::getline(file, line)) {
            if (line.find("Time") != std::string::npos) continue;

            std::stringstream ss(line);
            std::string token;
            double real_in, imag_in; // Использование double
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
        arma::cx_mat mat_data(inputs.size(), 2); // Создание матрицы
        for (size_t i = 0; i < inputs.size(); ++i) {
            mat_data(i, 0) = inputs[i];
            mat_data(i, 1) = outputs[i];
        }
        return mat_data; // Возврат матрицы
    }

    void fit() {
        int n_samples = data.n_rows - M;
        arma::cx_mat phi_MP(n_samples, (M + 1) * K);
        arma::cx_colvec y_trimmed(n_samples);

        // Старт вычислений в SYCL
        {
            cl::sycl::queue queue;

            // Передача данных из Armadillo в SYCL
            cl::sycl::buffer<std::complex<double>, 1> phi_buffer(phi_MP.memptr(), cl::sycl::range<1>(phi_MP.n_elem));
            cl::sycl::buffer<std::complex<double>, 1> y_trimmed_buffer(y_trimmed.memptr(), cl::sycl::range<1>(y_trimmed.n_elem));
            cl::sycl::buffer<std::complex<double>, 1> input_data_buffer(data.memptr(), cl::sycl::range<1>(data.n_elem));

            queue.submit([&](cl::sycl::handler& cgh) {
                auto phi_access = phi_buffer.get_access<cl::sycl::access::mode::write>(cgh);
                auto input_access = input_data_buffer.get_access<cl::sycl::access::mode::read>(cgh);

                cgh.parallel_for(cl::sycl::range<2>(n_samples, (M + 1) * K), [=](cl::sycl::item<2> item) {
                    size_t n = item.get_id(0) + M; // Индекс в исходных данных
                    size_t k = item.get_id(1) % K; 
                    size_t m = item.get_id(1) / K; 

                    if (n < data.n_rows && (n - m) >= 0) {
                        std::complex<double> input_value = input_access[n - m]; // Изменение типа на double

                        // Проверка безопасности перед возведением в степень
                        if (std::abs(input_value) > 1e-10) {
                            auto value = input_value * std::pow(std::abs(input_value), k);
                            phi_access[item.get_id(0) * ((M + 1) * K) + (m * K + k)] = value;
                        } else {
                            // Обработка случая, когда входное значение слишком малое
                            phi_access[item.get_id(0) * ((M + 1) * K) + (m * K + k)] = std::complex<double>(0, 0);
                        }
                    }
                });
            }).wait();
        }

        // Обработка результата из буфера
        coefficients = arma::pinv(phi_MP) * y_trimmed;
    }

    arma::cx_colvec predict() {
        int n_samples = data.n_rows - M;
        arma::cx_colvec y_pred(n_samples);

        for (int n = M; n < data.n_rows; ++n) {
            arma::cx_rowvec phi_n = arma::zeros<arma::cx_rowvec>((M + 1) * K);

            for (int m = 0; m <= M; ++m) {
                for (int k = 1; k <= K; ++k) {
                    phi_n(m * K + k - 1) = data(n - m, 0) * std::pow(std::abs(data(n - m, 0)), k - 1);
                }
            }

            y_pred(n - M) = arma::dot(phi_n, coefficients);
        }

        return y_pred;
    }

    std::pair<long double, long double> calculate_rmse(const arma::cx_colvec& y_true, const arma::cx_colvec& y_pred) {
        long double rmse_real = arma::norm(arma::real(y_true) - arma::real(y_pred)) / std::sqrt(y_true.size());
        long double rmse_imag = arma::norm(arma::imag(y_true) - arma::imag(y_pred)) / std::sqrt(y_true.size());

        return {rmse_real, rmse_imag};
    }

    arma::cx_colvec get_output_data() const {
        return data.col(1).tail(data.n_rows - M);
    }
};

int main(int argc, char* argv[]) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <filename> <K> <M>" << std::endl;
            return 1;
        }

        std::string filename = argv[1];
        int K = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);

        PolynomialMemoryModel model(filename, K, M);
        model.fit();

        arma::cx_colvec y_pred = model.predict();
        arma::cx_colvec y_true = model.get_output_data();

        auto [rmse_real, rmse_imag] = model.calculate_rmse(y_true, y_pred);

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "RMSE (Real part): " << rmse_real << ", RMSE (Imaginary part): " << rmse_imag << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
