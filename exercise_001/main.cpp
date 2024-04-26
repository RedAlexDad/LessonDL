#include <vector>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace std;

// Функция для сложения векторов на CPU
void add_vectors_cpu(const vector<int>& a, const vector<int>& b, vector<int>& result) {
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
}

// Функция для сложения векторов с использованием SYCL
void add_vectors_sycl(const vector<int>& a, const vector<int>& b, vector<int>& result) {
    size_t size = a.size();

    sycl::queue queue(hipsycl::sycl::gpu_selector{});

    sycl::buffer<int, 1> a_buf(a.data(), sycl::range<1>(size));
    sycl::buffer<int, 1> b_buf(b.data(), sycl::range<1>(size));
    sycl::buffer<int, 1> result_buf(result.data(), sycl::range<1>(size));

    queue.submit([&](sycl::handler& cgh) {
        auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
        auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
        auto result_acc = result_buf.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class add_kernel>(sycl::range<1>(size), [=](sycl::id<1> id) {
            result_acc[id] = a_acc[id] + b_acc[id];
        });
    });

    queue.wait();
}

int main() {
    const int N = 1000;
    vector<int> a(N), b(N), result_cpu(N), result_sycl(N);

    // Инициализация векторов
    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Замер времени CPU
    auto start_cpu = chrono::high_resolution_clock::now();
    add_vectors_cpu(a, b, result_cpu);
    auto end_cpu = chrono::high_resolution_clock::now();
    auto duration_cpu = chrono::duration_cast<chrono::microseconds>(end_cpu - start_cpu);

    // Замер времени SYCL
    auto start_sycl = chrono::high_resolution_clock::now();
    add_vectors_sycl(a, b, result_sycl);
    auto end_sycl = chrono::high_resolution_clock::now();
    auto duration_sycl = chrono::duration_cast<chrono::microseconds>(end_sycl - start_sycl);

    // Проверка результатов
    for (int i = 0; i < N; ++i) {
        if (result_cpu[i] != result_sycl[i]) {
            cout << "Ошибка: результаты не совпадают!" << endl;
            return 1;
        }
    }

    // Вывод результатов
    cout << "Время выполнения (CPU): " << duration_cpu.count() << " микросекунд" << endl;
    cout << "Время выполнения (SYCL): " << duration_sycl.count() << " микросекунд" << endl;

    double speedup = static_cast<double>(duration_cpu.count()) / duration_sycl.count();
    cout << "SYCL быстрее в " << speedup << " раз" << endl;

    return 0;
}
