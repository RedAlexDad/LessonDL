#include <sycl/sycl.hpp>
#include <iostream>

int main() {
  // Размер векторов
  const int N = 1000;

  // Входные векторы
  std::vector<float> a(N), b(N);
  // Выходной вектор
  std::vector<float> c(N);

  // Инициализация векторов
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = 2 * i;
  }

  // Создание очереди SYCL
  sycl::queue queue;

  // Создание буферов для векторов
  sycl::buffer<float, 1> a_buffer(a.data(), sycl::range<1>(N));
  sycl::buffer<float, 1> b_buffer(b.data(), sycl::range<1>(N));
  sycl::buffer<float, 1> c_buffer(c.data(), sycl::range<1>(N));

  // Отправка задания в очередь
  queue.submit([&](sycl::handler &cgh) {
    // Получение доступа к буферам
    auto a_accessor = a_buffer.get_access<sycl::access::mode::read>(cgh);
    auto b_accessor = b_buffer.get_access<sycl::access::mode::read>(cgh);
    auto c_accessor = c_buffer.get_access<sycl::access::mode::write>(cgh);

    // Выполнение параллельного сложения
    cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
      c_accessor[id] = a_accessor[id] + b_accessor[id];
    });
  });

  // Ожидание завершения задания
  queue.wait();

  // Вывод результата
  for (int i = 0; i < N; ++i) {
    std::cout << c[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}