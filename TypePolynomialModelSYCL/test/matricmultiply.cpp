#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>
#include <cmath> // Include for std::fabs

/**
 * Matrix size constants.
 */
constexpr int m_size = 150 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

/**
 * Verify the results from device by performing host computation.
 */
int VerifyResult(float (*c_back)[P]);

int main() {
  float(*c_back)[P] = new float[M][P];

  // Initialize c_back
  for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++) c_back[i][j] = 0.0f;

  try {
    sycl::queue q(sycl::default_selector{});

    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    sycl::buffer<float, 2> a_buf(sycl::range(M, N));
    sycl::buffer<float, 2> b_buf(sycl::range(N, P));
    sycl::buffer<float, 2> c_buf(reinterpret_cast<float *>(c_back), sycl::range(M, P));

    std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
              << ") * b(" << N << "," << P << ")\n";

    // Initialize matrix a
    q.submit([&](sycl::handler &h) {
      sycl::accessor a(a_buf, h, sycl::write_only);
      h.parallel_for(sycl::range(M, N), [=](sycl::item<2> index) {
        a[index] = 1.0f;
      });
    });

    // Initialize matrix b
    q.submit([&](sycl::handler &h) {
      sycl::accessor b(b_buf, h, sycl::write_only);
      h.parallel_for(sycl::range(N, P), [=](sycl::item<2> index) {
        b[index] = index[0] + 1.0f;
      });
    });

    // Matrix multiplication: c = a * b
    q.submit([&](sycl::handler &h) {
      sycl::accessor a(a_buf, h, sycl::read_only);
      sycl::accessor b(b_buf, h, sycl::read_only);
      sycl::accessor c(c_buf, h, sycl::write_only);

      int width_a = a_buf.get_range()[1];

      h.parallel_for(sycl::range(M, P), [=](sycl::item<2> index) {
        int row = index[0];
        int col = index[1];
        float sum = 0.0f;

        for (int i = 0; i < width_a; i++) {
          sum += a[row][i] * b[i][col];
        }

        c[index] = sum;
      });
    });
    
  } catch (sycl::exception const &e) {
    std::cout << "An exception is caught while multiplying matrices.\n";
    std::terminate();
  }

  int result;
  std::cout << "Result of matrix multiplication using SYCL: ";
  result = VerifyResult(c_back);
  delete[] c_back;

  return result;
}

bool ValueSame(float a, float b) {
  return std::fabs(a - b) < std::numeric_limits<float>::epsilon();
}

int VerifyResult(float (*c_back)[P]) {
  int i, j, k;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[P] = new float[N][P];
  float(*c_host)[P] = new float[M][P];

  // Each element of matrix a is 1.
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) a_host[i][j] = 1.0f;

  // Each column of b_host is the sequence 1,2,...,N
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++) b_host[i][j] = i + 1.0f;

  // c_host is initialized to zero.
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = 0.0f;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      // Each element of the product is just the sum 1+2+...+n
      for (j = 0; j < P; j++) {
        c_host[i][j] += a_host[i][k] * b_host[k][j];
      }
    }
  }

  bool mismatch_found = false;

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int print_count = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        std::cout << "Fail - The result is incorrect for element: [" << i << ", "
                  << j << "], expected: " << c_host[i][j]
                  << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }

    if (print_count == 5) break;
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;

  if (!mismatch_found) {
    std::cout << "Success - The results are correct!\n";
    return 0;
  } else {
    std::cout << "Fail - The results mismatch!\n";
    return -1;
  }
}
