cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)

# Путь к libtorch
set(Torch_DIR "/usr/local/include/libtorch/share/cmake/Torch")
# Найдите компоненты libtorch
find_package(Torch REQUIRED PATHS ${Torch_DIR})
# Добавьте эту строку для отключения опции --arch=sm_30
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35)

add_executable(
        LessonDL
        #        driver.cpp
        hello.cpp
)

# Подключение к библиотекам SYCL
target_link_libraries(
        LessonDL PRIVATE
        ${TORCH_LIBRARIES}
)

# Добавление директорий для заголовочных файлов SYCL
target_include_directories(
        LessonDL PRIVATE
        ${TORCH_INCLUDE_DIRS}
)