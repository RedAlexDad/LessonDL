cmake_minimum_required(VERSION 3.26)
project(LessonDL)

# Установите переменную окружения для игнорирования предупреждения
set(ENV{LD_PRELOAD} "/usr/local/cuda-12.3/lib64/libnvrtc.so")

# Путь к CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.3")

# Указание архитектуры для компилятора CUDA
set(CMAKE_CUDA_ARCHITECTURES 75)

# Путь к libtorch
set(Torch_DIR "/usr/local/include/libtorch/share/cmake/Torch")
# Найдите компоненты libtorch
find_package(Torch REQUIRED PATHS ${Torch_DIR} NO_DEFAULT_PATH)

# Добавьте директории для заголовочных файлов и библиотек libtorch
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARIES})

# Установите переменную окружения для игнорирования предупреждения
set(ENV{LD_PRELOAD} "libnvrtc.so.11.0")

# Создание исполняемого файла
add_executable(
        LessonDL
        #        driver.cpp
        hello.cpp
)

# Указание компилятора CUDA и его опций для цели LessonDL
target_compile_options(LessonDL PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75 -std=c++14>)

# Подключение к библиотекам SYCL
target_link_libraries(
        LessonDL PRIVATE
        "${TORCH_LIBRARIES}"
)

# Добавление директорий для заголовочных файлов SYCL
target_include_directories(
        LessonDL PRIVATE
        ${TORCH_INCLUDE_DIRS}
)
