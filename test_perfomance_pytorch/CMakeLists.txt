cmake_minimum_required(VERSION 3.26)
project(test_perfomance_pytorch)

# Получаем список всех файлов в текущем каталоге исходного кода
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Создание исполняемого файла
add_executable(${PROJECT_NAME} ${SOURCES})

# Путь к Libtorch
set(Torch_DIR "/usr/local/include/libtorch/share/cmake/Torch")

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}/usr/local/include/libtorch")

# Найдем необходимые пакеты
find_package(Torch REQUIRED)

# Добавление пути к заголовочным файлам PyTorch
target_include_directories(test_perfomance_pytorch PRIVATE ${TORCH_INCLUDE_DIRS})

# Линковка с библиотекой PyTorch
target_link_libraries(test_perfomance_pytorch PRIVATE ${TORCH_LIBRARIES})

set_property(TARGET test_perfomance_pytorch PROPERTY CXX_STANDARD 20)

# Создаем shared library
add_library(
        test_perfomance_pytorchLib SHARED
        ${SOURCES}
)

# Устанавливаем флаги компиляции для создания позиционно-независимого кода
set_property(TARGET test_perfomance_pytorchLib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Указываем имя выходного файла .so
# Если нет файла, то компилируем вручную
# cmake --build cmake-build-debug
set_target_properties(test_perfomance_pytorchLib PROPERTIES OUTPUT_NAME "test_perfomance_pytorch")

message(STATUS "CMake System Name: ${CMAKE_SYSTEM_NAME}")
