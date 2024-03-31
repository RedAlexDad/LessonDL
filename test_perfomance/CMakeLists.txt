cmake_minimum_required(VERSION 3.26)
project(test_perfomance)

set(CMAKE_CXX_STANDARD 20)

# Путь к заголовочным файлам TensorFlow
include_directories(/usr/local/include)

# Поиск пакета OpenCL
find_package(OpenCL REQUIRED)

# Добавляем исполняемый файл, если это необходимо
add_executable(
        test_perfomance
        main.cpp
)
# Добавление директорий для заголовочных файлов и библиотек OpenCL
target_include_directories(
        test_perfomance PRIVATE
        ${OpenCL_INCLUDE_DIRS}
)

# Подключение к библиотекам TensorFlow
target_link_libraries(
        test_perfomance PRIVATE
        /usr/local/lib/libtensorflow.so
        ${OpenCL_LIBRARIES}
)


# Создаем shared library
add_library(
        test_perfomanceLib SHARED
        main.cpp
)

# Устанавливаем флаги компиляции для создания позиционно-независимого кода
set_property(TARGET test_perfomanceLib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Указываем имя выходного файла .so
# Если нет файла, то компилируем вручную
# cmake --build cmake-build-debug
set_target_properties(test_perfomanceLib PROPERTIES OUTPUT_NAME "test_perfomance")

message(STATUS "CMake System Name: ${CMAKE_SYSTEM_NAME}")
