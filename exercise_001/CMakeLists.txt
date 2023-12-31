cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)

# Путь к заголовочным файлам TensorFlow
include_directories(/usr/local/include)

# Поиск пакета OpenCL
find_package(OpenCL REQUIRED)

# Добавляем исполняемый файл, если это необходимо
add_executable(
        LessonDL
        main.cpp
)
# Добавление директорий для заголовочных файлов и библиотек OpenCL
target_include_directories(
        LessonDL PRIVATE
        ${OpenCL_INCLUDE_DIRS}
)

# Подключение к библиотекам TensorFlow
target_link_libraries(
        LessonDL PRIVATE
        /usr/local/lib/libtensorflow.so
        ${OpenCL_LIBRARIES}
)


# Создаем shared library
add_library(
        LessonDLLib SHARED
        main.cpp
)

# Устанавливаем флаги компиляции для создания позиционно-независимого кода
set_property(TARGET LessonDLLib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Указываем имя выходного файла .so
# Если нет файла, то компилируем вручную
# cmake --build cmake-build-debug
set_target_properties(LessonDLLib PROPERTIES OUTPUT_NAME "LessonDL")

message(STATUS "CMake System Name: ${CMAKE_SYSTEM_NAME}")
