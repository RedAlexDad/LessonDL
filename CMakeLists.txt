cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
# Устанавливаем ComputeCpp_DIR
set(ComputeCpp_DIR "/home/redalexdad/Programs/ComputeCpp-CE-2.11.0-x86_64-linux-gnu")

# Поиск пакета ComputeCpp
find_package(ComputeCpp REQUIRED CONFIG PATHS ${ComputeCpp_DIR})

add_executable(
        LessonDL
        accessors.cpp
)

# Подключение к библиотекам SYCL
target_link_libraries(
        LessonDL PRIVATE
        ${ComputeCpp_LIBRARY}
)

# Добавление директорий для заголовочных файлов SYCL
target_include_directories(
        LessonDL PRIVATE
        ${ComputeCpp_INCLUDE_DIRS}
)

target_compile_options(LessonDL PRIVATE -std=c++17)