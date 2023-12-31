cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER /usr/bin/g++-9)
set(ComputeCpp_DIR "/home/redalexdad/Programs/ComputeCpp-CE-2.11.0-x86_64-linux-gnu/include/")

# Поиск пакета ComputeCpp
find_package(ComputeCpp REQUIRED)

# Путь к заголовочным файлам TensorFlow
#include_directories(/usr/local/include)

add_executable(
        LessonDL
        simple-vector-add.cpp
        #        exercise_02.cpp
)

# Подключение к библиотекам TensorFlow, OpenCL и SYCL
target_link_libraries(LessonDL
#        /usr/local/lib/libtensorflow.so
        ${ComputeCpp_LIBRARY}
)

# Добавление директорий для заголовочных файлов OpenCL и SYCL
target_include_directories(LessonDL PRIVATE
#        ${OpenCL_INCLUDE_DIRS}
        ${ComputeCpp_INCLUDE_DIRS}
)
