cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)

# Путь к заголовочным файлам TensorFlow
include_directories(/usr/local/include)

add_executable(
        LessonDL
#        driver.cpp
#        test_speed0.cpp
        exercise/exercise_01.cpp
#        hello_tf.cpp
)

# Подключение к библиотекам SYCL
target_link_libraries(
        LessonDL PRIVATE
        /usr/local/lib/libtensorflow.so
)

# Добавление директорий для заголовочных файлов SYCL
target_include_directories(
        LessonDL PRIVATE
        ${OpenCL_INCLUDE_DIRS}
)

