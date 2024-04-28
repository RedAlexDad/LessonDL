cmake_minimum_required(VERSION 3.26)
project(Hello_tf)

set(CMAKE_CXX_STANDARD 17)

# Путь к заголовочным файлам TensorFlow
#include_directories(/usr/local/include)

add_executable(
        Hello_tf
        hello_tf.cpp
)

# Подключение к библиотекам Tensorflow
target_link_libraries(
        Hello_tf PRIVATE
        /usr/local/lib/libtensorflow.so
)

