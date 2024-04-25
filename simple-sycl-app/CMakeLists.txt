cmake_minimum_required(VERSION 3.10)
project(LessonDL)

# Укажите путь к заголовочным файлам SYCL от OneAPI
set(SYCL_INCLUDE_DIR "/opt/intel/oneapi/compiler/latest/include/sycl/CL")

# Укажите путь к библиотекам компилятора OneAPI (если это необходимо)
set(SYCL_LIBRARY_DIR "/opt/intel/oneapi/compiler/latest/lib")

# Добавляем путь к заголовочным файлам SYCL
message(STATUS "SYCL_INCLUDE_DIR: ${SYCL_INCLUDE_DIR}")
include_directories(${SYCL_INCLUDE_DIR})

# Добавляем путь к библиотекам компилятора OneAPI (если это необходимо)
link_directories(${SYCL_LIBRARY_DIR})

# Добавляем исполняемый файл вашего проекта
add_executable(LessonDL simple-sycl-app.cpp)

# Линковка с библиотекой SYCL от OneAPI (если это необходимо)
target_link_libraries(LessonDL sycl)
