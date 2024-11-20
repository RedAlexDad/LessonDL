cmake_minimum_required(VERSION 3.26)
project(LessonDL LANGUAGES CXX)

# C++ стандарт
set(CPP_VERSION 20)
set(CMAKE_CXX_STANDARD ${CPP_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_EXTENSIONS OFF)

# Установка компиляторов
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)
# Включение подробного вывода
set(CMAKE_VERBOSE_MAKEFILE ON)

# Добавим флаги для указания конкретного устройства
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
# string(APPEND CMAKE_CXX_FLAGS " -fsycl-targets=nvptx64-nvidia-cuda")
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -O2")

# # Если необходимо, укажите отдельные переменные
set(SYCL_INCLUDE_DIR "/opt/intel/oneapi/compiler/2025.0/include")
set(SYCL_LIBRARY_DIR "/opt/intel/oneapi/compiler/2025.0/lib")

# Добавление пути к заголовкам и библиотекам SYCL
include_directories(${SYCL_INCLUDE_DIR})
link_directories(${SYCL_LIBRARY_DIR})

# IntelSYCL
find_package(IntelSYCL REQUIRED)

# Функция для добавления исполняемого файла из папки
function(add_executable_from_folder folder_name)
    file(GLOB_RECURSE src_files "${CMAKE_CURRENT_SOURCE_DIR}/${folder_name}/*.cpp")
    foreach(src_file ${src_files})
        get_filename_component(executable_name ${src_file} NAME_WE)
        string(REPLACE "/" "_" target_name ${folder_name})
        add_executable(${target_name}_${executable_name} ${src_file})
    endforeach()
endfunction()

# Добавление исполняемых файлов из каждой папки
add_executable_from_folder(check_device)
add_executable_from_folder(exercise_000)
add_executable_from_folder(exercise_001)
add_executable_from_folder(simple-sycl-app)
add_executable_from_folder(machine_learning_001)
add_executable_from_folder(machine_learning_002)
add_executable_from_folder(machine_learning_003)
add_executable_from_folder(machine_learning_004)
