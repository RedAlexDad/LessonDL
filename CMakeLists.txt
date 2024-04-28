cmake_minimum_required(VERSION 3.26)
project(LessonDL)

function(add_executable_from_folder folder_name)
    file(GLOB_RECURSE src_files "${CMAKE_CURRENT_SOURCE_DIR}/${folder_name}/*.cpp")
    foreach (src_file ${src_files})
        get_filename_component(executable_name ${src_file} NAME_WE)
        add_executable(${executable_name} ${src_file})
        target_include_directories(${executable_name} PRIVATE)
        target_link_libraries(${executable_name} PRIVATE tensorflow) # Link with TensorFlow library
    endforeach ()
endfunction()


# Добавление исполняемых файлов из каждой папки
add_executable_from_folder(hello_tf)
add_executable_from_folder(driver)
add_executable_from_folder(exercise_01)
#add_executable_from_folder(exercise_02)
add_executable_from_folder(exercise_03)
#add_executable_from_folder(tensorflow-cc-examples)
#add_executable_from_folder(test_speed_0)
#add_executable_from_folder(test_speed_1)
#add_executable_from_folder(test_speed_2)

find_package(CUDA REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR /usr/lib/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CUDA_ARCH_NAME "compute_75") # Архитектура GPU NVIDIA GeForce GTX 1650 Ti Laptop