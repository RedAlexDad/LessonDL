cmake_minimum_required(VERSION 3.26)
project(LessonDL)

# Установите путь к папке, содержащей установленный Torch
set(Torch_DIR "/usr/local/lib/libtorch_cuda/share/cmake/Torch")

# Найдем необходимые пакеты
find_package(Torch REQUIRED)

# Функция для добавления исполняемых файлов из папки
function(add_executable_from_folder folder_name)
    file(GLOB_RECURSE src_files "${CMAKE_CURRENT_SOURCE_DIR}/${folder_name}/*.cpp")
    foreach (src_file ${src_files})
        get_filename_component(executable_name ${src_file} NAME_WE)
        string(REPLACE "." "_" target_name ${folder_name})
        add_executable(${target_name}_${executable_name} ${src_file})
        target_include_directories(${target_name}_${executable_name} PRIVATE ${TORCH_INCLUDE_DIRS})
        target_link_libraries(${target_name}_${executable_name} PRIVATE ${TORCH_LIBRARIES})
        set_property(TARGET ${target_name}_${executable_name} PROPERTY CXX_STANDARD 17)
    endforeach ()
endfunction()

# Добавляем поддиректории
add_executable_from_folder(exercise_000)
add_executable_from_folder(exercise_001)
add_executable_from_folder(exercise_002)
add_executable_from_folder(check_torch_cpu)
add_executable_from_folder(check_torch_cuda)

find_package(CUDA REQUIRED)
set(CUDA_TOOLKIT_ROOT_DIR /usr/lib/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)