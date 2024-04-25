cmake_minimum_required(VERSION 3.26)
project(LessonDL)

# Поиск пакета OpenCL
find_package(OpenCL REQUIRED)

# Функция для добавления исполняемого файла из папки
function(add_executable_from_folder folder_name)
    file(GLOB_RECURSE src_files "${CMAKE_CURRENT_SOURCE_DIR}/${folder_name}/*.cpp")
    foreach(src_file ${src_files})
        get_filename_component(executable_name ${src_file} NAME_WE)
        string(REPLACE "/" "_" target_name ${folder_name})
        add_executable(${target_name}_${executable_name} ${src_file})
        target_include_directories(${target_name}_${executable_name} PRIVATE ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(${target_name}_${executable_name} PRIVATE ${OpenCL_LIBRARIES})
    endforeach()
endfunction()

# Добавление исполняемых файлов из каждой папки
add_executable_from_folder(simple-sycl-app)