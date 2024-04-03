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
add_executable_from_folder(check_driver_000)
add_executable_from_folder(example_000)
add_executable_from_folder(example_001)
add_executable_from_folder(example_001ru)
add_executable_from_folder(example_002)
add_executable_from_folder(example_002ru)
add_executable_from_folder(example_003)
add_executable_from_folder(example_003ru)
add_executable_from_folder(hello_world)
add_executable_from_folder(search_device)
add_executable_from_folder(test_speed_000)
add_executable_from_folder(test_speed_001)