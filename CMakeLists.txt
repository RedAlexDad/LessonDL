cmake_minimum_required(VERSION 3.26)
project(LessonDL)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Поиск пакета OpenCL
find_package(OpenCL REQUIRED)

# Устанавливаем переменную CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "/usr/local/lib/AdaptiveCpp")
# Находим пакет AdaptiveCpp
find_package(AdaptiveCpp CONFIG REQUIRED)

# Функция для добавления исполняемого файла из папки
function(add_executable_from_folder folder_name)
    file(GLOB_RECURSE src_files "${CMAKE_CURRENT_SOURCE_DIR}/${folder_name}/*.cpp")
    foreach(src_file ${src_files})
        get_filename_component(executable_name ${src_file} NAME_WE)
        string(REPLACE "/" "_" target_name ${folder_name})
        add_executable(${target_name}_${executable_name} ${src_file})
        target_include_directories(${target_name}_${executable_name} PRIVATE ${OpenCL_INCLUDE_DIRS} ${AdaptiveCpp_INCLUDE_DIRS})
        target_link_libraries(${target_name}_${executable_name} PRIVATE ${OpenCL_LIBRARIES} ${AdaptiveCpp_LIBRARIES})
        add_sycl_to_target(TARGET ${target_name}_${executable_name} SOURCES ${src_file})
        install(TARGETS ${target_name}_${executable_name} RUNTIME DESTINATION share/hipSYCL/examples/)
    endforeach()
endfunction()

# Добавление исполняемых файлов из каждой папки
add_executable_from_folder(exercise_000)
add_executable_from_folder(exercise_001)
add_executable_from_folder(simple-sycl-app)