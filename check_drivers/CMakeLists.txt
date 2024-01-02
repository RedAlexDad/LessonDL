cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)

# Поиск пакета OpenCL
find_package(OpenCL REQUIRED)

# Добавление исполняемого файла
add_executable(
        LessonDL
#                check.cpp
#        search_device.cpp
                hello_world.cpp
)

# Добавление директорий для заголовочных файлов и библиотек OpenCL
target_include_directories(LessonDL PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(LessonDL PRIVATE ${OpenCL_LIBRARIES})

# Запустите сборку с помощью ninja:
# ninja
# После сборки
# ./LessonDL