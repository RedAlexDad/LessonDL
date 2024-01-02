cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)

# Поиск пакета OpenCL
find_package(OpenCL REQUIRED)

# Добавление исполняемого файла
add_executable(
        LessonDL
        hello_world.cpp
#        testspeed.cpp
#        testspeed1.cpp
)

# Добавление директорий для заголовочных файлов и библиотек OpenCL
target_include_directories(LessonDL PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(LessonDL PRIVATE ${OpenCL_LIBRARIES})
