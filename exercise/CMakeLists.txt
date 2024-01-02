cmake_minimum_required(VERSION 3.26)
project(LessonDL)

set(CMAKE_CXX_STANDARD 20)

# Поиск пакета OpenCL
find_package(OpenCL REQUIRED)

# Добавление исполняемого файла
add_executable(
        LessonDL
        testspeed.cpp
        #        testspeed1.cpp
        #        example_000.cpp
        #        example_001.cpp
        #        example_001ru.cpp
        #        example_002.cpp
        #        example_002ru.cpp
        #        example_003.cpp
        #        example_003ru.cpp
)

# Добавление директорий для заголовочных файлов и библиотек OpenCL
target_include_directories(LessonDL PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(LessonDL PRIVATE ${OpenCL_LIBRARIES})
