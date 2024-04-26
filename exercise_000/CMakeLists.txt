cmake_minimum_required(VERSION 3.5)
project(exercise_000 LANGUAGES CXX)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Устанавливаем переменную CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "/usr/local/lib/AdaptiveCpp")
# Находим пакет AdaptiveCpp
find_package(AdaptiveCpp CONFIG REQUIRED)

add_executable(exercise_000 main.cpp)
add_sycl_to_target(TARGET exercise_000 SOURCES main.cpp)
install(TARGETS exercise_000 RUNTIME DESTINATION share/hipSYCL/examples/)

# Устанавливаем уровень отладки для AdaptiveCpp
# add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})