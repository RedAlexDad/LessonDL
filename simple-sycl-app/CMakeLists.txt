cmake_minimum_required(VERSION 3.5)
project(opensycl-examples LANGUAGES CXX)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Устанавливаем переменную CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "/home/redalexdad/AdaptiveCpp")
# Находим пакет AdaptiveCpp
find_package(AdaptiveCpp CONFIG REQUIRED)

add_executable(opensycl-examples simple-sycl-app.cpp)
add_sycl_to_target(TARGET opensycl-examples SOURCES simple-sycl-app.cpp)
install(TARGETS opensycl-examples RUNTIME DESTINATION share/hipSYCL/examples/)

# Устанавливаем тип сборки, если не указан
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Устанавливаем уровень отладки, если не указан
if(NOT ACPP_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ACPP_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(ACPP_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

# Устанавливаем уровень отладки для AdaptiveCpp
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})