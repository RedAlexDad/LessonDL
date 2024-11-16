cmake_minimum_required(VERSION 3.12)

# Название проекта
project(TypePolynomialModelOpenCL)

# Устанавливаем стандарт C++
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Найдем библиотеку Armadillo
find_package(Armadillo REQUIRED)

# Найдем библиотеку OpenCL
find_package(OpenCL REQUIRED)

# Добавление исполняемого файла для check_drive (проверка драйвера)
add_executable(CheckOpenCL check_drive.cpp)
# Подключение библиотеки OpenCL
target_link_libraries(CheckOpenCL OpenCL::OpenCL)

# Добавление исполняемого файла для HybridMemoryPolynomialModel
add_executable(HybridMemoryPolynomialModel HybridMemoryPolynomialModel.cpp)
# Ссылка на библиотеки Armadillo и OpenCL
target_link_libraries(HybridMemoryPolynomialModel
    armadillo
    OpenCL::OpenCL
)
