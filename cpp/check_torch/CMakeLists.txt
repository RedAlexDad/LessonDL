cmake_minimum_required(VERSION 3.26)
project(check_torch)

# Получаем список всех файлов в текущем каталоге исходного кода
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Создание исполняемого файла
add_executable(${PROJECT_NAME} ${SOURCES})

# Путь к Libtorch
set(Torch_DIR "/usr/local/include/libtorch/share/cmake/Torch")

# Найдем необходимые пакеты
find_package(Torch REQUIRED)

# Добавление пути к заголовочным файлам PyTorch
target_include_directories(check_torch PRIVATE ${TORCH_INCLUDE_DIRS})

# Линковка с библиотекой PyTorch
target_link_libraries(check_torch PRIVATE ${TORCH_LIBRARIES})

set_property(TARGET check_torch PROPERTY CXX_STANDARD 20)