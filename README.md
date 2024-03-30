# LessonDPC

Ознакомление с DPC++, Tensorflow, OpenCL и SYCL для DL, ML

## Последовательность команды создания исполняемого файла и библиотеки .so

1. Создать папку
```bash
mkdir build
```
2. Перейти на папку
```bash
  cd build/
 ```
3. Создать исполняемый файл
```bash
 cmake ..
 ```
4. Вернуть назад на предыдущую папку
```bash
 cd ..
 ```
5. Создать библиотеку
```bash
 cmake --build build/
 ```
6. Запустить исполняемый файл
``` bash
./build/LessonDL
```