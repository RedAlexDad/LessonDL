#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// OpenCL ядро, умножающее каждый элемент массива на 2
const char *kernelSource = R"(
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    __kernel void mult(__global double *v) {
        int id;
        id = get_global_id(0);
        v[id] = 2*v[id];
    }
)";


int main(int argc, char *argv[]) {
    // Объявления, связанные с задачей
    unsigned int N = 128;
    size_t N_bytes = N * sizeof(double);

    // Объявления OpenCL
    cl_platform_id platform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_mult;

    // Хост-версия массива v
    double *h_v;  // вещественные и мнимые части
    h_v = (double *) malloc(N_bytes);

    // Инициализация массива v на хосте
    int i;
    for (i = 0; i < N; i++) {
        h_v[i] = i;
    }

    // Глобальное и локальное количество потоков
    size_t globalSize, localSize;
    globalSize = N;
    localSize = 32;

    // Показать поддерживаемые расширения
    /*
    cl_char extensions[2048] = {0};
    clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(extensions), &extensions, NULL);
    printf("%s\n", extensions);
    */

    // Настройка OpenCL
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);

    // Построение исполняемой программы
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Ошибка компиляции программы\n");
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char *) malloc(log_size);
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("%s\n", log);
        }
    }
    k_mult = clCreateKernel(program, "mult", &err);

    // Создание массивов на хосте и их запись
    cl_mem d_v;
    d_v = clCreateBuffer(context, CL_MEM_READ_WRITE, N_bytes, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_v, CL_TRUE, 0, N_bytes, h_v, 0, NULL, NULL);

    err = clSetKernelArg(k_mult, 0, sizeof(cl_mem), &d_v);

    err = clEnqueueNDRangeKernel(queue, k_mult, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(queue);

    // Передача обратно
    clEnqueueReadBuffer(queue, d_v, CL_TRUE, 0, N_bytes, h_v, 0, NULL, NULL);
    clFinish(queue);

    // Проверка
    int correct = 1;
    for (i = 0; i < N; i++) {
        if (h_v[i] != (double) 2 * i)
            correct = 0;
    }
    if (correct)
        printf("Массив верен!\n");
    else
        printf("Массив неверен :(\n");

    // Освобождение ресурсов OpenCL
    clReleaseMemObject(d_v);
    clReleaseProgram(program);
    clReleaseKernel(k_mult);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Освобождение памяти на хосте
    free(h_v);

    return 0;
}
