#include <cstdio>
#include <tensorflow/c/c_api.h>


// gcc hello_tf.cpp -ltensorflow -o hello_tf
// ./hello_tf

int main() {
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    return 0;
}