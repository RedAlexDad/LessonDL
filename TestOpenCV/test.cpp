#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Загрузка изображения
    Mat image = imread("example.jpg");

    // Проверка успешности загрузки
    if (image.empty()) {
        std::cout << "Ошибка: Невозможно загрузить изображение." << std::endl;
        return -1;
    }

    // Отображение изображения
    imshow("Example image", image);
    waitKey(0);

    return 0;
}
