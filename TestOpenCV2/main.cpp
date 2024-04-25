#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Ошибка: Невозможно открыть камеру." << std::endl;
        return -1;
    }

    // Устанавливаем желаемую частоту кадров в 30 FPS
    cap.set(CAP_PROP_FPS, 30);

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Ошибка: Невозможно получить кадр с камеры." << std::endl;
            break;
        }

        imshow("Камера", frame);
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
