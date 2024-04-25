#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Ошибка: Невозможно открыть камеру." << std::endl;
        return -1;
    }

    cap.set(CAP_PROP_FPS, 30);

    // Загрузка модели
    std::string model_prototxt_path = "models/caffemodel/MobileNetSSD_deploy.prototxt";
    std::string model_caffemodel_path = "models/caffemodel/MobileNetSSD_deploy.caffemodel";
    cv::dnn::Net load_model = cv::dnn::readNetFromCaffe(model_prototxt_path, model_caffemodel_path);

    float minimal_confidence = 0.1;

    // std::vector<std::string> categories_list = {"фон", "самолет", "велосипед", "птица", "лодка", "бутылка", "автобус", "автомобиль", "кошка", "стул", "корова", "стол", "собака", "лошадь", "мотоцикл", "человек", "цветок", "овца", "диван", "поезд", "телевизор"};
    std::vector<std::string> categories_list = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    cv::RNG rng(600000); // для лучших цветов
    std::vector<cv::Scalar> colors;
    for (size_t i = 0; i < categories_list.size(); ++i) {
        colors.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Ошибка: Невозможно получить кадр с камеры." << std::endl;
            break;
        }

        int height = frame.rows;
        int width = frame.cols;

        cv::Mat blob;
        cv::resize(frame, blob, cv::Size(300, 300));
        blob = cv::dnn::blobFromImage(blob, 0.007, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false);

        load_model.setInput(blob);
        cv::Mat detected_objects = load_model.forward();
        
        for (int objectIndex = 0; objectIndex < detected_objects.size[2]; ++objectIndex) {
            float confidence = detected_objects.ptr<float>(0, 0, objectIndex)[2];

            if (confidence > minimal_confidence) {
                int category_index = static_cast<int>(detected_objects.ptr<float>(0, 0, objectIndex)[1]);

                int upper_left_x = static_cast<int>(detected_objects.ptr<float>(0, 0, objectIndex)[3] * width);
                int upper_left_y = static_cast<int>(detected_objects.ptr<float>(0, 0, objectIndex)[4] * height);
                int lower_right_x = static_cast<int>(detected_objects.ptr<float>(0, 0, objectIndex)[5] * width);
                int lower_right_y = static_cast<int>(detected_objects.ptr<float>(0, 0, objectIndex)[6] * height);

                cv::rectangle(frame, cv::Point(upper_left_x, upper_left_y), cv::Point(lower_right_x, lower_right_y), colors[category_index], 2);

                std::string format_confidence = (confidence == 1.0f) ? "100" : std::to_string(confidence);
                std::string detected_obj_text = categories_list[category_index] + ": " + format_confidence + "%";
                cv::putText(frame, detected_obj_text, cv::Point(upper_left_x, upper_left_y - 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, colors[category_index], 2);
            }
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
