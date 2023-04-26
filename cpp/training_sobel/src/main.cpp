#include <numeric>
#include <iostream>
#include <sstream>
#include <filesystem>
namespace fs = std::filesystem;

#include "deep_learning_tutorial.hpp"

#include "image_io_tutorial.hpp"

int main() {

    const int padding = 1;

    auto dataset = load_dataset("../images/", padding, false);

    const auto test_instance = dataset.back();
    dataset.pop_back();

    Matrix kernel = 0.05 * Matrix::Random(3, 3);

    auto check_instance_callback = [&test_instance, &kernel](int epoch, double loss) 
    {
        auto T = Convolution2D(test_instance.first, kernel, padding);
        cv::Mat temp, matY, matT;
        cv::eigen2cv(T, temp);
        temp *= 255.;
        temp.convertTo(matT, CV_8UC1);
        cv::eigen2cv(test_instance.second, temp);
        temp *= 255.;
        temp.convertTo(matY, CV_8UC1);

        temp = Mat::zeros(cv::Size(2*matY.cols, matY.rows), CV_8UC1);

        matY.copyTo(temp(cv::Rect(0, 0, matY.cols, matY.rows)));
        matT.copyTo(temp(cv::Rect(matY.cols, 0, matT.cols, matT.rows)));

        Mat output;
        cv::cvtColor(temp, output, cv::COLOR_GRAY2BGR);
        std::stringstream stream;
        stream << "Epoch #" << epoch << ", loss = ";
        stream << std::fixed << std::setprecision(6) << loss;
        std::string msg = stream.str();
        cv::putText(output, msg, cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Ground-truth x current state", output);
        cv::waitKey(10);
    };

    const int MAX_EPOCHS = 1000;
    const double learning_rate = 0.1;

    auto history = gradient_descent(kernel, dataset, learning_rate, MAX_EPOCHS, check_instance_callback);

    std::cout << "Original kernel is:\n\n" << std::fixed << std::setprecision(2) << Sobel.Gx << "\n\n";

    std::cout << "Trained kernel is:\n\n" << std::fixed << std::setprecision(2) << kernel << "\n\n";

    plot_performance(history);

    return 0;
}