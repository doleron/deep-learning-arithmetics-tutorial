#ifndef IMAGE_IO_TUTORIAL_H_
#define IMAGE_IO_TUTORIAL_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using Mat = cv::Mat;

Mat resize_image(const Mat &image, int target_rows, int target_cols)
{
    const int image_rows = image.rows;
    const int image_cols = image.cols;

    int new_rows = 0;
    int new_cols = 0;

    if (image_rows > image_cols) {
        new_rows = target_rows;
        new_cols = image_cols * target_rows / image_rows;
    } else {
        new_cols = target_cols;
        new_rows = image_rows * target_cols / image_cols;
    }
    Mat resized;
    resize(image, resized, cv::Size(new_cols, new_rows), cv::INTER_LINEAR);

    Mat result = Mat::zeros(cv::Size(target_cols, target_rows), CV_8UC1);

    resized.copyTo(result(cv::Rect((target_cols - new_cols)/2, (target_rows - new_rows)/2, resized.cols, resized.rows)));

    return result;
}

auto load_dataset = [](std::string data_folder, const int padding, bool show_images = false) {

    Dataset dataset;

    std::vector<std::string> files;

    for (const auto & entry : fs::directory_iterator(data_folder)) {
        files.push_back(data_folder + entry.path().c_str());
    }
    std::sort(files.begin(), files.end());

    for (const auto & file : files) {
        Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
        Mat formatted_image = resize_image(image, 640, 640);
        Matrix X;
        cv::cv2eigen(formatted_image, X);

        X /= 255.;
        auto Y = Convolution2D(X, Sobel.Gx, padding);

        auto pair = std::make_pair(X, Y);
        dataset.push_back(pair);

        if (show_images) {
            cv::Mat temp, Y_mat;
            cv::eigen2cv(Y, temp);
            temp *= 255.;
            temp.convertTo(Y_mat, CV_8UC1);
            
            cv::imshow("X", formatted_image);
            cv::imshow("Y", Y_mat);
            cv::waitKey();
        }
    }

    if (show_images) {
        cv::destroyAllWindows();
    }

    return dataset;

};

auto plot_performance = [](std::vector<double> &losses)
{
    const int N = losses.size();

    int begin_x = 100;
    const int width = N + 2*begin_x;
    const int height = 600;

    Mat result = Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    int end_x = width - begin_x;
    int step_x = (end_x - begin_x) / 10;

    int begin_y = 100;
    int end_y = height - begin_y;
    int step_y = (end_y - begin_y) / 10;

    double max = *std::max_element(losses.begin(), losses.end());

    double min = *std::min_element(losses.begin(), losses.end());

    double delta = (max - min) / 10;

    max += delta;
    min -= delta;

    double step_loss = (max - min) / 10;

    cv::line(result, cv::Point(begin_x, begin_y), cv::Point(begin_x, end_y), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::line(result, cv::Point(begin_x, end_y), cv::Point(end_x, end_y), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    for (int i = 1; i <= 10; ++i) {

        int anchor_x = begin_x + step_x * i;

        int anchor_y = end_y - step_y * i;

        double loss = step_loss * (i + 1) + min;

        std::stringstream stream_loss;
        stream_loss << std::fixed << std::setprecision(3) << loss;
        cv::putText(result, stream_loss.str(), cv::Point(begin_x - 50, anchor_y + 5), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 0), 1);
        cv::line(result, cv::Point(begin_x - 5, anchor_y), cv::Point(begin_x + 5, anchor_y), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        cv::line(result, cv::Point(anchor_x, end_y - 5), cv::Point(anchor_x, end_y + 5), cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(result, std::to_string((N * i) / 10), cv::Point(anchor_x - 5, end_y + 20), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 0), 1);

    }

    auto convert = [&min, &max, &begin_y, &end_y](double v) {
        double norm = (v - min) / (max - min);

        int result = static_cast<int>(norm * (end_y - begin_y));

        return result;
    };

    const int loss_step = (end_x - begin_x) / N;

    for (int i = 0; i < N - 1; ++i) {
        double from = losses[i];
        double to = losses[i + 1];
        int from_i = convert(from);
        int to_i = convert(to);
        int x = begin_x + loss_step * i;
        cv::line(result, cv::Point(x, end_y - from_i), cv::Point(x + loss_step, end_y - to_i), cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }

    cv::putText(result, "Epochs", cv::Point(width / 2 - 30, height - 50), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 0), 1);

    cv::imshow("Loss performance", result);
    cv::waitKey();
};

#endif