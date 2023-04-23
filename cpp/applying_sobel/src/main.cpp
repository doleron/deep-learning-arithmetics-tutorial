#include <iostream>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using Matrix = Eigen::MatrixXd;

int main(int, char **) {

    auto Convolution2D_v2 = [](const Matrix &input, const Matrix &kernel, int padding){

        const int input_rows = input.rows();
        const int input_cols = input.cols();

        const int kernel_rows = kernel.rows();
        const int kernel_cols = kernel.cols();

        if (input_rows < kernel_rows) throw std::invalid_argument("The input has less rows than the kernel");
        if (input_cols < kernel_cols) throw std::invalid_argument("The input has less columns than the kernel");

        const int rows = input_rows - kernel_rows + 2*padding + 1;
        const int cols = input_cols - kernel_cols + 2*padding + 1;

        Matrix result = Matrix::Zero(rows, cols);

        auto fit_dims = [&padding](int pos, int k, int length) {
            int input = pos - padding;
            int kernel = 0;
            int size = k;

            if (input < 0) {
                kernel = -input;
                size += input;
                input = 0;
            }

            if (input + size > length) {
                size = length - input;
            }

            return std::make_tuple(input, kernel, size);
        };

        for(int i = 0; i < rows; ++i) {

            const auto [input_i, kernel_i, size_i] = fit_dims(i, kernel_rows, input_rows);

            for(int j = 0; size_i > 0 && j < cols; ++j) {

                const auto [input_j, kernel_j, size_j] = fit_dims(j, kernel_cols, input_cols);

                if (size_j > 0) {
                    auto input_tile = input.block(input_i, input_j, size_i, size_j);
                    auto input_kernel = kernel.block(kernel_i, kernel_j, size_i, size_j);
                    result(i, j) = input_tile.cwiseProduct(input_kernel).sum();
                }
            }
        }
        return result;
    }; 

    Matrix Gx(3, 3), Gy(3, 3);

    Gx << 
        1., 0., -1.,
        2., 0., -2.,
        1., 0., -1.;

    Gy << 
        1., 2., 1.,
        0., 0., 0.,
        -1., -2., -1.;

    cv::Mat source = cv::imread("../francisco_brennand.png", cv::IMREAD_COLOR);
    cv::Mat image;
    cv::cvtColor(source, image, cv::COLOR_BGR2GRAY);
    Matrix X;
    cv::cv2eigen(image, X);

    const int padding = (Gx.rows() - 1) / 2;
    auto Gx_convoluted = Convolution2D_v2(X, Gx, padding);
    auto Gy_convoluted = Convolution2D_v2(X, Gy, padding);

    cv::Mat temp, Gx_result, Gy_result;

    cv::eigen2cv(Gx_convoluted, temp);
    temp.convertTo(Gx_result, CV_8UC1);
    
    cv::eigen2cv(Gy_convoluted, temp);
    temp.convertTo(Gy_result, CV_8UC1);

    cv::imshow("source", source);
    cv::imshow("gray scale", image);
    cv::imshow("applying Gx", Gx_result);
    cv::imshow("applying Gy", Gy_result);

    cv::waitKey();

    return 0;

}