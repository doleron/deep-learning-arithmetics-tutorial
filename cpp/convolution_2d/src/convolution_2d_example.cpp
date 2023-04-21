#include <iostream>

#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;

int main(int, char **) {

    Matrix kernel(3, 3);

    kernel << 
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1;

    std::cout << "Kernel:\n" << kernel << "\n\n";

    Matrix input(6, 6);
    input << 3, 1, 0, 2, 5, 6,
        4, 2, 1, 1, 4, 7,
        5, 4, 0, 0, 1, 2,
        1, 2, 2, 1, 3, 4,
        6, 3, 1, 0, 5, 2,
        3, 1, 0, 1, 3, 3;

    std::cout << "Input:\n" << input << "\n\n";

    auto Convolution2D = [](const Matrix &input, const Matrix &kernel){
        const int kernel_rows = kernel.rows();
        const int kernel_cols = kernel.cols();

        const int rows = (input.rows() - kernel_rows) + 1;
        const int cols = (input.cols() - kernel_cols) + 1;

        Matrix result = Matrix::Zero(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double sum = input.block(i, j, kernel_rows, kernel_cols).cwiseProduct(kernel).sum();
                result(i, j) = sum;
            }
        }

        return result;
    }; 

    auto output = Convolution2D(input, kernel);
    std::cout << "Convolution:\n" << output << "\n";

    return 0;
}