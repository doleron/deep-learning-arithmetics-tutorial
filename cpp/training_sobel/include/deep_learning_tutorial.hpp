#ifndef DEEP_LEARNING_TUTORIAL_H_
#define DEEP_LEARNING_TUTORIAL_H_

#include <numeric>

#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;

auto Convolution2D = [](const Matrix &input, const Matrix &kernel, int padding)
{
    const int input_rows = input.rows();
    const int input_cols = input.cols();

    const int kernel_rows = kernel.rows();
    const int kernel_cols = kernel.cols();

    if (input_rows < kernel_rows)
        throw std::invalid_argument("The input has less rows than the kernel");
    if (input_cols < kernel_cols)
        throw std::invalid_argument("The input has less columns than the kernel");

    const int rows = input_rows - kernel_rows + 2 * padding + 1;
    const int cols = input_cols - kernel_cols + 2 * padding + 1;

    Matrix result = Matrix::Zero(rows, cols);

    auto fit_dims = [&padding](int pos, int k, int length)
    {
        int input = pos - padding;
        int kernel = 0;
        int size = k;

        if (input < 0)
        {
            kernel = -input;
            size += input;
            input = 0;
        }

        if (input + size > length)
        {
            size = length - input;
        }

        return std::make_tuple(input, kernel, size);
    };

    for (int i = 0; i < rows; ++i)
    {

        const auto [input_i, kernel_i, size_i] = fit_dims(i, kernel_rows, input_rows);

        for (int j = 0; size_i > 0 && j < cols; ++j)
        {

            const auto [input_j, kernel_j, size_j] = fit_dims(j, kernel_cols, input_cols);

            if (size_j > 0)
            {
                auto input_tile = input.block(input_i, input_j, size_i, size_j);
                auto input_kernel = kernel.block(kernel_i, kernel_j, size_i, size_j);
                result(i, j) = input_tile.cwiseProduct(input_kernel).sum();
            }
        }
    }
    return result;
};

auto MSE = [](const std::vector<Matrix> &Y_true, const std::vector<Matrix> &Y_pred)
{
    if (Y_true.empty())
        throw std::invalid_argument("Y_true cannot be empty.");

    if (Y_true.size() != Y_pred.size())
        throw std::invalid_argument("Y_true and Y_pred sizes do not match.");

    const int N = Y_true.size();
    const int R = Y_true[0].rows();
    const int C = Y_true[0].cols();

    auto quadratic = [](const Matrix a, const Matrix b)
    {
        Matrix result = a - b;
        return result.cwiseProduct(result).sum();
    };

    double acc = std::inner_product(Y_true.begin(), Y_true.end(), Y_pred.begin(), 0.0, std::plus<>(), quadratic);

    double result = acc / (N * R * C);

    return result;
};

auto gradient = [](const std::vector<Matrix> xs, std::vector<Matrix> ys, std::vector<Matrix> ts, const int padding)
{
    return Matrix::Zero(3, 3);
};

using Dataset = std::vector<std::pair<Matrix, Matrix>>;

auto gradient_descent = [](
                            Matrix &kernel, Dataset &dataset, const double learning_rate, const int MAX_EPOCHS,
                            std::function<void(int)> epoch_callback = [](int) {})
{
    std::vector<double> losses;
    losses.reserve(MAX_EPOCHS);

    const int padding = kernel.rows() / 2;

    const int N = dataset.size();
    std::vector<Matrix> xs;
    xs.reserve(N);
    std::vector<Matrix> ys;
    ys.reserve(N);
    std::vector<Matrix> ts;
    ts.reserve(N);

    int epoch = 0;
    while (epoch < MAX_EPOCHS)
    {
        xs.clear();
        ys.clear();
        ts.clear();

        for (std::pair<Matrix, Matrix> &instance : dataset)
        {
            const auto &X = instance.first;
            const auto &Y = instance.second;
            const auto T = Convolution2D(X, kernel, padding);

            xs.push_back(X);
            ys.push_back(Y);
            ts.push_back(T);
        }

        losses.push_back(MSE(ys, ts));

        auto grad = gradient(xs, ys, ts, padding);

        auto update = grad * learning_rate;

        kernel -= update;

        epoch++;

        epoch_callback(epoch);
    }

    return losses;
};

#endif