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

auto gradient = [](const std::vector<Matrix> &xs, std::vector<Matrix> &ys, std::vector<Matrix> &ts, const int padding)
{
    const int N = xs.size();
    Matrix result;
    for (int n = 0; n < N; ++n) {
        const auto &X = xs[n];
        const auto &Y = ys[n];
        const auto &T = ts[n];

        Matrix error = T - Y;
        Matrix update = Convolution2D(X, error, padding);
        if (result.size() == 0) {
            result = Matrix::Zero(update.rows(), update.cols());
        }
        result = result + update;
    }

    const int R = xs[0].rows();
    const int C = xs[0].cols();

    result *= 2.0/(R * C);

    return result;
};

using Dataset = std::vector<std::pair<Matrix, Matrix>>;

auto momentum_optimizer = [V = Matrix()](const Matrix &gradient) mutable 
    {
        if (V.size() == 0) V = Matrix::Zero(gradient.rows(), gradient.cols());
        
        double beta = 0.7;

        V *= beta;

        V += gradient;
        return V;
    };

auto gradient_descent = [](Matrix &kernel, Dataset &dataset, const double learning_rate, const int MAX_EPOCHS,
                            std::function<void(int, double)> epoch_callback = [](int, double) {})
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

        auto grad = gradient(xs, ys, ts, padding);

        auto update = grad * learning_rate;

        kernel -= momentum_optimizer(update);

        double loss = MSE(ys, ts);

        losses.push_back(loss);

        epoch_callback(epoch, loss);

        epoch++;
    }

    return losses;
};

struct {

    Matrix Gx = (Eigen::Matrix3d() << 1., 0., -1., 2., 0., -2.,1., 0., -1.).finished();
    Matrix Gy = (Eigen::Matrix3d() << 1., 2., 1.,0., 0., 0.,-1., -2., -1.).finished();

} Sobel;

#endif