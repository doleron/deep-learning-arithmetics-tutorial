#include <numeric>
#include <iostream>

#include <Eigen/Core>

using Eigen::MatrixXd;

int main() {

    auto MSE = [](const std::vector<MatrixXd> &Y_true, const std::vector<MatrixXd> &Y_pred) {

        if (Y_true.empty()) throw std::invalid_argument("Y_true cannot be empty.");

        if (Y_true.size() != Y_pred.size()) throw std::invalid_argument("Y_true and Y_pred sizes do not match.");

        const int N = Y_true.size();
        const int R = Y_true[0].rows();
        const int C = Y_true[0].cols();

        auto quadratic = [](const MatrixXd a, const MatrixXd b) {
            MatrixXd result = a - b;
            return result.cwiseProduct(result).sum();
        };

        double acc = std::inner_product(Y_true.begin(), Y_true.end(), Y_pred.begin(), 0.0, std::plus<>(), quadratic);

        double result = acc / (N * R * C);

        return result;
    };

    std::cout << "MSE: " << MSE(
        {MatrixXd::Constant(1, 1, 1.), MatrixXd::Constant(1, 1, 0.), MatrixXd::Constant(1, 1, 1.), MatrixXd::Constant(1, 1, 1.), MatrixXd::Constant(1, 1, 1.), MatrixXd::Constant(1, 1, 0.)}, 
        {MatrixXd::Constant(1, 1, 0.), MatrixXd::Constant(1, 1, 0.), MatrixXd::Constant(1, 1, 1.), MatrixXd::Constant(1, 1, 1.), MatrixXd::Constant(1, 1, 0.), MatrixXd::Constant(1, 1, 0.)}) 
    << "\n\n";

    std::cout << "MSE: " << MSE(
        {MatrixXd::Constant(1, 1, 1.9), MatrixXd::Constant(1, 1, -1.), MatrixXd::Constant(1, 1, 3.5), MatrixXd::Constant(1, 1, 0.)}, 
        {MatrixXd::Constant(1, 1, 2.), MatrixXd::Constant(1, 1, -1.2), MatrixXd::Constant(1, 1, 4.1), MatrixXd::Constant(1, 1, 0.2)}) 
    << "\n\n";

    std::vector<MatrixXd> Y(4, MatrixXd::Zero(1, 2)); 
    Y[0] << 1., 2.; Y[1] << 0., 1.; Y[2] << -1., 1.; Y[3] << -2., 3.;

    std::vector<MatrixXd> T(4, MatrixXd::Zero(1, 2)); 
    T[0] << 0.5, 2.; T[1] << 0.5, 1.5; T[2] << -2.5, 1.; T[3] << -3.5, 3.5;

    std::cout << "MSE: " << MSE(Y, T) << "\n\n";

    std::vector<MatrixXd> A(4, MatrixXd::Zero(2, 3)); 
    A[0] << 1., 2., 1., -3., 0, 2.;
    A[1] << 5., -1., 3., 1., 0.5, -1.5; 
    A[2] << -2., -2., 1., 1., -1., 1.; 
    A[3] << -2., 0., 1., -1., -1., 3.;

    std::vector<MatrixXd> B(4, MatrixXd::Zero(2, 3)); 
    B[0] << 0.5, 2., 1., 1., 1., 2.; 
    B[1] << 4., -2., 2.5, 0.5, 1.5, -2.; 
    B[2] << -2.5, -2.8, 0., 1.5, -1.2, 1.8; 
    B[3] << -3., 1., -1., -1., -1., 3.5;

    std::cout << "MSE: " << MSE(A, B) << "\n";

    return 0;
}