#include <algorithm>
#include <numeric>
#include <iostream>

using vector = std::vector<double>;

int main() {

    auto MSE = [](const vector &Y_true, const vector &Y_pred) {

        if (Y_true.empty()) throw std::invalid_argument("Y_true cannot be empty.");

        if (Y_true.size() != Y_pred.size()) throw std::invalid_argument("Y_true and Y_pred sizes do not match.");

        const int n = Y_true.size();

        vector temp(n);
        std::transform(Y_true.begin(), Y_true.end(), Y_pred.begin(), temp.begin(),
                       [](double y_true, const double y_pred) {
                           double diff = y_true - y_pred;
                           return diff * diff;
                       });

        double sum = std::accumulate(temp.begin(), temp.end(), 0.0);
        double result = sum / n;

        return result;
    };

    vector Y{1., 0., 1., 1., 1., 0.};

    vector T{0., 0., 1., 1., 0., 0.};

    std::cout << "MSE: " << MSE(Y, T) << "\n";

    return 0;
}