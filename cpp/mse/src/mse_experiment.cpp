#include <numeric>
#include <algorithm>
#include <iostream>
#include <random>

std::default_random_engine dre(time(0));

int main() {

    auto MSE = [](const std::vector<double> &Y_true, const std::vector<double> &Y_pred) {

        if (Y_true.empty()) throw std::invalid_argument("Y_true cannot be empty.");

        if (Y_true.size() != Y_pred.size()) throw std::invalid_argument("Y_true and Y_pred sizes do not match.");

        const int N = Y_true.size();

        auto quadratic = [](const double a, const double b) {
            double result = a - b;
            return result * result;
        };

        double acc = std::inner_product(Y_true.begin(), Y_true.end(), Y_pred.begin(), 0.0, std::plus<>(), quadratic);

        double result = acc / N;

        return result;
    };

    std::normal_distribution<double> gaussian_dist(0., 0.1);
    std::uniform_real_distribution<double> uniform_dist(0., 1.);

    std::vector<std::pair<double, double>> sample(60);
 
    std::generate(sample.begin(), sample.end(), [&gaussian_dist, &uniform_dist]() {
        double x = uniform_dist(dre);
        double noise = gaussian_dist(dre);
        double y = 2. * x + noise;
        return std::make_pair(x, y);
    });

    for (auto & s : sample) {
        std::cout << s.first << "\t" << s.second << "\n";
    }

    std::vector<double> ys(sample.size());
    std::transform(sample.begin(), sample.end(), ys.begin(), [](const auto &pair) {
        return pair.second;
    });

    std::vector<std::pair<double, double>> measures;

    double smallest_mse = 1'000'000'000.;
    double best_k = -1;

    for (double k = 0.; k < 4.1; k += 0.1) {
        std::vector<double> ts(sample.size());
        std::transform(sample.begin(), sample.end(), ts.begin(), [k](const auto &pair) {
            return pair.first * k;
        });

        double mse = MSE(ys, ts);
        if (mse < smallest_mse) {
            smallest_mse = mse;
            best_k = k;
        }

        measures.push_back(std::make_pair(k, mse));
    }

    for (auto & mse : measures) {
        std::cout << mse.first << "\t" << mse.second << "\n";
    }

    std::cout << "best k was " << best_k << " for a MSE of " << smallest_mse << "\n";

    return 0;
}