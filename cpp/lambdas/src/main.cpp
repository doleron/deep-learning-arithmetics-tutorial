#include <algorithm>
#include <numeric>
#include <iostream>

using vector = std::vector<double>;

int main() {

    auto L2 = [](const vector &V)
    {
        return std::inner_product(V.begin(), V.end(), V.begin(), 0.0);
    };

    vector weights{1., 2., 3., 4., 5., 6.};

    std::cout << L2(weights) << "\n";

    auto momentum_optimizer = [V = vector()](const vector &gradient) mutable {
        if (V.empty()) V.resize(gradient.size(), 0.);
        std::transform(V.begin(), V.end(), gradient.begin(), V.begin(), [](double v, double dx) {
            double beta = 0.3;
            return v = beta * v + dx; 
        });
        return V;
    };

    auto print = [](double d) { std::cout << d << " "; };

    const vector current_grads{1., 0., 1., 1., 0., 1.};
    for (int i = 0; i < 3; ++i) {
        vector weight_update = momentum_optimizer(current_grads);
        std::for_each(weight_update.begin(), weight_update.end(), print);
        std::cout << "\n";
    }

    return 0;
}