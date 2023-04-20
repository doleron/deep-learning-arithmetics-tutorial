#include <numeric>
#include <vector>
#include <iostream>

int main() {

    std::vector<double> X {1., 2., 3., 4., 5., 6.};
    std::vector<double> Y {1., 1., 0., 1., 0., 1.};
 
    auto result = std::inner_product(X.begin(), X.end(), Y.begin(), 0.0);
    std::cout << "Inner product of X and Y is " << result << '\n';

    std::vector<double> V {1., 2., 3., 4., 5.};
 
    double sum = std::accumulate(V.begin(), V.end(), 0.0);

    std::cout << "Summation of V is " << sum << '\n';

    double product = std::accumulate(V.begin(), V.end(), 1.0, std::multiplies<double>());

    std::cout << "Productory of V is " << product << '\n';

    double reduction = std::reduce(V.begin(), V.end(), 1.0,  std::multiplies<double>());

    std::cout << "Reduction of V is " << reduction << '\n';

    return 0;
 
}