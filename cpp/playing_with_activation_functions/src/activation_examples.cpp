#include <iostream>

#include "activation_functions.hpp"

void plot(ActivationFunction * act, double from, double to, double step) {

    std::cout << act->get_name() << "\n\n";

    for (double x = from; x <= to; x += step) {
        Matrix X = Matrix::Ones(1, 1) * x;
        double Y = (*act)(X)(0,0);
        double dY = act->jacobian(X)(0,0);
        std:: cout << x << "\t" <<  Y << "\t" << dY << "\n";
    }

    std::cout << "\n\n";

    delete act;
}

int main() {

    plot(new Sigmoid(), -5., 5., 0.1);

    plot(new Tanh(), -5., 5., 0.1);

    plot(new ReLU(), -5., 5., 0.1);

    plot(new Identity(), -5., 5., 0.1);

    return 0;
}