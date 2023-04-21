#include <iostream>
#include <Eigen/Dense>

int main(int, char **) {

    Eigen::MatrixXd A(2, 2);
    A(0, 0) = 2.;
    A(1, 0) = -2.;
    A(0, 1) = 3.;
    A(1, 1) = 1.;

    Eigen::MatrixXd B(2, 3);
    B(0, 0) = 1.;
    B(1, 0) = 1.;
    B(0, 1) = 2.;
    B(1, 1) = 2.;
    B(0, 2) = -1.;
    B(1, 2) = 1.;

    auto C = A * B;

    std::cout << "A:\n" << A << std::endl;
    std::cout << "B:\n" << B << std::endl;
    std::cout << "C:\n" << C << std::endl;

    auto D = B.cwiseProduct(C);
    std::cout << "coeficient-wise multiplication of B & C is:\n" << D << std::endl;

    auto E = B + C;
    std::cout << "The sum of B & C is:\n" << E << std::endl;

    std::cout << "The transpose of B is:\n" << B.transpose() << std::endl;

    std::cout << "The A inverse is:\n" << A.inverse() << std::endl;

    std::cout << "The determinant of A is:\n" << A.determinant() << std::endl;

    auto my_func = [](double x){return x * x;};
    std::cout << A.unaryExpr(my_func) << std::endl;

    return 0;
}
