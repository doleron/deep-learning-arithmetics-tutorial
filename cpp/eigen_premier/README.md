# Eigen Premier
This is a simple project to make you start up using Eigen.

## Hidding Eigen
The Eigen types names are hidden in `matrix_defininitions.hpp` using Type Alias:

```c++
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using DiagonalMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
```
