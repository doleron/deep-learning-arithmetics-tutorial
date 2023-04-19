#ifndef MATRIX_DEFINITIONS_H_
#define MATRIX_DEFINITIONS_H_

#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using DiagonalMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

#endif