#ifndef ACTIVATION_FUNCIONS_H_
#define ACTIVATION_FUNCIONS_H_

#include <functional>

#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using DiagonalMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

class ActivationFunction
{
    public:
        ActivationFunction(std::string _name): name(_name) {}
        virtual Matrix operator()(const Matrix &z) const = 0;
        virtual Matrix jacobian(const Vector &z) const = 0;
        const std::string get_name() const {
            return this->name;
        };
        virtual ~ActivationFunction() {}
    private:
        std::string name;

};

class Sigmoid : public ActivationFunction
{
    public:

        Sigmoid(): ActivationFunction("sigmoid") {}

        virtual Matrix operator()(const Matrix &z) const
        {
            return z.unaryExpr(std::ref(Sigmoid::helper));
        }

        virtual Matrix jacobian(const Vector &z) const
        {
            Vector output = (*this)(z);

            Vector diagonal = output.unaryExpr([](double y) {
                return (1.0 - y) * y;
            });

            DiagonalMatrix result = diagonal.asDiagonal();

            return result;
        }
        virtual ~Sigmoid() {}

    private:

        static double helper(double z)
        {
            double result;
            if (z >= 45) result = 1;
            else if (z <= -45) result = 0;
            else result = 1.0 / (1.0 + exp(-z));
            return result;
        }

};

class Tanh : public ActivationFunction
{
    public:

        Tanh(): ActivationFunction("tanh") {}
        virtual Matrix operator()(const Matrix &z) const
        {
            return z.unaryExpr(std::ref(tanh));
        }

        virtual Matrix jacobian(const Vector &z) const
        {
            Vector output = (*this)(z);

            Vector diagonal = output.unaryExpr([](double y) {
                return (1.0 - y * y);
            });

            DiagonalMatrix result = diagonal.asDiagonal();

            return result;
        }
        virtual ~Tanh() {}
};

class ReLU : public ActivationFunction
{

    public:

        ReLU(): ActivationFunction("relu") {}

        virtual Matrix operator()(const Matrix &z) const
        {
            return z.unaryExpr([](double v) {
                return std::max(0.0, v);
            });
        }

        virtual Matrix jacobian(const Vector &z) const
        {

            Vector output = (*this)(z);
            Vector diagonal = output.unaryExpr([](double y) {
                double result = 0.;
                if (y > 0) result = 1.; 
                return result;
            });

            DiagonalMatrix result = diagonal.asDiagonal();

            return result;
        }
        virtual ~ReLU() {}

};

class Identity : public ActivationFunction
{
    public:

        Identity(): ActivationFunction("identity") {}
        virtual Matrix operator()(const Matrix &z) const { return z; }

        virtual Matrix jacobian(const Vector &z) const
        {

            Vector diagonal = Vector::Ones(z.rows());

            DiagonalMatrix result = diagonal.asDiagonal();

            return result;
        }
        virtual ~Identity() {}
};

class Softmax : public ActivationFunction
{
    public:

        Softmax(): ActivationFunction("softmax") {}
        virtual Matrix operator()(const Matrix &z) const
        {

            if (z.rows() == 1)
            {
                throw std::invalid_argument("Softmax is not suitable for single value outputs. Use sigmoid/tanh instead.");
            }
            Vector maxs = z.colwise().maxCoeff();
            Matrix reduc = z.rowwise() - maxs.transpose();
            Matrix expo = reduc.array().exp();
            Vector sums = expo.colwise().sum();
            Matrix result = expo.array().rowwise() / sums.transpose().array();
            return result;
        }

        virtual Matrix jacobian(const Vector &z) const
        {
            Matrix output = (*this)(z);

            Matrix outputAsDiagonal = output.asDiagonal();

            Matrix result = outputAsDiagonal - (output * output.transpose());

            return result;
        }
        virtual ~Softmax() {}

};

#endif