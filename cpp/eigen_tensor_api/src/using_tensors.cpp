#include <iostream>

#include <numeric>

#include <unsupported/Eigen/CXX11/Tensor>

int main(int, char **)
{

    {
        // Tensor initialization

        Eigen::Tensor<int, 3> my_tensor(2, 3, 4);
        my_tensor.setConstant(42);

        my_tensor.setValues({{{1, 2, 3, 4}, {5, 6, 7, 8}}});

        std::cout << "my_tensor:\n\n"
                  << my_tensor << "\n\n";

        my_tensor.setRandom();

        std::cout << "my_tensor:\n\n"
                  << my_tensor << "\n\n";

        std::cout << "tensor size is " << my_tensor.size() << "\n\n";

        Eigen::Tensor<float, 2> kernel(3, 3);
        kernel.setRandom();
        std::cout << "kernel:\n\n"
                  << kernel << "\n\n";
    }

    {
        // TensorMap example

        std::vector<float> storage(4 * 3);

        std::iota(storage.begin(), storage.end(), 1.);

        for (float v : storage)
            std::cout << v << ',';
        std::cout << "\n\n";

        Eigen::TensorMap<Eigen::Tensor<float, 2>> my_tensor_view(storage.data(), 4, 3);

        std::cout << "my_tensor_view before update:\n\n"
                  << my_tensor_view << "\n\n";

        storage[4] = -1.;

        std::cout << "my_tensor_view after update:\n\n"
                  << my_tensor_view << "\n\n";

        my_tensor_view(2, 1) = -8;

        std::cout << "vector after two updates:\n\n";
        for (float v : storage)
            std::cout << v << ',';
        std::cout << "\n\n";
    }

    {
        // expressions

        Eigen::Tensor<float, 2> A(2, 3), B(2, 3);
        A.setRandom();
        B.setRandom();
        Eigen::Tensor<float, 2> C = 2.f * A + B.exp();

        std::cout << "A is\n\n"
                  << A << "\n\n";
        std::cout << "B is\n\n"
                  << B << "\n\n";
        std::cout << "C is\n\n"
                  << C << "\n\n";

        auto cosine = [](float v)
        { return cos(v); };

        Eigen::Tensor<float, 2> D = A.unaryExpr(cosine);

        std::cout << "D is\n\n"
                  << D << "\n\n";

        auto fun = [](float a, float b)
        { return 2 * a + b; };

        Eigen::Tensor<float, 2> E = A.binaryExpr(B, fun);

        std::cout << "E is\n\n"
                  << E << "\n\n";
    }

    {
        // reductions example

        Eigen::Tensor<float, 3> X(5, 2, 3);
        X.setRandom();

        std::cout << "X is\n\n"
                  << X << "\n\n";

        std::cout << "X.sum(): " << X.sum() << "\n\n";
        std::cout << "X.maximum(): " << X.maximum() << "\n\n";

        Eigen::array<int, 2> dims({1, 2});

        std::cout << "X.sum(dims): " << X.sum(dims) << "\n\n";
        std::cout << "X.maximum(dims): " << X.maximum(dims) << "\n\n";
    }

    {
        // convolution example

        Eigen::Tensor<float, 4> input(1, 6, 6, 3);
        input.setRandom();

        Eigen::Tensor<float, 2> kernel(3, 3);
        kernel.setRandom();

        Eigen::array<int, 2> dims({1, 2});
        Eigen::Tensor<float, 4> output(1, 4, 4, 3);
        output = input.convolve(kernel, dims);

        std::cout << "input:\n\n"
                  << input << "\n\n";
        std::cout << "kernel:\n\n"
                  << kernel << "\n\n";
        std::cout << "output:\n\n"
                  << output << "\n\n";
    }

    {
        // transpose example

        auto transpose = [](const Eigen::Tensor<float, 2> &tensor)
        {
            Eigen::array<int, 2> dims({1, 0});

            return tensor.shuffle(dims);
        };

        Eigen::Tensor<float, 2> a_tensor(3, 4);
        a_tensor.setRandom();

        std::cout << "a_tensor is\n\n"
                  << a_tensor << "\n\n";

        std::cout << "a_tensor transpose is\n\n"
                  << transpose(a_tensor) << "\n\n";
    }

    {
        // reshape example

        Eigen::Tensor<float, 2> X(2, 3);
        X.setValues({{1, 2, 3}, {4, 5, 6}});

        std::cout << "X is\n\n"
                  << X << "\n\n";

        std::cout << "Size of X is " << X.size() << "\n\n";

        Eigen::array<int, 3> new_dims({3, 1, 2});
        Eigen::Tensor<float, 3> Y = X.reshape(new_dims);

        std::cout << "Y is\n\n"
                  << Y << "\n\n";

        std::cout << "Size of Y is " << Y.size() << "\n\n";
    }

    {
        // broadcast example

        Eigen::Tensor<float, 2> Z(1, 3);
        Z.setValues({{1, 2, 3}});
        Eigen::array<int, 2> bcast({4, 2});
        Eigen::Tensor<float, 2> W = Z.broadcast(bcast);

        std::cout << "Z is\n\n"
                  << Z << "\n\n";
        std::cout << "W is\n\n"
                  << W << "\n\n";
    }

    return 0;
}