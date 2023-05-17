#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

auto softmax(const Eigen::Tensor<float, 3> &z)
{

    auto dimensions = z.dimensions();

    int batches = dimensions.at(0);
    int instances_per_batch = dimensions.at(1);
    int instance_length = dimensions.at(2);

    Eigen::array<int, 1> depth_dim({2});
    auto z_max = z.maximum(depth_dim);

    Eigen::array<int, 3> reshape_dim({batches, instances_per_batch, 1});
    auto max_reshaped = z_max.reshape(reshape_dim);

    Eigen::array<int, 3> bcast = Eigen::array<int, 3>({1, 1, instance_length});
    auto max_values = max_reshaped.broadcast(bcast);

    auto diff = z - max_values;

    auto expo = diff.exp();
    auto expo_sums = expo.sum(depth_dim);
    auto sums_reshaped = expo_sums.reshape(reshape_dim);
    auto sums = sums_reshaped.broadcast(bcast);
    auto result = expo / sums;

    return result;
}

int main(int, char **)
{
    Eigen::Tensor<float, 3> input(2, 4, 3);
    input.setValues({
        {{0.1, 1., -2.},{10., 2., 5.},{5., -5., 0.},{2., 3., 2.}},
        {{100., 1000., -500.},{3., 3., 3.},{-1, 1., -1.},{-11., -0.2, -.1}}
    });

    std::cout << "input:\n\n" << input << "\n\n";

    Eigen::Tensor<float, 3> output = softmax(input);
    std::cout << "output:\n\n" << output << "\n\n";

    std::cout << "output dims: \n\n" << output.dimensions() << "\n\n";

    Eigen::array<int, 1> depth_dim({2});
    Eigen::Tensor<float, 2> test = output.sum(depth_dim);
    std::cout << "test:\n\n" << test << "\n\n";

    return 0;
}
