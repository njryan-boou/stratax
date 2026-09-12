#include <iostream>

#include <stratax.h>

int main()
{
    // Tensors store flat row-major values behind N-dimensional shape metadata.
    stratax::container::Tensor<double> tensor(stratax::core::Shape{2, 2, 2}, 1.0);
    auto eye = stratax::creation::identity<double>(3);

    // Multi-index access follows the tensor rank.
    tensor(0, 0, 0) = 10.0;
    tensor(1, 1, 1) = 20.0;
    double last_value = tensor[tensor.size() - 1];
    double far_corner = tensor.at(-1, -1, -1);

    // Tensor arithmetic supports scalars and broadcast-compatible arrays.
    auto shifted = tensor + 2.0;
    auto negated = -tensor;

    // Slicing can mix full-axis slices and stepped slices.
    auto stepped = stratax::indexing::slice(
        tensor,
        stratax::core::Slice{0, 2},
        stratax::core::Slice{0, 2},
        stratax::core::Slice{0, 2, 2});

    // Reshape, flatten, and conversion helpers preserve row-major values.
    auto reshaped = stratax::manipulation::reshape(tensor, stratax::core::Shape{4, 2});
    auto flattened = stratax::manipulation::flatten(tensor);
    auto as_matrix = stratax::conversion::to_matrix(eye);

    // Reductions can target negative axes and can keep reduced dimensions.
    auto sum_all = reduction::sum(tensor);
    auto sum_last_axis = reduction::sum(tensor, -1);
    auto mean_keepdims = reduction::mean(tensor, 0, true);

    std::cout << "tensor:\n" << tensor << '\n';
    std::cout << "shifted:\n" << shifted << '\n';
    std::cout << "negated:\n" << negated << '\n';
    std::cout << "stepped slice [:, :, ::2]:\n" << stepped << '\n';
    std::cout << "reshaped tensor:\n" << reshaped << '\n';
    std::cout << "flattened vector: " << flattened << '\n';
    std::cout << "identity via creation helper:\n" << eye << '\n';
    std::cout << "identity converted to matrix:\n" << as_matrix << '\n';
    std::cout << "sum(tensor): " << sum_all << '\n';
    std::cout << "sum over last axis (axis=-1):\n" << sum_last_axis << '\n';
    std::cout << "mean keepdims over axis 0:\n" << mean_keepdims << '\n';
    std::cout << "rank: " << tensor.rank() << '\n';
    std::cout << "shape: " << tensor.shape() << '\n';
    std::cout << "strides: " << tensor.strides() << '\n';
    std::cout << "linear index 7: " << tensor[7] << '\n';
    std::cout << "last value via flat index: " << last_value << '\n';
    std::cout << "far corner via at(-1, -1, -1): " << far_corner << '\n';

    return 0;
}
