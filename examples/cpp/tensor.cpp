#include <iostream>

#include <stratax.h>

int main()
{
    using stratax::container::Tensor;
    using stratax::core::Shape;
    using stratax::core::Slice;

    Tensor<double> tensor(Shape{2, 2, 2}, 1.0);
    auto eye = creation::identity<double>(3);

    tensor(0, 0, 0) = 10.0;
    tensor(1, 1, 1) = 20.0;

    auto shifted = tensor + 2.0;
    auto negated = -tensor;
    auto stepped = slice(tensor, Slice{0, 2}, Slice{0, 2}, Slice{0, 2, 2});
    auto reshaped = reshape(tensor, Shape{4, 2});
    auto flattened = flatten(tensor);
    auto as_matrix = to_matrix(eye);
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
    std::cout << "linear index 7: " << tensor[7] << '\n';

    return 0;
}
