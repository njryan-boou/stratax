#include <iostream>

#include <stratax.h>

int main()
{
    using stratax::container::Tensor;
    using stratax::core::Shape;

    Tensor<double> tensor(Shape{2, 2, 2}, 1.0);

    tensor(0, 0, 0) = 10.0;
    tensor(1, 1, 1) = 20.0;

    auto shifted = tensor + 2.0;
    auto negated = -tensor;

    std::cout << "tensor:\n" << tensor << '\n';
    std::cout << "shifted:\n" << shifted << '\n';
    std::cout << "negated:\n" << negated << '\n';
    std::cout << "rank: " << tensor.rank() << '\n';
    std::cout << "shape: " << tensor.shape() << '\n';
    std::cout << "linear index 7: " << tensor[7] << '\n';

    return 0;
}
