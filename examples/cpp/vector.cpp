#include <iostream>

#include <stratax.h>

int main()
{
    using stratax::core::Shape;
    using stratax::core::Slice;
    using stratax::container::Vector;

    Vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0};
    Vector<double> weights(values.size(), 10.0);

    values(1) = 5.0;

    auto sum = values + weights;
    auto scaled = values * 2.0;
    auto stepped = slice(values, Slice{0, values.size(), 2});
    auto reshaped = reshape(values, Shape{1, values.size()});
    auto roundtrip = to_vector(reshaped);

    auto total = reduction::sum(values);
    auto average = reduction::mean(values);

    Vector<int> flags{3, 5, 6};
    auto masked = flags & 2;
    auto shifted_left = flags << 1;

    std::cout << "values: " << values << '\n';
    std::cout << "sum: " << sum << '\n';
    std::cout << "scaled: " << scaled << '\n';
    std::cout << "stepped slice [::2]: " << stepped << '\n';
    std::cout << "reshaped to tensor: " << reshaped << '\n';
    std::cout << "roundtrip to vector: " << roundtrip << '\n';
    std::cout << "sum(values): " << total << '\n';
    std::cout << "mean(values): " << average << '\n';
    std::cout << "flags: " << flags << '\n';
    std::cout << "flags & 2: " << masked << '\n';
    std::cout << "flags << 1: " << shifted_left << '\n';
    std::cout << "front: " << values.front() << '\n';
    std::cout << "shape: " << values.shape() << '\n';

    return 0;
}
