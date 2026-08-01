#include <iostream>

#include <stratax.h>

int main()
{
    // Build a rank-1 vector and a same-sized vector used for element-wise work.
    stratax::Vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0};
    stratax::Vector<double> weights(values.size(), 10.0);

    // Checked access via at() supports negative indexes; operator() is unchecked.
    values(1) = 5.0;
    double last_value = values.at(-1);

    // Arithmetic works with matching containers or scalars.
    auto sum = values + weights;
    auto scaled = values * 2.0;

    // Slice, reshape, and conversion helpers preserve flat row-major values.
    auto stepped = stratax::slice(values, stratax::Slice{0, values.size(), 2});
    auto reshaped = stratax::reshape(values, stratax::Shape{1, values.size()});
    auto roundtrip = stratax::to_vector(reshaped);

    // Reductions return scalars without an axis.
    auto total = stratax::sum(values);
    auto average = stratax::mean(values);

    // Integral containers can use bitwise operators.
    stratax::Vector<int> flags{3, 5, 6};
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
    std::cout << "last value via at(-1): " << last_value << '\n';
    std::cout << "shape: " << values.shape() << '\n';
    std::cout << "strides: " << values.strides() << '\n';

    return 0;
}
