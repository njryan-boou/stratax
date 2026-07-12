#include <iostream>

#include <stratax.h>

int main()
{
    using stratax::container::Vector;

    Vector<double> values{1.0, 2.0, 3.0};
    Vector<double> weights(3, 10.0);

    values(1) = 5.0;

    auto sum = values + weights;
    auto scaled = values * 2.0;

    std::cout << "values: " << values << '\n';
    std::cout << "sum: " << sum << '\n';
    std::cout << "scaled: " << scaled << '\n';
    std::cout << "front: " << values.front() << '\n';
    std::cout << "shape: " << values.shape() << '\n';

    return 0;
}
