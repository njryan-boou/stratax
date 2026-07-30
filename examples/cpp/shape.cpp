#include <iostream>

#include <stratax.h>

int main()
{
    // Shape stores dimensions only. It does not own array values.
    stratax::Shape image_shape{3, 224, 224};
    stratax::Shape empty_shape{};
    stratax::Shape zero_sized{4, 0, 2};
    stratax::Shape copied{image_shape};

    std::cout << "image shape: " << image_shape << '\n';
    std::cout << "rank: " << image_shape.rank() << '\n';
    std::cout << "elements: " << image_shape.elements() << '\n';
    std::cout << "first dimension: " << image_shape(0) << '\n';
    std::cout << "last dimension: " << image_shape[-1] << '\n';
    std::cout << "dimensions:";
    for (std::size_t dim : image_shape) {
        std::cout << ' ' << dim;
    }
    std::cout << '\n';
    std::cout << "copied equals original: " << (copied == image_shape) << '\n';

    std::cout << "empty shape: " << empty_shape << '\n';
    std::cout << "zero-sized shape: " << zero_sized
              << " has " << zero_sized.elements() << " elements\n";

    return 0;
}
