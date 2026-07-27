#include <iostream>

#include <stratax.h>

int main()
{
    using stratax::core::Shape;
    using stratax::core::Slice;
    using stratax::container::Matrix;

    Matrix<double> matrix{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Matrix<double> bias(2, 3, 0.5);

    matrix(1, 2) = 9.0;

    auto shifted = matrix + bias;
    auto doubled = matrix * 2.0;
    auto stepped = slice(matrix, Slice{0, matrix.rows(), 1}, Slice{0, matrix.cols(), 2});
    auto reshaped = reshape(matrix, Shape{3, 2});
    auto flattened = flatten(matrix);
    auto roundtrip = to_matrix(reshaped);
    auto row_sums = reduction::sum(matrix, 1);
    auto col_sums_keepdims = reduction::sum(matrix, 0, true);

    Matrix<int> bits{{3, 5, 7}, {8, 10, 12}};
    auto bitmask = bits & 2;
    auto bitshift = bits >> 1;

    std::cout << "matrix:\n" << matrix << '\n';
    std::cout << "shifted:\n" << shifted << '\n';
    std::cout << "doubled:\n" << doubled << '\n';
    std::cout << "stepped slice [:, ::2]:\n" << stepped << '\n';
    std::cout << "reshaped tensor:\n" << reshaped << '\n';
    std::cout << "flattened vector: " << flattened << '\n';
    std::cout << "roundtrip to matrix:\n" << roundtrip << '\n';
    std::cout << "row sums (axis=1): " << row_sums << '\n';
    std::cout << "column sums keepdims (axis=0):\n" << col_sums_keepdims << '\n';
    std::cout << "bits:\n" << bits << '\n';
    std::cout << "bits & 2:\n" << bitmask << '\n';
    std::cout << "bits >> 1:\n" << bitshift << '\n';
    std::cout << "rows: " << matrix.rows() << ", cols: " << matrix.cols() << '\n';
    std::cout << "linear index 2: " << matrix[2] << '\n';

    return 0;
}
