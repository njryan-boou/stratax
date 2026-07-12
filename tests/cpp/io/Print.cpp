#include <gtest/gtest.h>
#include <complex>
#include <sstream>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(IoPrint, shape_prints_empty)
{
    Shape shape;

    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "()");
}

TEST(IoPrint, shape_prints_rank_one_with_trailing_comma)
{
    Shape shape{3};

    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "(3,)");
}

TEST(IoPrint, shape_prints_multiple_dimensions)
{
    Shape shape{2, 3, 4};

    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "(2, 3, 4)");
}

TEST(IoPrint, vector_prints_empty)
{
    Vector<int> vector;

    std::ostringstream out;
    out << vector;

    EXPECT_TRUE(out.str() == "[]");
}

TEST(IoPrint, vector_prints_values)
{
    Vector<int> vector{1, 2, 3};

    std::ostringstream out;
    out << vector;

    EXPECT_TRUE(out.str() == "[1, 2, 3]");
}

TEST(IoPrint, vector_prints_complex_values)
{
    Vector<std::complex<double>> vector{
        {1.0, 2.0},
        {3.0, -4.0}
    };

    std::ostringstream out;
    out << vector;

    EXPECT_TRUE(out.str() == "[(1,2), (3,-4)]");
}

TEST(IoPrint, matrix_prints_empty_shape)
{
    Matrix<int> matrix(0, 0);

    std::ostringstream out;
    out << matrix;

    EXPECT_TRUE(out.str() == "[\n\n]");
}

TEST(IoPrint, matrix_prints_zero_columns)
{
    Matrix<int> matrix(2, 0);

    std::ostringstream out;
    out << matrix;

    EXPECT_TRUE(out.str() == "[\n    []\n    []\n]");
}

TEST(IoPrint, matrix_prints_values)
{
    Matrix<int> matrix{
        {1, 2},
        {3, 4}
    };

    std::ostringstream out;
    out << matrix;

    EXPECT_TRUE(out.str() == "[\n    [1, 2]\n    [3, 4]\n]");
}

TEST(IoPrint, tensor_prints_zero_length)
{
    Tensor<int> tensor(Shape{0});

    std::ostringstream out;
    out << tensor;

    EXPECT_TRUE(out.str() == "[]");
}

TEST(IoPrint, tensor_prints_rank_one)
{
    Tensor<int> tensor(Shape{3});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;

    std::ostringstream out;
    out << tensor;

    EXPECT_TRUE(out.str() == "[1, 2, 3]");
}

TEST(IoPrint, tensor_prints_rank_two)
{
    Tensor<int> tensor(Shape{2, 2});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;
    tensor(3) = 4;

    std::ostringstream out;
    out << tensor;

    EXPECT_TRUE(out.str() == "[\n    [1, 2],\n    [3, 4]\n]");
}

TEST(IoPrint, tensor_prints_rank_three)
{
    Tensor<int> tensor(Shape{2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<int>(i + 1);
    }

    std::ostringstream out;
    out << tensor;

    EXPECT_TRUE(out.str() ==
        "[\n"
        "    [\n"
        "        [1, 2],\n"
        "        [3, 4]\n"
        "    ],\n"
        "    [\n"
        "        [5, 6],\n"
        "        [7, 8]\n"
        "    ]\n"
        "]");
}

TEST(IoPrint, tensor_prints_zero_dimension)
{
    Tensor<int> tensor(Shape{2, 0, 3});

    std::ostringstream out;
    out << tensor;

    EXPECT_TRUE(out.str() == "[]");
}

