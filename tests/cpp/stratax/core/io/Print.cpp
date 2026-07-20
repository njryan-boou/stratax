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

// ============================================================================
// Numeric Type Coverage
// ============================================================================

TEST(IoPrint, vector_prints_double_values)
{
    Vector<double> vector(Shape{3});

    vector(0) = 1.5;
    vector(1) = 2.5;
    vector(2) = 3.5;

    std::ostringstream out;
    out << vector;

    std::string result = out.str();
    EXPECT_TRUE(result.find("1.5") != std::string::npos);
    EXPECT_TRUE(result.find("2.5") != std::string::npos);
    EXPECT_TRUE(result.find("3.5") != std::string::npos);
}

TEST(IoPrint, vector_prints_float_values)
{
    Vector<float> vector(Shape{2});

    vector(0) = 1.1f;
    vector(1) = 2.2f;

    std::ostringstream out;
    out << vector;

    std::string result = out.str();
    EXPECT_TRUE(result.find('[') != std::string::npos);
    EXPECT_TRUE(result.find(']') != std::string::npos);
}

TEST(IoPrint, matrix_prints_double_values)
{
    Matrix<double> matrix(2, 2);

    matrix(0, 0) = 1.1;
    matrix(0, 1) = 2.2;
    matrix(1, 0) = 3.3;
    matrix(1, 1) = 4.4;

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result.find("1.1") != std::string::npos);
    EXPECT_TRUE(result.find("[") != std::string::npos);
}

TEST(IoPrint, tensor_prints_negative_values)
{
    Tensor<int> tensor(Shape{2, 2});

    tensor(0) = -1;
    tensor(1) = -2;
    tensor(2) = -3;
    tensor(3) = -4;

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    EXPECT_TRUE(result.find("-1") != std::string::npos);
    EXPECT_TRUE(result.find("-4") != std::string::npos);
}

TEST(IoPrint, tensor_prints_zero_values)
{
    Tensor<int> tensor(Shape{3});

    tensor(0) = 0;
    tensor(1) = 0;
    tensor(2) = 0;

    std::ostringstream out;
    out << tensor;

    EXPECT_TRUE(out.str() == "[0, 0, 0]");
}

// ============================================================================
// Edge Cases: Container Dimensions
// ============================================================================

TEST(IoPrint, vector_single_element)
{
    Vector<int> vector(Shape{1});

    vector(0) = 42;

    std::ostringstream out;
    out << vector;

    EXPECT_TRUE(out.str() == "[42]");
}

TEST(IoPrint, vector_large_size)
{
    Vector<int> vector(Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        vector(i) = static_cast<int>(i);
    }

    std::ostringstream out;
    out << vector;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
    EXPECT_TRUE(result.find("]") != std::string::npos);
    EXPECT_TRUE(result.find("0") != std::string::npos);
    EXPECT_TRUE(result.find("99") != std::string::npos);
}

TEST(IoPrint, matrix_single_element)
{
    Matrix<int> matrix(1, 1);

    matrix(0, 0) = 99;

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result.find("99") != std::string::npos);
}

TEST(IoPrint, matrix_single_row)
{
    Matrix<int> matrix(1, 5);

    for (std::size_t i = 0; i < 5; ++i) {
        matrix(0, i) = static_cast<int>(i + 1);
    }

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
    EXPECT_TRUE(result.find("1") != std::string::npos);
    EXPECT_TRUE(result.find("5") != std::string::npos);
}

TEST(IoPrint, matrix_single_column)
{
    Matrix<int> matrix(5, 1);

    for (std::size_t i = 0; i < 5; ++i) {
        matrix(i, 0) = static_cast<int>(i + 10);
    }

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result.find("10") != std::string::npos);
    EXPECT_TRUE(result.find("14") != std::string::npos);
}

TEST(IoPrint, matrix_wide)
{
    Matrix<int> matrix(2, 5);

    for (std::size_t i = 0; i < 10; ++i) {
        matrix(i / 5, i % 5) = static_cast<int>(i);
    }

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
}

TEST(IoPrint, matrix_tall)
{
    Matrix<int> matrix(5, 2);

    for (std::size_t i = 0; i < 10; ++i) {
        matrix(i / 2, i % 2) = static_cast<int>(i);
    }

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
}

// ============================================================================
// High-Rank Tensors
// ============================================================================

TEST(IoPrint, tensor_rank_four)
{
    Tensor<int> tensor(Shape{2, 2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i);
    }

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
    EXPECT_TRUE(result.find("]") != std::string::npos);
}

TEST(IoPrint, tensor_high_rank)
{
    Tensor<int> tensor(Shape{2, 2, 2, 2, 2});

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor(i) = static_cast<int>(i % 10);
    }

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
    EXPECT_TRUE(result.find("]") != std::string::npos);
}

// ============================================================================
// Format Verification
// ============================================================================

TEST(IoPrint, vector_contains_commas)
{
    Vector<int> vector(Shape{3});

    vector(0) = 1;
    vector(1) = 2;
    vector(2) = 3;

    std::ostringstream out;
    out << vector;

    std::string result = out.str();
    EXPECT_TRUE(result.find(",") != std::string::npos);
}

TEST(IoPrint, matrix_contains_newlines)
{
    Matrix<int> matrix(2, 2);

    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(1, 0) = 3;
    matrix(1, 1) = 4;

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result.find("\n") != std::string::npos);
}

TEST(IoPrint, matrix_starts_with_bracket)
{
    Matrix<int> matrix(2, 2);

    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(1, 0) = 3;
    matrix(1, 1) = 4;

    std::ostringstream out;
    out << matrix;

    EXPECT_TRUE(out.str()[0] == '[');
}

TEST(IoPrint, matrix_ends_with_bracket)
{
    Matrix<int> matrix(2, 2);

    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(1, 0) = 3;
    matrix(1, 1) = 4;

    std::ostringstream out;
    out << matrix;

    std::string result = out.str();
    EXPECT_TRUE(result[result.length() - 1] == ']');
}

TEST(IoPrint, tensor_bracket_balance)
{
    Tensor<int> tensor(Shape{2, 2});

    tensor(0) = 1;
    tensor(1) = 2;
    tensor(2) = 3;
    tensor(3) = 4;

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    int open_count = 0;
    int close_count = 0;

    for (char c : result) {
        if (c == '[') open_count++;
        if (c == ']') close_count++;
    }

    EXPECT_TRUE(open_count == close_count);
    EXPECT_TRUE(open_count > 0);
}

// ============================================================================
// Special Value Ranges
// ============================================================================

TEST(IoPrint, vector_prints_max_int)
{
    Vector<int> vector(Shape{1});

    vector(0) = 2147483647;

    std::ostringstream out;
    out << vector;

    std::string result = out.str();
    EXPECT_TRUE(result.find("2147483647") != std::string::npos);
}

TEST(IoPrint, vector_prints_negative_max_int)
{
    Vector<int> vector(Shape{1});

    vector(0) = -2147483648;

    std::ostringstream out;
    out << vector;

    std::string result = out.str();
    EXPECT_TRUE(result.find("-") != std::string::npos);
}

TEST(IoPrint, tensor_mixed_positive_negative)
{
    Tensor<int> tensor(Shape{2, 2});

    tensor(0) = 100;
    tensor(1) = -50;
    tensor(2) = 0;
    tensor(3) = -200;

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    EXPECT_TRUE(result.find("100") != std::string::npos);
    EXPECT_TRUE(result.find("-50") != std::string::npos);
    EXPECT_TRUE(result.find("0") != std::string::npos);
    EXPECT_TRUE(result.find("-200") != std::string::npos);
}

// ============================================================================
// Output Stream Integration
// ============================================================================

TEST(IoPrint, multiple_vectors_in_sequence)
{
    Vector<int> v1(Shape{2});
    Vector<int> v2(Shape{2});

    v1(0) = 1;
    v1(1) = 2;
    v2(0) = 3;
    v2(1) = 4;

    std::ostringstream out;
    out << v1 << " " << v2;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[1, 2]") != std::string::npos);
    EXPECT_TRUE(result.find("[3, 4]") != std::string::npos);
}

TEST(IoPrint, tensor_output_length)
{
    Tensor<int> tensor(Shape{3, 3});

    for (std::size_t i = 0; i < 9; ++i) {
        tensor(i) = static_cast<int>(i);
    }

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    EXPECT_TRUE(result.length() > 0);
}

// ============================================================================
// Asymmetric Tensors
// ============================================================================

TEST(IoPrint, tensor_1x2x3)
{
    Tensor<int> tensor(Shape{1, 2, 3});

    for (std::size_t i = 0; i < 6; ++i) {
        tensor(i) = static_cast<int>(i);
    }

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
    EXPECT_TRUE(result.find("0") != std::string::npos);
}

TEST(IoPrint, tensor_3x1x2)
{
    Tensor<int> tensor(Shape{3, 1, 2});

    for (std::size_t i = 0; i < 6; ++i) {
        tensor(i) = static_cast<int>(i);
    }

    std::ostringstream out;
    out << tensor;

    std::string result = out.str();
    EXPECT_TRUE(result.find("[") != std::string::npos);
}

