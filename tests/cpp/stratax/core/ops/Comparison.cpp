#include <gtest/gtest.h>
#include <complex>

#include <stratax.h>

using namespace stratax::container;

TEST(OpsComparison, vector_equal)
{
    Vector<int> a{1, 2, 3};
    Vector<int> b{1, 2, 3};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, const_vector_equal)
{
    const Vector<int> a{1, 2, 3};
    const Vector<int> b{1, 2, 3};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, empty_vectors_equal)
{
    Vector<int> a(stratax::core::Shape{0});
    Vector<int> b(stratax::core::Shape{0});

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, vector_not_equal_values)
{
    Vector<int> a{1, 2, 3};
    Vector<int> b{1, 9, 3};

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, vector_not_equal_shape)
{
    Vector<int> a{1, 2, 3};
    Vector<int> b{1, 2};

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, complex_vector_equal)
{
    Vector<std::complex<double>> a{
        {1.0, 2.0},
        {3.0, -4.0}
    };
    Vector<std::complex<double>> b{
        {1.0, 2.0},
        {3.0, -4.0}
    };

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));

    b[1] = {3.0, 4.0};

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, matrix_equal)
{
    Matrix<int> a{
        {1, 2},
        {3, 4}
    };
    Matrix<int> b{
        {1, 2},
        {3, 4}
    };

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, matrix_not_equal_values)
{
    Matrix<int> a{
        {1, 2},
        {3, 4}
    };
    Matrix<int> b{
        {1, 2},
        {3, 9}
    };

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, matrix_not_equal_shape)
{
    Matrix<int> a{
        {1, 2},
        {3, 4}
    };
    Matrix<int> b{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, zero_dimension_matrices_equal)
{
    Matrix<int> a(0, 3);
    Matrix<int> b(0, 3);
    Matrix<int> c(0, 4);

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
    EXPECT_TRUE(!(a == c));
    EXPECT_TRUE(a != c);
}

TEST(OpsComparison, tensor_equal)
{
    Tensor<int> a(stratax::core::Shape{2, 2});
    Tensor<int> b(stratax::core::Shape{2, 2});

    a(0) = 1;
    a(1) = 2;
    a(2) = 3;
    a(3) = 4;

    b(0) = 1;
    b(1) = 2;
    b(2) = 3;
    b(3) = 4;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, tensor_not_equal_values)
{
    Tensor<int> a(stratax::core::Shape{2, 2}, 1);
    Tensor<int> b(stratax::core::Shape{2, 2}, 1);

    b(3) = 9;

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, tensor_not_equal_shape)
{
    Tensor<int> a(stratax::core::Shape{2, 2}, 1);
    Tensor<int> b(stratax::core::Shape{4}, 1);

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, empty_tensors_equal)
{
    Tensor<int> a(stratax::core::Shape{});
    Tensor<int> b(stratax::core::Shape{});

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, zero_dimension_tensors_equal)
{
    Tensor<int> a(stratax::core::Shape{2, 0, 3});
    Tensor<int> b(stratax::core::Shape{2, 0, 3});
    Tensor<int> c(stratax::core::Shape{2, 0, 4});

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
    EXPECT_TRUE(!(a == c));
    EXPECT_TRUE(a != c);
}

