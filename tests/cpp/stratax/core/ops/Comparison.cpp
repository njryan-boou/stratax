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

TEST(OpsComparison, float_vector_equal)
{
    Vector<float> a{1.5f, 2.5f, 3.5f};
    Vector<float> b{1.5f, 2.5f, 3.5f};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, float_vector_not_equal)
{
    Vector<float> a{1.5f, 2.5f, 3.5f};
    Vector<float> b{1.5f, 2.6f, 3.5f};

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, double_vector_equal)
{
    Vector<double> a{1.123456, 2.789012, 3.456789};
    Vector<double> b{1.123456, 2.789012, 3.456789};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, long_vector_equal)
{
    Vector<long> a{1000000L, 2000000L, 3000000L};
    Vector<long> b{1000000L, 2000000L, 3000000L};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, unsigned_int_vector_equal)
{
    Vector<unsigned int> a{1u, 2u, 3u};
    Vector<unsigned int> b{1u, 2u, 3u};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, negative_vector_equal)
{
    Vector<int> a{-1, -2, -3};
    Vector<int> b{-1, -2, -3};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, mixed_sign_vector_equal)
{
    Vector<int> a{-5, 0, 5, -10, 10};
    Vector<int> b{-5, 0, 5, -10, 10};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, single_element_vector_equal)
{
    Vector<int> a{42};
    Vector<int> b{42};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, single_element_vector_not_equal)
{
    Vector<int> a{42};
    Vector<int> b{43};

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, large_vector_equal)
{
    Vector<int> a(stratax::core::Shape{100});
    Vector<int> b(stratax::core::Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        a[i] = static_cast<int>(i);
        b[i] = static_cast<int>(i);
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, large_vector_not_equal)
{
    Vector<int> a(stratax::core::Shape{100});
    Vector<int> b(stratax::core::Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        a[i] = static_cast<int>(i);
        b[i] = static_cast<int>(i);
    }

    b[50] = 999;

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, float_matrix_equal)
{
    Matrix<float> a{
        {1.5f, 2.5f},
        {3.5f, 4.5f}
    };
    Matrix<float> b{
        {1.5f, 2.5f},
        {3.5f, 4.5f}
    };

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, all_zero_matrix_equal)
{
    Matrix<int> a(5, 5);
    Matrix<int> b(5, 5);

    for (std::size_t i = 0; i < 25; ++i) {
        a[i] = 0;
        b[i] = 0;
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, all_negative_matrix_equal)
{
    Matrix<int> a{
        {-1, -2, -3},
        {-4, -5, -6}
    };
    Matrix<int> b{
        {-1, -2, -3},
        {-4, -5, -6}
    };

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, single_element_matrix_equal)
{
    Matrix<int> a(1, 1);
    Matrix<int> b(1, 1);

    a[0] = 99;
    b[0] = 99;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, single_row_matrix_equal)
{
    Matrix<int> a(1, 5);
    Matrix<int> b(1, 5);

    for (std::size_t i = 0; i < 5; ++i) {
        a[i] = static_cast<int>(i);
        b[i] = static_cast<int>(i);
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, single_column_matrix_equal)
{
    Matrix<int> a(5, 1);
    Matrix<int> b(5, 1);

    for (std::size_t i = 0; i < 5; ++i) {
        a[i] = static_cast<int>(i);
        b[i] = static_cast<int>(i);
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, rectangular_matrix_3x2_equal)
{
    Matrix<int> a(3, 2);
    Matrix<int> b(3, 2);

    int val = 1;
    for (std::size_t i = 0; i < 6; ++i) {
        a[i] = val;
        b[i] = val;
        ++val;
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, rectangular_matrix_2x3_not_equal)
{
    Matrix<int> a(2, 3);
    Matrix<int> b(2, 3);

    int val = 1;
    for (std::size_t i = 0; i < 6; ++i) {
        a[i] = val;
        b[i] = val;
        ++val;
    }

    b[3] = 999;

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, large_matrix_equal)
{
    Matrix<int> a(50, 50);
    Matrix<int> b(50, 50);

    for (std::size_t i = 0; i < 2500; ++i) {
        a[i] = static_cast<int>(i % 256);
        b[i] = static_cast<int>(i % 256);
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, large_matrix_not_equal)
{
    Matrix<int> a(50, 50);
    Matrix<int> b(50, 50);

    for (std::size_t i = 0; i < 2500; ++i) {
        a[i] = static_cast<int>(i);
        b[i] = static_cast<int>(i);
    }

    b[1250] = 999999;

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, high_rank_tensor_4d_equal)
{
    Tensor<int> a(stratax::core::Shape{2, 3, 2, 3});
    Tensor<int> b(stratax::core::Shape{2, 3, 2, 3});

    int val = 1;
    for (std::size_t i = 0; i < 36; ++i) {
        a(i) = val;
        b(i) = val;
        ++val;
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, high_rank_tensor_5d_equal)
{
    Tensor<int> a(stratax::core::Shape{2, 2, 2, 2, 2});
    Tensor<int> b(stratax::core::Shape{2, 2, 2, 2, 2});

    int val = 1;
    for (std::size_t i = 0; i < 32; ++i) {
        a(i) = val;
        b(i) = val;
        ++val;
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, high_rank_tensor_5d_not_equal)
{
    Tensor<int> a(stratax::core::Shape{2, 2, 2, 2, 2});
    Tensor<int> b(stratax::core::Shape{2, 2, 2, 2, 2});

    int val = 1;
    for (std::size_t i = 0; i < 32; ++i) {
        a(i) = val;
        b(i) = val;
        ++val;
    }

    b(15) = 999;

    EXPECT_TRUE(!(a == b));
    EXPECT_TRUE(a != b);
}

TEST(OpsComparison, double_matrix_equal)
{
    Matrix<double> a{
        {1.123, 2.456},
        {3.789, 4.012}
    };
    Matrix<double> b{
        {1.123, 2.456},
        {3.789, 4.012}
    };

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, double_tensor_equal)
{
    Tensor<double> a(stratax::core::Shape{2, 2, 2});
    Tensor<double> b(stratax::core::Shape{2, 2, 2});

    for (std::size_t i = 0; i < 8; ++i) {
        a(i) = 1.5 * static_cast<double>(i);
        b(i) = 1.5 * static_cast<double>(i);
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, vector_self_comparison)
{
    Vector<int> a{1, 2, 3, 4, 5};

    EXPECT_TRUE(a == a);
    EXPECT_TRUE(!(a != a));
}

TEST(OpsComparison, matrix_self_comparison)
{
    Matrix<int> a{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_TRUE(a == a);
    EXPECT_TRUE(!(a != a));
}

TEST(OpsComparison, tensor_self_comparison)
{
    Tensor<int> a(stratax::core::Shape{2, 3, 4});

    for (std::size_t i = 0; i < 24; ++i) {
        a(i) = static_cast<int>(i);
    }

    EXPECT_TRUE(a == a);
    EXPECT_TRUE(!(a != a));
}

TEST(OpsComparison, vector_copy_comparison)
{
    Vector<int> a{1, 2, 3};
    Vector<int> b = a;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, matrix_copy_comparison)
{
    Matrix<int> a{
        {1, 2},
        {3, 4}
    };
    Matrix<int> b = a;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, tensor_copy_comparison)
{
    Tensor<int> a(stratax::core::Shape{3, 3}, 5);
    Tensor<int> b = a;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, asymmetric_tensor_2d_equal)
{
    Tensor<int> a(stratax::core::Shape{1, 5});
    Tensor<int> b(stratax::core::Shape{1, 5});

    for (std::size_t i = 0; i < 5; ++i) {
        a(i) = static_cast<int>(i);
        b(i) = static_cast<int>(i);
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, asymmetric_tensor_3d_equal)
{
    Tensor<int> a(stratax::core::Shape{1, 2, 3});
    Tensor<int> b(stratax::core::Shape{1, 2, 3});

    for (std::size_t i = 0; i < 6; ++i) {
        a(i) = static_cast<int>(i);
        b(i) = static_cast<int>(i);
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, const_matrix_equal)
{
    const Matrix<int> a{
        {1, 2, 3},
        {4, 5, 6}
    };
    const Matrix<int> b{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, const_tensor_equal)
{
    const Tensor<int> a(stratax::core::Shape{2, 2, 2}, 7);
    const Tensor<int> b(stratax::core::Shape{2, 2, 2}, 7);

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, mixed_const_mutable_vector_equal)
{
    const Vector<int> a{10, 20, 30};
    Vector<int> b{10, 20, 30};

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

TEST(OpsComparison, zero_filled_large_vector)
{
    Vector<int> a(stratax::core::Shape{100});
    Vector<int> b(stratax::core::Shape{100});

    for (std::size_t i = 0; i < 100; ++i) {
        a[i] = 0;
        b[i] = 0;
    }

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
}

