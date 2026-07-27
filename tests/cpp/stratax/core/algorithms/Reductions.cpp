#include <gtest/gtest.h>
#include <cmath>

#include <stratax.h>

using namespace stratax::container;
using stratax::core::Shape;

// ============================================================================
// Global Reduction Tests
// ============================================================================

TEST(AlgorithmsReductions, vector_sum)
{
    Vector<int> v{1, 2, 3, 4, 5};
    int result = reduction::sum(v);
    EXPECT_EQ(result, 15);
}

TEST(AlgorithmsReductions, vector_prod)
{
    Vector<int> v{1, 2, 3, 4};
    int result = reduction::prod(v);
    EXPECT_EQ(result, 24);
}

TEST(AlgorithmsReductions, vector_max)
{
    Vector<int> v{3, 1, 4, 1, 5};
    int result = reduction::max(v);
    EXPECT_EQ(result, 5);
}

TEST(AlgorithmsReductions, vector_min)
{
    Vector<int> v{3, 1, 4, 1, 5};
    int result = reduction::min(v);
    EXPECT_EQ(result, 1);
}

TEST(AlgorithmsReductions, vector_argmax)
{
    Vector<int> v{3, 1, 4, 1, 5};
    std::size_t idx = reduction::argmax(v);
    EXPECT_EQ(idx, 4);
    EXPECT_EQ(v[idx], 5);
}

TEST(AlgorithmsReductions, vector_argmin)
{
    Vector<int> v{3, 1, 4, 1, 5};
    std::size_t idx = reduction::argmin(v);
    EXPECT_EQ(idx, 1);
    EXPECT_EQ(v[idx], 1);
}

TEST(AlgorithmsReductions, vector_mean)
{
    Vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0};
    double result = reduction::mean(v);
    EXPECT_DOUBLE_EQ(result, 3.0);
}

TEST(AlgorithmsReductions, matrix_sum)
{
    Matrix<int> m{
        {1, 2, 3},
        {4, 5, 6}
    };
    int result = reduction::sum(m);
    EXPECT_EQ(result, 21);
}

TEST(AlgorithmsReductions, matrix_prod)
{
    Matrix<int> m{
        {1, 2},
        {3, 4}
    };
    int result = reduction::prod(m);
    EXPECT_EQ(result, 24);
}

TEST(AlgorithmsReductions, matrix_max)
{
    Matrix<int> m{
        {1, 5, 3},
        {2, 4, 6}
    };
    int result = reduction::max(m);
    EXPECT_EQ(result, 6);
}

TEST(AlgorithmsReductions, matrix_min)
{
    Matrix<int> m{
        {1, 5, 3},
        {2, 4, 6}
    };
    int result = reduction::min(m);
    EXPECT_EQ(result, 1);
}

TEST(AlgorithmsReductions, tensor_sum)
{
    Tensor<int> t(Shape{2, 3, 2});
    for (std::size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<int>(i + 1);
    }
    int result = reduction::sum(t);
    EXPECT_EQ(result, 78);  // 1+2+...+12
}

TEST(AlgorithmsReductions, tensor_max)
{
    Tensor<double> t(Shape{2, 3});
    t[0] = 1.5;
    t[1] = 3.2;
    t[2] = 2.1;
    t[3] = 4.5;
    t[4] = 2.8;
    t[5] = 1.0;
    
    double result = reduction::max(t);
    EXPECT_DOUBLE_EQ(result, 4.5);
}

// ============================================================================
// Axis Reduction Tests - Vector
// ============================================================================

// Note: Skipping rank-0 reduction test for vectors (reducing on only axis).
// This would return a rank-0 tensor, which has limited support in the current API.
// Future work: Add proper rank-0 tensor support.

// ============================================================================
// Axis Reduction Tests - Matrix
// ============================================================================

TEST(AlgorithmsReductions, matrix_sum_axis_0)
{
    Matrix<int> m{
        {1, 2, 3},
        {4, 5, 6}
    };
    auto result = reduction::sum(m, 0);
    
    EXPECT_EQ(result.rank(), 1);
    EXPECT_EQ(result.shape()(0), 3);
    EXPECT_EQ(result[0], 5);   // 1 + 4
    EXPECT_EQ(result[1], 7);   // 2 + 5
    EXPECT_EQ(result[2], 9);   // 3 + 6
}

TEST(AlgorithmsReductions, matrix_sum_axis_1)
{
    Matrix<int> m{
        {1, 2, 3},
        {4, 5, 6}
    };
    auto result = reduction::sum(m, 1);
    
    EXPECT_EQ(result.rank(), 1);
    EXPECT_EQ(result.shape()(0), 2);
    EXPECT_EQ(result[0], 6);   // 1 + 2 + 3
    EXPECT_EQ(result[1], 15);  // 4 + 5 + 6
}

TEST(AlgorithmsReductions, matrix_prod_axis_0)
{
    Matrix<int> m{
        {1, 2, 3},
        {2, 3, 4}
    };
    auto result = reduction::prod(m, 0);
    
    EXPECT_EQ(result[0], 2);   // 1 * 2
    EXPECT_EQ(result[1], 6);   // 2 * 3
    EXPECT_EQ(result[2], 12);  // 3 * 4
}

TEST(AlgorithmsReductions, matrix_max_axis_0)
{
    Matrix<int> m{
        {1, 5, 3},
        {4, 2, 6}
    };
    auto result = reduction::max(m, 0);
    
    EXPECT_EQ(result[0], 4);   // max(1, 4)
    EXPECT_EQ(result[1], 5);   // max(5, 2)
    EXPECT_EQ(result[2], 6);   // max(3, 6)
}

TEST(AlgorithmsReductions, matrix_min_axis_1)
{
    Matrix<int> m{
        {3, 1, 4},
        {2, 7, 1}
    };
    auto result = reduction::min(m, 1);
    
    EXPECT_EQ(result[0], 1);   // min(3, 1, 4)
    EXPECT_EQ(result[1], 1);   // min(2, 7, 1)
}

TEST(AlgorithmsReductions, matrix_argmax_axis_1)
{
    Matrix<int> m{
        {3, 1, 4},
        {2, 7, 1}
    };
    auto result = reduction::argmax(m, 1);
    
    EXPECT_EQ(result[0], 2);   // argmax(3, 1, 4) = 2
    EXPECT_EQ(result[1], 1);   // argmax(2, 7, 1) = 1
}

TEST(AlgorithmsReductions, matrix_argmin_axis_0)
{
    Matrix<int> m{
        {3, 1, 4},
        {2, 7, 1}
    };
    auto result = reduction::argmin(m, 0);
    
    EXPECT_EQ(result[0], 1);   // argmin(3, 2) = 1
    EXPECT_EQ(result[1], 0);   // argmin(1, 7) = 0
    EXPECT_EQ(result[2], 1);   // argmin(4, 1) = 1
}

// ============================================================================
// Axis Reduction Tests - Tensor (3D)
// ============================================================================

TEST(AlgorithmsReductions, tensor_sum_axis_0)
{
    Tensor<int> t(Shape{2, 2, 3});
    for (std::size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<int>(i + 1);
    }
    
    auto result = reduction::sum(t, 0);
    
    EXPECT_EQ(result.rank(), 2);
    EXPECT_EQ(result.shape()(0), 2);
    EXPECT_EQ(result.shape()(1), 3);
    
    // First row of result: sum over first dimension
    EXPECT_EQ(result(0, 0), 8);   // 1 + 7
    EXPECT_EQ(result(0, 1), 10);  // 2 + 8
}

TEST(AlgorithmsReductions, tensor_sum_axis_1)
{
    Tensor<int> t(Shape{2, 2, 3});
    for (std::size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<int>(i + 1);
    }
    
    auto result = reduction::sum(t, 1);
    
    EXPECT_EQ(result.rank(), 2);
    EXPECT_EQ(result.shape()(0), 2);
    EXPECT_EQ(result.shape()(1), 3);
}

TEST(AlgorithmsReductions, tensor_sum_axis_2)
{
    Tensor<int> t(Shape{2, 2, 3});
    for (std::size_t i = 0; i < t.size(); ++i) {
        t[i] = static_cast<int>(i + 1);
    }
    
    auto result = reduction::sum(t, 2);
    
    EXPECT_EQ(result.rank(), 2);
    EXPECT_EQ(result.shape()(0), 2);
    EXPECT_EQ(result.shape()(1), 2);
    
    EXPECT_EQ(result(0, 0), 6);   // 1 + 2 + 3
    EXPECT_EQ(result(0, 1), 15);  // 4 + 5 + 6
}

TEST(AlgorithmsReductions, tensor_max_axis_0)
{
    Tensor<int> t(Shape{2, 2, 2});
    t[0] = 1; t[1] = 2; t[2] = 3; t[3] = 4;
    t[4] = 5; t[5] = 6; t[6] = 7; t[7] = 8;
    
    auto result = reduction::max(t, 0);
    
    EXPECT_EQ(result.shape()(0), 2);
    EXPECT_EQ(result.shape()(1), 2);
    EXPECT_EQ(result(0, 0), 5);   // max(1, 5)
    EXPECT_EQ(result(0, 1), 6);   // max(2, 6)
    EXPECT_EQ(result(1, 0), 7);   // max(3, 7)
    EXPECT_EQ(result(1, 1), 8);   // max(4, 8)
}

// ============================================================================
// Double Precision Tests
// ============================================================================

TEST(AlgorithmsReductions, vector_mean_double)
{
    Vector<double> v{1.5, 2.5, 3.5, 4.5};
    double result = reduction::mean(v);
    EXPECT_DOUBLE_EQ(result, 3.0);
}

TEST(AlgorithmsReductions, matrix_mean_axis_0)
{
    Matrix<double> m{
        {1.0, 2.0, 3.0},
        {5.0, 6.0, 7.0}
    };
    auto result = reduction::mean(m, 0);
    
    EXPECT_DOUBLE_EQ(result[0], 3.0);   // mean(1, 5)
    EXPECT_DOUBLE_EQ(result[1], 4.0);   // mean(2, 6)
    EXPECT_DOUBLE_EQ(result[2], 5.0);   // mean(3, 7)
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(AlgorithmsReductions, single_element_vector)
{
    Vector<int> v{42};
    
    EXPECT_EQ(reduction::sum(v), 42);
    EXPECT_EQ(reduction::prod(v), 42);
    EXPECT_EQ(reduction::max(v), 42);
    EXPECT_EQ(reduction::min(v), 42);
    EXPECT_EQ(reduction::argmax(v), 0);
    EXPECT_EQ(reduction::argmin(v), 0);
}

TEST(AlgorithmsReductions, negative_values)
{
    Vector<int> v{-5, -2, -8, -1};
    
    EXPECT_EQ(reduction::sum(v), -16);
    EXPECT_EQ(reduction::max(v), -1);
    EXPECT_EQ(reduction::min(v), -8);
    EXPECT_EQ(reduction::argmax(v), 3);
    EXPECT_EQ(reduction::argmin(v), 2);
}

TEST(AlgorithmsReductions, matrix_single_row)
{
    Matrix<int> m{{1, 2, 3}};
    
    auto result_axis_0 = reduction::sum(m, 0);
    EXPECT_EQ(result_axis_0.rank(), 1);
    EXPECT_EQ(result_axis_0[0], 1);
    EXPECT_EQ(result_axis_0[1], 2);
    EXPECT_EQ(result_axis_0[2], 3);
    
    auto result_axis_1 = reduction::sum(m, 1);
    EXPECT_EQ(result_axis_1.rank(), 1);
    EXPECT_EQ(result_axis_1[0], 6);
}

TEST(AlgorithmsReductions, matrix_single_column)
{
    Matrix<int> m{
        {1},
        {2},
        {3}
    };
    
    auto result_axis_0 = reduction::sum(m, 0);
    EXPECT_EQ(result_axis_0[0], 6);
    
    auto result_axis_1 = reduction::sum(m, 1);
    EXPECT_EQ(result_axis_1.rank(), 1);
    EXPECT_EQ(result_axis_1[0], 1);
    EXPECT_EQ(result_axis_1[1], 2);
    EXPECT_EQ(result_axis_1[2], 3);
}
