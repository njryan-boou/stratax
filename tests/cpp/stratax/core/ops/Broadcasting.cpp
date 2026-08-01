#include <gtest/gtest.h>

#include <functional>

#include <stratax.h>

using namespace stratax::container;

TEST(OpsBroadcasting, validates_and_computes_broadcasted_shape)
{
    const stratax::core::Shape column{3, 1};
    const stratax::core::Shape row{1, 4};

    EXPECT_TRUE(validate_broadcast(column, row));
    EXPECT_TRUE(broadcasted_shape(column, row) == stratax::core::Shape({3, 4}));
    EXPECT_FALSE(validate_broadcast(
        stratax::core::Shape{2, 3},
        stratax::core::Shape{3, 2}));
}

TEST(OpsBroadcasting, matrix_arithmetic_broadcasts_row_and_column)
{
    Matrix<int> column{{1}, {2}};
    Matrix<int> row{{10, 20, 30}};

    Matrix<int> sum = column + row;

    EXPECT_TRUE(sum.shape() == stratax::core::Shape({2, 3}));
    EXPECT_TRUE(sum[0] == 11);
    EXPECT_TRUE(sum[2] == 31);
    EXPECT_TRUE(sum[3] == 12);
    EXPECT_TRUE(sum[5] == 32);
}

TEST(OpsBroadcasting, tensor_arithmetic_broadcasts_across_leading_axes)
{
    Tensor<int> values(stratax::core::Shape{2, 3, 4}, 20);
    Tensor<int> divisors(stratax::core::Shape{4});
    divisors[0] = 1;
    divisors[1] = 2;
    divisors[2] = 4;
    divisors[3] = 5;

    Tensor<int> quotient = values / divisors;

    EXPECT_TRUE(quotient.shape() == values.shape());
    EXPECT_TRUE(quotient[0] == 20);
    EXPECT_TRUE(quotient[1] == 10);
    EXPECT_TRUE(quotient[2] == 5);
    EXPECT_TRUE(quotient[3] == 4);
    EXPECT_TRUE(quotient[23] == 4);
}

TEST(OpsBroadcasting, direct_operation_supports_scalars_on_either_side)
{
    Vector<int> values{1, 2, 3};

    Vector<int> sum = broadcasted_op(values, 10, std::plus<>{});
    Vector<int> difference = broadcasted_op(10, values, std::minus<>{});

    EXPECT_TRUE(sum[2] == 13);
    EXPECT_TRUE(difference[2] == 7);
}

TEST(OpsBroadcasting, incompatible_shapes_throw)
{
    Vector<int> short_vector{1, 2};
    Vector<int> long_vector{1, 2, 3};
    EXPECT_THROW(short_vector + long_vector, Exceptions::BroadcastError);

    Matrix<int> wide{{1, 2, 3}, {4, 5, 6}};
    Matrix<int> tall{{1, 2}, {3, 4}, {5, 6}};
    EXPECT_THROW(wide - tall, Exceptions::BroadcastError);

    Tensor<int> flat(stratax::core::Shape{4}, 1);
    Tensor<int> square(stratax::core::Shape{2, 2}, 1);
    EXPECT_THROW(flat * square, Exceptions::BroadcastError);
}

TEST(OpsBroadcasting, failed_compound_operation_preserves_left_operand)
{
    Vector<int> lhs{1, 2};
    Vector<int> rhs{1, 2, 3};

    EXPECT_THROW(lhs += rhs, Exceptions::BroadcastError);
    EXPECT_TRUE(lhs.size() == 2);
    EXPECT_TRUE(lhs[0] == 1);
    EXPECT_TRUE(lhs[1] == 2);
}
