#include <gtest/gtest.h>

#include <stratax.h>

using namespace stratax::container;
using stratax::core::Shape;

TEST(OpsBitwise, vector_unary_not)
{
    Vector<int> values{0, 1, 2, 7};

    Vector<int> result = ~values;

    EXPECT_EQ(result.size(), values.size());
    EXPECT_EQ(result[0], ~0);
    EXPECT_EQ(result[1], ~1);
    EXPECT_EQ(result[2], ~2);
    EXPECT_EQ(result[3], ~7);
}

TEST(OpsBitwise, vector_array_array_bitwise)
{
    Vector<int> lhs{10, 12, 15};  // 1010, 1100, 1111
    Vector<int> rhs{3, 10, 6};    // 0011, 1010, 0110

    auto and_result = lhs & rhs;
    auto or_result = lhs | rhs;
    auto xor_result = lhs ^ rhs;

    EXPECT_EQ(and_result[0], 2);
    EXPECT_EQ(and_result[1], 8);
    EXPECT_EQ(and_result[2], 6);

    EXPECT_EQ(or_result[0], 11);
    EXPECT_EQ(or_result[1], 14);
    EXPECT_EQ(or_result[2], 15);

    EXPECT_EQ(xor_result[0], 9);
    EXPECT_EQ(xor_result[1], 6);
    EXPECT_EQ(xor_result[2], 9);
}

TEST(OpsBitwise, vector_shift_array_array)
{
    Vector<int> values{1, 8, 12};
    Vector<int> shifts{1, 2, 1};

    auto left_result = values << shifts;
    auto right_result = values >> shifts;

    EXPECT_EQ(left_result[0], 2);
    EXPECT_EQ(left_result[1], 32);
    EXPECT_EQ(left_result[2], 24);

    EXPECT_EQ(right_result[0], 0);
    EXPECT_EQ(right_result[1], 2);
    EXPECT_EQ(right_result[2], 6);
}

TEST(OpsBitwise, vector_array_scalar_and_reverse_scalar)
{
    Vector<int> values{3, 5, 6};

    auto and_scalar = values & 2;
    auto or_scalar = values | 1;
    auto xor_scalar = values ^ 7;

    auto and_reverse = 2 & values;
    auto or_reverse = 1 | values;
    auto xor_reverse = 7 ^ values;

    EXPECT_EQ(and_scalar[0], 2);
    EXPECT_EQ(and_scalar[1], 0);
    EXPECT_EQ(and_scalar[2], 2);

    EXPECT_EQ(or_scalar[0], 3);
    EXPECT_EQ(or_scalar[1], 5);
    EXPECT_EQ(or_scalar[2], 7);

    EXPECT_EQ(xor_scalar[0], 4);
    EXPECT_EQ(xor_scalar[1], 2);
    EXPECT_EQ(xor_scalar[2], 1);

    EXPECT_EQ(and_reverse[0], and_scalar[0]);
    EXPECT_EQ(and_reverse[1], and_scalar[1]);
    EXPECT_EQ(and_reverse[2], and_scalar[2]);

    EXPECT_EQ(or_reverse[0], or_scalar[0]);
    EXPECT_EQ(or_reverse[1], or_scalar[1]);
    EXPECT_EQ(or_reverse[2], or_scalar[2]);

    EXPECT_EQ(xor_reverse[0], xor_scalar[0]);
    EXPECT_EQ(xor_reverse[1], xor_scalar[1]);
    EXPECT_EQ(xor_reverse[2], xor_scalar[2]);
}

TEST(OpsBitwise, vector_shift_scalar_and_reverse_scalar)
{
    Vector<int> values{1, 2, 3};

    auto left_scalar = values << 2;
    auto right_scalar = values >> 1;

    auto left_reverse = 32 << values;
    auto right_reverse = 32 >> values;

    EXPECT_EQ(left_scalar[0], 4);
    EXPECT_EQ(left_scalar[1], 8);
    EXPECT_EQ(left_scalar[2], 12);

    EXPECT_EQ(right_scalar[0], 0);
    EXPECT_EQ(right_scalar[1], 1);
    EXPECT_EQ(right_scalar[2], 1);

    EXPECT_EQ(left_reverse[0], 64);
    EXPECT_EQ(left_reverse[1], 128);
    EXPECT_EQ(left_reverse[2], 256);

    EXPECT_EQ(right_reverse[0], 16);
    EXPECT_EQ(right_reverse[1], 8);
    EXPECT_EQ(right_reverse[2], 4);
}

TEST(OpsBitwise, matrix_inplace_bitwise_array_and_scalar)
{
    Matrix<int> values{{3, 5}, {6, 8}};
    Matrix<int> mask{{1, 1}, {7, 3}};

    values &= mask;
    EXPECT_EQ(values(0, 0), 1);
    EXPECT_EQ(values(0, 1), 1);
    EXPECT_EQ(values(1, 0), 6);
    EXPECT_EQ(values(1, 1), 0);

    values |= 2;
    EXPECT_EQ(values(0, 0), 3);
    EXPECT_EQ(values(0, 1), 3);
    EXPECT_EQ(values(1, 0), 6);
    EXPECT_EQ(values(1, 1), 2);

    values ^= 1;
    EXPECT_EQ(values(0, 0), 2);
    EXPECT_EQ(values(0, 1), 2);
    EXPECT_EQ(values(1, 0), 7);
    EXPECT_EQ(values(1, 1), 3);
}

TEST(OpsBitwise, tensor_inplace_shift_array_and_scalar)
{
    Tensor<int> values(Shape{2, 2});
    Tensor<int> shifts(Shape{2, 2});

    values[0] = 1;
    values[1] = 2;
    values[2] = 4;
    values[3] = 8;

    shifts[0] = 1;
    shifts[1] = 2;
    shifts[2] = 0;
    shifts[3] = 1;

    values <<= shifts;
    EXPECT_EQ(values[0], 2);
    EXPECT_EQ(values[1], 8);
    EXPECT_EQ(values[2], 4);
    EXPECT_EQ(values[3], 16);

    values >>= 1;
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 4);
    EXPECT_EQ(values[2], 2);
    EXPECT_EQ(values[3], 8);
}

TEST(OpsBitwise, array_shape_mismatch_throws)
{
    Vector<int> short_vector{1, 2};
    Vector<int> long_vector{1, 2, 3};

    bool vector_threw = false;
    try {
        [[maybe_unused]] auto ignored = short_vector & long_vector;
    }
    catch (const Exceptions::ShapeError&) {
        vector_threw = true;
    }
    EXPECT_TRUE(vector_threw);

    Matrix<int> wide{{1, 2, 3}, {4, 5, 6}};
    Matrix<int> tall{{1, 2}, {3, 4}, {5, 6}};

    bool matrix_threw = false;
    try {
        [[maybe_unused]] auto ignored = wide ^ tall;
    }
    catch (const Exceptions::ShapeError&) {
        matrix_threw = true;
    }
    EXPECT_TRUE(matrix_threw);
}

TEST(OpsBitwise, compound_shape_mismatch_throws_without_mutation)
{
    Vector<int> lhs{3, 5};
    Vector<int> rhs{1, 2, 3};

    bool threw = false;
    try {
        lhs |= rhs;
    }
    catch (const Exceptions::ShapeError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
    EXPECT_EQ(lhs.size(), 2);
    EXPECT_EQ(lhs[0], 3);
    EXPECT_EQ(lhs[1], 5);
}
