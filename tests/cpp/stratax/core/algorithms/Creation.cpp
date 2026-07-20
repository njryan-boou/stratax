#include <gtest/gtest.h>
#include <complex>
#include <limits>
#include <string>

#include <stratax.h>

template<typename T>
concept CanCreateZeros = requires(const stratax::core::Shape& shape) {
    creation::zeros<T>(shape);
};

static_assert(CanCreateZeros<int>);
static_assert(!CanCreateZeros<std::string>);

TEST(ContainersCreation, zeros_creates_tensor_with_shape_and_zero_values)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{2, 3});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.size() == 6);

    for (int value : tensor) {
        EXPECT_TRUE(value == 0);
    }
}

TEST(ContainersCreation, ones_creates_tensor_with_shape_and_one_values)
{
    auto tensor = creation::ones<int>(stratax::core::Shape{2, 2});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor.size() == 4);

    for (int value : tensor) {
        EXPECT_TRUE(value == 1);
    }
}

TEST(ContainersCreation, full_creates_tensor_with_shape_and_fill_value)
{
    auto tensor = creation::full<int>(stratax::core::Shape{2, 2, 2}, 7);

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);
    EXPECT_TRUE(tensor.shape()(2) == 2);
    EXPECT_TRUE(tensor.size() == 8);

    for (int value : tensor) {
        EXPECT_TRUE(value == 7);
    }
}

TEST(ContainersCreation, full_supports_complex_values)
{
    const std::complex<double> value{1.5, -2.5};
    auto tensor = creation::full<std::complex<double>>(stratax::core::Shape{2}, value);

    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.size() == 2);

    for (const auto& element : tensor) {
        EXPECT_TRUE(element == value);
    }
}

TEST(ContainersCreation, identity_creates_square_identity_tensor)
{
    auto tensor = creation::identity<int>(3);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.size() == 9);

    EXPECT_TRUE(tensor(0, 0) == 1);
    EXPECT_TRUE(tensor(0, 1) == 0);
    EXPECT_TRUE(tensor(0, 2) == 0);
    EXPECT_TRUE(tensor(1, 0) == 0);
    EXPECT_TRUE(tensor(1, 1) == 1);
    EXPECT_TRUE(tensor(1, 2) == 0);
    EXPECT_TRUE(tensor(2, 0) == 0);
    EXPECT_TRUE(tensor(2, 1) == 0);
    EXPECT_TRUE(tensor(2, 2) == 1);
}

TEST(ContainersCreation, identity_zero_size_creates_empty_square_tensor)
{
    auto tensor = creation::identity<int>(0);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 0);
    EXPECT_TRUE(tensor.shape()(1) == 0);
    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
}

TEST(ContainersCreation, identity_supports_complex_values)
{
    auto tensor = creation::identity<std::complex<double>>(2);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 2);

    EXPECT_TRUE(tensor(0, 0) == std::complex<double>{1});
    EXPECT_TRUE(tensor(0, 1) == std::complex<double>{});
    EXPECT_TRUE(tensor(1, 0) == std::complex<double>{});
    EXPECT_TRUE(tensor(1, 1) == std::complex<double>{1});
}

TEST(ContainersCreation, zero_sized_shape_creates_empty_tensor)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{0, 3});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 0);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
    EXPECT_TRUE(tensor.begin() == tensor.end());
}

TEST(ContainersCreation, empty_shape_creates_rank_zero_tensor)
{
    auto tensor = creation::ones<int>(stratax::core::Shape{});

    EXPECT_TRUE(tensor.rank() == 0);
    EXPECT_TRUE(tensor.size() == 0);
    EXPECT_TRUE(tensor.empty());
    EXPECT_TRUE(tensor.shape().empty());
}

TEST(ContainersCreation, overflow_shape_throws)
{
    EXPECT_THROW(
        creation::zeros<int>(stratax::core::Shape{2, std::numeric_limits<long long>::max(), 2}),
        Exceptions::StrataxError);

    EXPECT_THROW(
        creation::identity<int>(std::numeric_limits<std::size_t>::max()),
        Exceptions::StrataxError);
}

TEST(ContainersCreation, zeros_float_1d)
{
    auto tensor = creation::zeros<float>(stratax::core::Shape{5});

    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor.shape()(0) == 5);
    EXPECT_TRUE(tensor.size() == 5);

    for (float value : tensor) {
        EXPECT_TRUE(value == 0.0f);
    }
}

TEST(ContainersCreation, zeros_double_2d)
{
    auto tensor = creation::zeros<double>(stratax::core::Shape{3, 4});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.shape()(1) == 4);
    EXPECT_TRUE(tensor.size() == 12);

    for (double value : tensor) {
        EXPECT_TRUE(value == 0.0);
    }
}

TEST(ContainersCreation, zeros_3d_tensor)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{2, 3, 4});

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.shape()(2) == 4);
    EXPECT_TRUE(tensor.size() == 24);

    for (int value : tensor) {
        EXPECT_TRUE(value == 0);
    }
}

TEST(ContainersCreation, zeros_large_tensor)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{50, 50});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 50);
    EXPECT_TRUE(tensor.shape()(1) == 50);
    EXPECT_TRUE(tensor.size() == 2500);

    for (int value : tensor) {
        EXPECT_TRUE(value == 0);
    }
}

TEST(ContainersCreation, zeros_single_element)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{1});

    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor.shape()(0) == 1);
    EXPECT_TRUE(tensor[0] == 0);
}

TEST(ContainersCreation, ones_float_1d)
{
    auto tensor = creation::ones<float>(stratax::core::Shape{5});

    EXPECT_TRUE(tensor.rank() == 1);
    EXPECT_TRUE(tensor.shape()(0) == 5);

    for (float value : tensor) {
        EXPECT_TRUE(value == 1.0f);
    }
}

TEST(ContainersCreation, ones_double_3d)
{
    auto tensor = creation::ones<double>(stratax::core::Shape{2, 3, 2});

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.size() == 12);

    for (double value : tensor) {
        EXPECT_TRUE(value == 1.0);
    }
}

TEST(ContainersCreation, ones_large_matrix)
{
    auto tensor = creation::ones<int>(stratax::core::Shape{100, 100});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.size() == 10000);

    for (int value : tensor) {
        EXPECT_TRUE(value == 1);
    }
}

TEST(ContainersCreation, ones_single_element)
{
    auto tensor = creation::ones<int>(stratax::core::Shape{1});

    EXPECT_TRUE(tensor[0] == 1);
}

TEST(ContainersCreation, full_float_negative)
{
    auto tensor = creation::full<float>(stratax::core::Shape{3, 3}, -5.5f);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.size() == 9);

    for (float value : tensor) {
        EXPECT_TRUE(value == -5.5f);
    }
}

TEST(ContainersCreation, full_large_value)
{
    auto tensor = creation::full<int>(stratax::core::Shape{2, 2}, 999999);

    EXPECT_TRUE(tensor.size() == 4);

    for (int value : tensor) {
        EXPECT_TRUE(value == 999999);
    }
}

TEST(ContainersCreation, full_3d_tensor)
{
    auto tensor = creation::full<int>(stratax::core::Shape{2, 3, 4}, 42);

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.size() == 24);

    for (int value : tensor) {
        EXPECT_TRUE(value == 42);
    }
}

TEST(ContainersCreation, full_double_mixed_shape)
{
    auto tensor = creation::full<double>(stratax::core::Shape{1, 5, 1}, 3.14);

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 1);
    EXPECT_TRUE(tensor.shape()(1) == 5);
    EXPECT_TRUE(tensor.shape()(2) == 1);
    EXPECT_TRUE(tensor.size() == 5);

    for (double value : tensor) {
        EXPECT_TRUE(value > 3.13 && value < 3.15);
    }
}

TEST(ContainersCreation, full_zero_fill_value)
{
    auto tensor = creation::full<int>(stratax::core::Shape{3, 3}, 0);

    EXPECT_TRUE(tensor.size() == 9);

    for (int value : tensor) {
        EXPECT_TRUE(value == 0);
    }
}

TEST(ContainersCreation, full_negative_value)
{
    auto tensor = creation::full<int>(stratax::core::Shape{2, 2}, -7);

    EXPECT_TRUE(tensor.size() == 4);

    for (int value : tensor) {
        EXPECT_TRUE(value == -7);
    }
}

TEST(ContainersCreation, full_complex_double)
{
    const std::complex<double> value{2.0, 3.0};
    auto tensor = creation::full<std::complex<double>>(stratax::core::Shape{2, 2}, value);

    EXPECT_TRUE(tensor.size() == 4);

    for (const auto& element : tensor) {
        EXPECT_TRUE(element.real() == 2.0);
        EXPECT_TRUE(element.imag() == 3.0);
    }
}

TEST(ContainersCreation, identity_1x1)
{
    auto tensor = creation::identity<int>(1);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 1);
    EXPECT_TRUE(tensor.shape()(1) == 1);
    EXPECT_TRUE(tensor(0, 0) == 1);
}

TEST(ContainersCreation, identity_2x2)
{
    auto tensor = creation::identity<int>(2);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.size() == 4);
    EXPECT_TRUE(tensor(0, 0) == 1);
    EXPECT_TRUE(tensor(0, 1) == 0);
    EXPECT_TRUE(tensor(1, 0) == 0);
    EXPECT_TRUE(tensor(1, 1) == 1);
}

TEST(ContainersCreation, identity_5x5)
{
    auto tensor = creation::identity<int>(5);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 5);
    EXPECT_TRUE(tensor.shape()(1) == 5);

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            if (i == j) {
                EXPECT_TRUE(tensor(i, j) == 1);
            } else {
                EXPECT_TRUE(tensor(i, j) == 0);
            }
        }
    }
}

TEST(ContainersCreation, identity_float)
{
    auto tensor = creation::identity<float>(3);

    EXPECT_TRUE(tensor(0, 0) == 1.0f);
    EXPECT_TRUE(tensor(0, 1) == 0.0f);
    EXPECT_TRUE(tensor(1, 1) == 1.0f);
}

TEST(ContainersCreation, identity_double)
{
    auto tensor = creation::identity<double>(3);

    EXPECT_TRUE(tensor(0, 0) == 1.0);
    EXPECT_TRUE(tensor(0, 1) == 0.0);
    EXPECT_TRUE(tensor(1, 1) == 1.0);
}

TEST(ContainersCreation, identity_large)
{
    auto tensor = creation::identity<int>(100);

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 100);
    EXPECT_TRUE(tensor.shape()(1) == 100);

    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(tensor(i, i) == 1);
    }
}

TEST(ContainersCreation, zeros_rectangular_3x5)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{3, 5});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 3);
    EXPECT_TRUE(tensor.shape()(1) == 5);
    EXPECT_TRUE(tensor.size() == 15);
}

TEST(ContainersCreation, zeros_rectangular_5x3)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{5, 3});

    EXPECT_TRUE(tensor.rank() == 2);
    EXPECT_TRUE(tensor.shape()(0) == 5);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.size() == 15);
}

TEST(ContainersCreation, ones_asymmetric_2x3x4)
{
    auto tensor = creation::ones<int>(stratax::core::Shape{2, 3, 4});

    EXPECT_TRUE(tensor.rank() == 3);
    EXPECT_TRUE(tensor.shape()(0) == 2);
    EXPECT_TRUE(tensor.shape()(1) == 3);
    EXPECT_TRUE(tensor.shape()(2) == 4);
    EXPECT_TRUE(tensor.size() == 24);

    for (int value : tensor) {
        EXPECT_TRUE(value == 1);
    }
}

TEST(ContainersCreation, full_unsigned_int)
{
    auto tensor = creation::full<unsigned int>(stratax::core::Shape{2, 2}, 100u);

    EXPECT_TRUE(tensor.size() == 4);

    for (unsigned int value : tensor) {
        EXPECT_TRUE(value == 100u);
    }
}

TEST(ContainersCreation, full_long)
{
    auto tensor = creation::full<long>(stratax::core::Shape{2, 2}, 1000000L);

    EXPECT_TRUE(tensor.size() == 4);

    for (long value : tensor) {
        EXPECT_TRUE(value == 1000000L);
    }
}

TEST(ContainersCreation, zeros_rank_4)
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{2, 2, 2, 2});

    EXPECT_TRUE(tensor.rank() == 4);
    EXPECT_TRUE(tensor.size() == 16);

    for (int value : tensor) {
        EXPECT_TRUE(value == 0);
    }
}

TEST(ContainersCreation, ones_rank_5)
{
    auto tensor = creation::ones<int>(stratax::core::Shape{2, 2, 2, 2, 2});

    EXPECT_TRUE(tensor.rank() == 5);
    EXPECT_TRUE(tensor.size() == 32);

    for (int value : tensor) {
        EXPECT_TRUE(value == 1);
    }
}

TEST(ContainersCreation, full_rank_4)
{
    auto tensor = creation::full<int>(stratax::core::Shape{2, 2, 2, 2}, 5);

    EXPECT_TRUE(tensor.rank() == 4);
    EXPECT_TRUE(tensor.size() == 16);

    for (int value : tensor) {
        EXPECT_TRUE(value == 5);
    }
}

