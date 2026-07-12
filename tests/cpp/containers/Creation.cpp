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

