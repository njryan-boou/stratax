#include <cassert>
#include <complex>
#include <limits>
#include <string>

#include <stratax/containers/Creation.hpp>

template<typename T>
concept CanCreateZeros = requires(const stratax::core::Shape& shape) {
    creation::zeros<T>(shape);
};

static_assert(CanCreateZeros<int>);
static_assert(!CanCreateZeros<std::string>);

void test_zeros_creates_tensor_with_shape_and_zero_values()
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{2, 3});

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 3);
    assert(tensor.size() == 6);

    for (int value : tensor) {
        assert(value == 0);
    }
}

void test_ones_creates_tensor_with_shape_and_one_values()
{
    auto tensor = creation::ones<int>(stratax::core::Shape{2, 2});

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 2);
    assert(tensor.size() == 4);

    for (int value : tensor) {
        assert(value == 1);
    }
}

void test_full_creates_tensor_with_shape_and_fill_value()
{
    auto tensor = creation::full<int>(stratax::core::Shape{2, 2, 2}, 7);

    assert(tensor.rank() == 3);
    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 2);
    assert(tensor.shape()(2) == 2);
    assert(tensor.size() == 8);

    for (int value : tensor) {
        assert(value == 7);
    }
}

void test_full_supports_complex_values()
{
    const std::complex<double> value{1.5, -2.5};
    auto tensor = creation::full<std::complex<double>>(stratax::core::Shape{2}, value);

    assert(tensor.rank() == 1);
    assert(tensor.shape()(0) == 2);
    assert(tensor.size() == 2);

    for (const auto& element : tensor) {
        assert(element == value);
    }
}

void test_identity_creates_square_identity_tensor()
{
    auto tensor = creation::identity<int>(3);

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 3);
    assert(tensor.shape()(1) == 3);
    assert(tensor.size() == 9);

    assert(tensor(0, 0) == 1);
    assert(tensor(0, 1) == 0);
    assert(tensor(0, 2) == 0);
    assert(tensor(1, 0) == 0);
    assert(tensor(1, 1) == 1);
    assert(tensor(1, 2) == 0);
    assert(tensor(2, 0) == 0);
    assert(tensor(2, 1) == 0);
    assert(tensor(2, 2) == 1);
}

void test_identity_zero_size_creates_empty_square_tensor()
{
    auto tensor = creation::identity<int>(0);

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 0);
    assert(tensor.shape()(1) == 0);
    assert(tensor.size() == 0);
    assert(tensor.empty());
}

void test_identity_supports_complex_values()
{
    auto tensor = creation::identity<std::complex<double>>(2);

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 2);
    assert(tensor.shape()(1) == 2);

    assert(tensor(0, 0) == std::complex<double>{1});
    assert(tensor(0, 1) == std::complex<double>{});
    assert(tensor(1, 0) == std::complex<double>{});
    assert(tensor(1, 1) == std::complex<double>{1});
}

void test_zero_sized_shape_creates_empty_tensor()
{
    auto tensor = creation::zeros<int>(stratax::core::Shape{0, 3});

    assert(tensor.rank() == 2);
    assert(tensor.shape()(0) == 0);
    assert(tensor.shape()(1) == 3);
    assert(tensor.size() == 0);
    assert(tensor.empty());
    assert(tensor.begin() == tensor.end());
}

void test_empty_shape_creates_rank_zero_tensor()
{
    auto tensor = creation::ones<int>(stratax::core::Shape{});

    assert(tensor.rank() == 0);
    assert(tensor.size() == 0);
    assert(tensor.empty());
    assert(tensor.shape().empty());
}

void test_overflow_shape_throws()
{
    bool zeros_threw = false;

    try {
        auto tensor = creation::zeros<int>(
            stratax::core::Shape{2, std::numeric_limits<std::size_t>::max(), 2});
    }
    catch (const Exceptions::DimensionError&) {
        zeros_threw = true;
    }

    assert(zeros_threw);

    bool identity_threw = false;

    try {
        auto tensor = creation::identity<int>(std::numeric_limits<std::size_t>::max());
    }
    catch (const Exceptions::DimensionError&) {
        identity_threw = true;
    }

    assert(identity_threw);
}

int main()
{
    test_zeros_creates_tensor_with_shape_and_zero_values();
    test_ones_creates_tensor_with_shape_and_one_values();
    test_full_creates_tensor_with_shape_and_fill_value();
    test_full_supports_complex_values();
    test_identity_creates_square_identity_tensor();
    test_identity_zero_size_creates_empty_square_tensor();
    test_identity_supports_complex_values();
    test_zero_sized_shape_creates_empty_tensor();
    test_empty_shape_creates_rank_zero_tensor();
    test_overflow_shape_throws();

    return 0;
}
