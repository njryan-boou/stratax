#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <type_traits>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

static_assert(Integral<signed char>);
static_assert(Integral<unsigned char>);
static_assert(Integral<std::int8_t>);
static_assert(Integral<std::uint8_t>);

static_assert(Numeric<signed char>);
static_assert(Numeric<unsigned char>);
static_assert(Numeric<std::int8_t>);
static_assert(Numeric<std::uint8_t>);

static_assert(Numeric<short>);
static_assert(Numeric<unsigned short>);
static_assert(Numeric<int>);
static_assert(Numeric<unsigned int>);
static_assert(Numeric<long>);
static_assert(Numeric<unsigned long>);
static_assert(Numeric<long long>);
static_assert(Numeric<unsigned long long>);
static_assert(Numeric<float>);
static_assert(Numeric<double>);
static_assert(Numeric<long double>);
static_assert(Numeric<std::complex<float>>);
static_assert(Numeric<std::complex<double>>);
static_assert(Numeric<std::complex<long double>>);

static_assert(!Numeric<bool>);
static_assert(!Numeric<char>);
static_assert(!Numeric<wchar_t>);
static_assert(!Numeric<char8_t>);
static_assert(!Numeric<char16_t>);
static_assert(!Numeric<char32_t>);

static_assert(DType<bool>);
static_assert(DType<const bool&>);
static_assert(DType<volatile bool&&>);
static_assert(DType<int>);
static_assert(DType<double>);
static_assert(DType<std::complex<double>>);
static_assert(!DType<char>);
static_assert(!DType<wchar_t>);
static_assert(!DType<char8_t>);
static_assert(!DType<char16_t>);
static_assert(!DType<char32_t>);

static_assert(Numeric<const std::int8_t&>);
static_assert(Numeric<volatile double&&>);

static_assert(Array<stratax::container::Vector<bool>>);
static_assert(Array<stratax::container::Matrix<bool>>);
static_assert(Array<stratax::container::Tensor<bool>>);

TEST(ConceptsNumeric, SupportsEightBitIntegerTypes)
{
	EXPECT_TRUE((Numeric<std::int8_t>));
	EXPECT_TRUE((Numeric<std::uint8_t>));
}

TEST(ConceptsDType, SupportsBooleanContainers)
{
	const stratax::container::Vector<bool> vector{true, false};
	const stratax::container::Matrix<bool> matrix{{true, false}};
	const stratax::container::Tensor<bool> tensor(
		stratax::core::Shape{2}, true);

	EXPECT_TRUE(vector[0]);
	EXPECT_FALSE(vector[1]);
	EXPECT_TRUE(matrix(0, 0));
	EXPECT_FALSE(matrix(0, 1));
	EXPECT_TRUE(tensor[0]);
	EXPECT_TRUE(tensor[1]);
}
