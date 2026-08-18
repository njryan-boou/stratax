#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <type_traits>

#include <stratax/concepts/Numeric.hpp>

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

static_assert(Numeric<const std::int8_t&>);
static_assert(Numeric<volatile double&&>);

TEST(ConceptsNumeric, SupportsEightBitIntegerTypes)
{
	EXPECT_TRUE((Numeric<std::int8_t>));
	EXPECT_TRUE((Numeric<std::uint8_t>));
}
