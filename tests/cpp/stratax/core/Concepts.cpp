#include <gtest/gtest.h>
#include <complex>
#include <string>
#include <vector>

#include <stratax.h>

using namespace stratax::core;

static_assert(Integral<int>);
static_assert(Integral<const int>);
static_assert(Integral<int&>);
static_assert(Integral<const volatile long&>);
static_assert(Integral<long>);
static_assert(Integral<dtype::int16>);
static_assert(Integral<dtype::uint64>);

static_assert(!Integral<bool>);
static_assert(!Integral<const bool>);
static_assert(!Integral<char>);
static_assert(!Integral<const char>);
static_assert(!Integral<char&>);
static_assert(!Integral<signed char>);
static_assert(!Integral<const signed char&>);
static_assert(!Integral<unsigned char>);
static_assert(!Integral<const unsigned char&>);
static_assert(!Integral<wchar_t>);
static_assert(!Integral<char8_t>);
static_assert(!Integral<char16_t>);
static_assert(!Integral<char32_t>);
static_assert(!Integral<dtype::bool_>);
static_assert(!Integral<dtype::int8>);
static_assert(!Integral<dtype::uint8>);
static_assert(!Integral<float>);
static_assert(!Integral<double>);
static_assert(!Integral<std::complex<float>>);
static_assert(!Integral<std::string>);

static_assert(Integer<int>);
static_assert(Integer<const unsigned long&>);
static_assert(Integer<dtype::int32>);
static_assert(Integer<dtype::uint64>);

static_assert(!Integer<bool>);
static_assert(!Integer<char>);
static_assert(!Integer<dtype::int8>);
static_assert(!Integer<dtype::uint8>);
static_assert(!Integer<float>);
static_assert(!Integer<std::complex<double>>);

static_assert(SignedInteger<int>);
static_assert(SignedInteger<const long&>);
static_assert(SignedInteger<dtype::int16>);
static_assert(SignedInteger<dtype::int64>);

static_assert(!SignedInteger<unsigned int>);
static_assert(!SignedInteger<bool>);
static_assert(!SignedInteger<signed char>);
static_assert(!SignedInteger<dtype::int8>);
static_assert(!SignedInteger<float>);

static_assert(UnsignedInteger<unsigned int>);
static_assert(UnsignedInteger<const unsigned long&>);
static_assert(UnsignedInteger<dtype::uint16>);
static_assert(UnsignedInteger<dtype::uint64>);

static_assert(!UnsignedInteger<int>);
static_assert(!UnsignedInteger<bool>);
static_assert(!UnsignedInteger<unsigned char>);
static_assert(!UnsignedInteger<dtype::uint8>);
static_assert(!UnsignedInteger<double>);

static_assert(Floating<float>);
static_assert(Floating<const float>);
static_assert(Floating<double&>);
static_assert(Floating<double>);
static_assert(Floating<long double>);
static_assert(Floating<dtype::float32>);
static_assert(Floating<dtype::float64>);
static_assert(Floating<dtype::float128>);

static_assert(!Floating<int>);
static_assert(!Floating<bool>);
static_assert(!Floating<std::complex<double>>);
static_assert(!Floating<std::string>);

static_assert(Float<float>);
static_assert(Float<const double&>);
static_assert(Float<dtype::float128>);

static_assert(!Float<int>);
static_assert(!Float<std::complex<float>>);

static_assert(Complex<std::complex<float>>);
static_assert(Complex<const std::complex<float>>);
static_assert(Complex<std::complex<double>&>);
static_assert(Complex<std::complex<double>>);
static_assert(Complex<std::complex<long double>>);
static_assert(Complex<dtype::complex64>);
static_assert(Complex<dtype::complex128>);
static_assert(Complex<dtype::complex256>);

static_assert(!Complex<float>);
static_assert(!Complex<double>);
static_assert(!Complex<int>);
static_assert(!Complex<std::string>);

static_assert(Real<int>);
static_assert(Real<unsigned int>);
static_assert(Real<float>);
static_assert(Real<const double&>);

static_assert(!Real<bool>);
static_assert(!Real<char>);
static_assert(!Real<std::complex<double>>);
static_assert(!Real<std::string>);

static_assert(Numeric<int>);
static_assert(Numeric<const int>);
static_assert(Numeric<int&>);
static_assert(Numeric<dtype::int32>);
static_assert(Numeric<dtype::uint32>);
static_assert(Numeric<float>);
static_assert(Numeric<const double&>);
static_assert(Numeric<double>);
static_assert(Numeric<std::complex<float>>);
static_assert(Numeric<const std::complex<double>&>);
static_assert(Numeric<std::complex<double>>);
static_assert(Numeric<std::complex<long double>>);
static_assert(Numeric<dtype::complex256>);

static_assert(!Numeric<bool>);
static_assert(!Numeric<const bool&>);
static_assert(!Numeric<char>);
static_assert(!Numeric<const char&>);
static_assert(!Numeric<signed char>);
static_assert(!Numeric<unsigned char>);
static_assert(!Numeric<dtype::int8>);
static_assert(!Numeric<dtype::uint8>);
static_assert(!Numeric<std::string>);
static_assert(!Numeric<std::vector<int>>);

static_assert(Scalar<int>);
static_assert(Scalar<float>);
static_assert(Scalar<std::complex<double>>);

static_assert(!Scalar<bool>);
static_assert(!Scalar<char>);
static_assert(!Scalar<std::string>);

static_assert(Array<stratax::container::Vector<int>>);
static_assert(Array<stratax::container::Matrix<int>>);
static_assert(Array<stratax::container::Tensor<int>>);
static_assert(Array<const stratax::container::Vector<double>&>);
static_assert(Array<stratax::container::Matrix<std::complex<double>>&&>);

static_assert(!Array<int>);
static_assert(!Array<std::vector<int>>);
static_assert(!Array<std::string>);

static_assert(NDarray<stratax::container::Vector<int>>);
static_assert(NDarray<const stratax::container::Matrix<double>>);
static_assert(NDarray<stratax::container::Tensor<float>&>);

static_assert(!NDarray<int>);
static_assert(!NDarray<std::vector<int>>);

template<Numeric T>
constexpr bool accepts_numeric(T)
{
    return true;
}

static_assert(accepts_numeric(42));
static_assert(accepts_numeric(3.14));
static_assert(accepts_numeric(std::complex<double>{1.0, 2.0}));


TEST(CoreConcepts, Compiles)
{
    SUCCEED();
}
