#include <complex>
#include <string>
#include <vector>

#include <stratax.hpp>

using namespace stratax::core;

static_assert(Integral<int>);
static_assert(Integral<long>);
static_assert(Integral<dtype::int16>);
static_assert(Integral<dtype::uint64>);

static_assert(!Integral<bool>);
static_assert(!Integral<char>);
static_assert(!Integral<signed char>);
static_assert(!Integral<unsigned char>);
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

static_assert(Floating<float>);
static_assert(Floating<double>);
static_assert(Floating<long double>);
static_assert(Floating<dtype::float32>);
static_assert(Floating<dtype::float64>);
static_assert(Floating<dtype::float128>);

static_assert(!Floating<int>);
static_assert(!Floating<bool>);
static_assert(!Floating<std::complex<double>>);
static_assert(!Floating<std::string>);

static_assert(Complex<std::complex<float>>);
static_assert(Complex<std::complex<double>>);
static_assert(Complex<std::complex<long double>>);
static_assert(Complex<dtype::complex64>);
static_assert(Complex<dtype::complex128>);
static_assert(Complex<dtype::complex256>);

static_assert(!Complex<float>);
static_assert(!Complex<double>);
static_assert(!Complex<int>);
static_assert(!Complex<std::string>);

static_assert(Numeric<int>);
static_assert(Numeric<dtype::int32>);
static_assert(Numeric<dtype::uint32>);
static_assert(Numeric<float>);
static_assert(Numeric<double>);
static_assert(Numeric<std::complex<float>>);
static_assert(Numeric<std::complex<double>>);
static_assert(Numeric<std::complex<long double>>);
static_assert(Numeric<dtype::complex256>);

static_assert(!Numeric<bool>);
static_assert(!Numeric<char>);
static_assert(!Numeric<signed char>);
static_assert(!Numeric<unsigned char>);
static_assert(!Numeric<dtype::int8>);
static_assert(!Numeric<dtype::uint8>);
static_assert(!Numeric<std::string>);
static_assert(!Numeric<std::vector<int>>);

template<Numeric T>
constexpr bool accepts_numeric(T)
{
    return true;
}

static_assert(accepts_numeric(42));
static_assert(accepts_numeric(3.14));
static_assert(accepts_numeric(std::complex<double>{1.0, 2.0}));

int main()
{
    return 0;
}
