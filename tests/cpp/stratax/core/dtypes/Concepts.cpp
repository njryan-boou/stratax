#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <type_traits>

#include <stratax/core/dtypes/Concepts.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

namespace dtype = stratax::dtype;
namespace detail = stratax::core::concept_detail;

struct Unrelated {};

static_assert(detail::SameAsAny<const dtype::int8&, dtype::int8>);
static_assert(!detail::SameAsAny<Unrelated, dtype::int8, dtype::float32>);

static_assert(detail::BoolLike<bool>);
static_assert(detail::BoolLike<const dtype::bool_&>);
static_assert(!detail::BoolLike<dtype::uint8>);

static_assert(detail::SupportedIntegral<dtype::int8>);
static_assert(detail::SupportedIntegral<dtype::int16>);
static_assert(detail::SupportedIntegral<dtype::int32>);
static_assert(detail::SupportedIntegral<dtype::int64>);
static_assert(detail::SupportedIntegral<dtype::uint8>);
static_assert(detail::SupportedIntegral<dtype::uint16>);
static_assert(detail::SupportedIntegral<dtype::uint32>);
static_assert(detail::SupportedIntegral<dtype::uint64>);
static_assert(!detail::SupportedIntegral<bool>);
static_assert(!detail::SupportedIntegral<dtype::float32>);

static_assert(detail::SupportedFloating<dtype::float32>);
static_assert(detail::SupportedFloating<dtype::float64>);
static_assert(detail::SupportedFloating<dtype::longdouble>);
static_assert(detail::SupportedFloating<const dtype::float64&>);
static_assert(!detail::SupportedFloating<dtype::int32>);

static_assert(detail::SupportedComplex<dtype::complex64>);
static_assert(detail::SupportedComplex<dtype::complex128>);
static_assert(detail::SupportedComplex<dtype::clongdouble>);
static_assert(detail::SupportedComplex<volatile dtype::complex128&&>);
static_assert(!detail::SupportedComplex<dtype::float64>);

static_assert(Integral<dtype::int8>);
static_assert(Integral<dtype::int16>);
static_assert(Integral<dtype::int32>);
static_assert(Integral<dtype::int64>);
static_assert(Integral<dtype::uint8>);
static_assert(Integral<dtype::uint16>);
static_assert(Integral<dtype::uint32>);
static_assert(Integral<dtype::uint64>);
static_assert(Integral<const dtype::int32&>);
static_assert(!Integral<bool>);
static_assert(!Integral<dtype::float32>);
static_assert(!Integral<dtype::complex64>);

static_assert(Numeric<dtype::int8>);
static_assert(Numeric<dtype::uint64>);
static_assert(Numeric<dtype::float32>);
static_assert(Numeric<dtype::float64>);
static_assert(Numeric<dtype::longdouble>);
static_assert(Numeric<dtype::complex64>);
static_assert(Numeric<dtype::complex128>);
static_assert(Numeric<dtype::clongdouble>);

static_assert(!Numeric<bool>);
static_assert(!Numeric<char>);
static_assert(!Numeric<wchar_t>);
static_assert(!Numeric<char8_t>);
static_assert(!Numeric<char16_t>);
static_assert(!Numeric<char32_t>);

static_assert(DType<bool>);
static_assert(DType<const bool&>);
static_assert(DType<volatile bool&&>);
static_assert(DType<dtype::int32>);
static_assert(DType<dtype::float64>);
static_assert(DType<dtype::complex128>);
static_assert(!DType<char>);
static_assert(!DType<wchar_t>);
static_assert(!DType<char8_t>);
static_assert(!DType<char16_t>);
static_assert(!DType<char32_t>);

static_assert(Numeric<const dtype::int8&>);
static_assert(Numeric<volatile dtype::float64&&>);

static_assert(Ordered<bool>);
static_assert(Ordered<dtype::int32>);
static_assert(Ordered<dtype::uint64>);
static_assert(Ordered<dtype::float32>);
static_assert(Ordered<const dtype::float64&>);
static_assert(!Ordered<dtype::complex64>);
static_assert(!Ordered<dtype::complex128>);
static_assert(!Ordered<Unrelated>);

static_assert(Array<stratax::container::Vector<bool>>);
static_assert(Array<stratax::container::Matrix<bool>>);
static_assert(Array<stratax::container::Tensor<bool>>);
static_assert(Array<const stratax::container::Vector<dtype::float32>&>);
static_assert(Array<volatile stratax::container::Matrix<dtype::int16>&&>);
static_assert(!Array<Unrelated>);
static_assert(!Array<int>);

static_assert(is_array<stratax::container::Vector<dtype::float64>>::value);
static_assert(is_array<stratax::container::Matrix<dtype::int32>>::value);
static_assert(is_array<stratax::container::Tensor<dtype::complex128>>::value);
static_assert(!is_array<const stratax::container::Vector<dtype::float64>>::value);
static_assert(!is_array<Unrelated>::value);

TEST(ConceptsNumeric, SupportsEightBitIntegerTypes)
{
	EXPECT_TRUE((Numeric<dtype::int8>));
	EXPECT_TRUE((Numeric<dtype::uint8>));
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
