#include <stratax.h>

namespace dtype = stratax::dtype;
using namespace stratax::core;

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int8, dtype::int8>,
	dtype::int8>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int8, dtype::int16>,
	dtype::int16>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int32, dtype::int64>,
	dtype::int64>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::uint8, dtype::uint32>,
	dtype::uint32>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::uint64, dtype::uint16>,
	dtype::uint64>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int8, dtype::uint8>,
	dtype::int16>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int16, dtype::uint8>,
	dtype::int16>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int16, dtype::uint16>,
	dtype::int32>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int32, dtype::uint32>,
	dtype::int64>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int64, dtype::uint32>,
	dtype::int64>);

static_assert(std::same_as<
	stratax::core::promote_t<dtype::int64, dtype::uint64>,
	dtype::float64>);

// Symmetry
static_assert(std::same_as<
	stratax::core::promote_t<dtype::uint16, dtype::int16>,
	stratax::core::promote_t<dtype::int16, dtype::uint16>>);

static_assert(std::same_as<
	promote_t<dtype::int16, dtype::float32>,
	dtype::float32>);

static_assert(std::same_as<
	promote_t<dtype::int32, dtype::float32>,
	dtype::float64>);

static_assert(std::same_as<
	promote_t<dtype::uint32, dtype::float32>,
	dtype::float64>);

static_assert(std::same_as<
	promote_t<dtype::int32, dtype::float64>,
	dtype::float64>);

static_assert(std::same_as<
	promote_t<dtype::float32, dtype::int32>,
	dtype::float64>);

static_assert(std::same_as<
	promote_t<dtype::int64, dtype::longdouble>,
	dtype::longdouble>);

static_assert(std::same_as<
	promote_t<dtype::float32, dtype::float64>,
	dtype::float64>);

static_assert(std::same_as<
	promote_t<dtype::float64, dtype::float32>,
	dtype::float64>);

static_assert(std::same_as<
	promote_t<dtype::float32, dtype::longdouble>,
	dtype::longdouble>);

static_assert(std::same_as<
	promote_t<dtype::float64, dtype::longdouble>,
	dtype::longdouble>);

static_assert(std::same_as<
	promote_t<dtype::complex64, dtype::complex128>,
	dtype::complex128>);

static_assert(std::same_as<
	promote_t<dtype::complex128, dtype::complex64>,
	dtype::complex128>);

static_assert(std::same_as<
	promote_t<dtype::complex64, dtype::clongdouble>,
	dtype::clongdouble>);

static_assert(std::same_as<
	promote_t<dtype::complex128, dtype::clongdouble>,
	dtype::clongdouble>);

static_assert(std::same_as<
	promote_t<dtype::float32, dtype::complex64>,
	dtype::complex64>);

static_assert(std::same_as<
	promote_t<dtype::float64, dtype::complex64>,
	dtype::complex128>);

static_assert(std::same_as<
	promote_t<dtype::float32, dtype::complex128>,
	dtype::complex128>);

static_assert(std::same_as<
	promote_t<dtype::longdouble, dtype::complex64>,
	dtype::clongdouble>);

static_assert(std::same_as<
	promote_t<dtype::int16, dtype::complex64>,
	dtype::complex64>);

static_assert(std::same_as<
	promote_t<dtype::int32, dtype::complex64>,
	dtype::complex128>);

static_assert(std::same_as<
	promote_t<dtype::int64, dtype::complex128>,
	dtype::complex128>);

static_assert(std::same_as<
	promote_t<dtype::complex64, dtype::float64>,
	promote_t<dtype::float64, dtype::complex64>>);

static_assert(std::same_as<
	promote_t<dtype::complex64, dtype::int32>,
	promote_t<dtype::int32, dtype::complex64>>);

// Same-type promotion for every category.
static_assert(std::same_as<
	promote_t<dtype::uint32, dtype::uint32>,
	dtype::uint32>);

static_assert(std::same_as<
	promote_t<dtype::float32, dtype::float32>,
	dtype::float32>);

static_assert(std::same_as<
	promote_t<dtype::complex64, dtype::complex64>,
	dtype::complex64>);

// cv/ref normalization.
static_assert(std::same_as<
	promote_t<const dtype::int16&, volatile dtype::uint16>,
	dtype::int32>);

// More mixed integer boundary cases.
static_assert(std::same_as<
	promote_t<dtype::int8, dtype::uint16>,
	dtype::int32>);

static_assert(std::same_as<
	promote_t<dtype::int32, dtype::uint16>,
	dtype::int32>);

static_assert(std::same_as<
	promote_t<dtype::uint64, dtype::int64>,
	dtype::float64>);

// Reverse integer/floating order.
static_assert(std::same_as<
	promote_t<dtype::float32, dtype::int16>,
	dtype::float32>);

static_assert(std::same_as<
	promote_t<dtype::float32, dtype::uint32>,
	dtype::float64>);

// Reverse real/complex order.
static_assert(std::same_as<
	promote_t<dtype::complex64, dtype::int16>,
	dtype::complex64>);

static_assert(std::same_as<
	promote_t<dtype::complex64, dtype::longdouble>,
	dtype::clongdouble>);
