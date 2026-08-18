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
