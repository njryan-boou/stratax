#pragma once

#include <climits>
#include <cstddef>
#include <limits>
#include <string_view>

#include <stratax/core/Types.hpp>

namespace stratax::core {

enum class DTypeKind
{
	Bool,
	SignedInteger,
	UnsignedInteger,
	Floating,
	Complex
};

template<typename T>
struct DTypeTraits;

#define STRATAX_DEFINE_DTYPE_TRAITS(TYPE, KIND, NAME) \
	template<> \
	struct DTypeTraits<TYPE> \
	{ \
		using type = TYPE; \
		static constexpr DTypeKind kind = DTypeKind::KIND; \
		static constexpr std::size_t bits = sizeof(type) * CHAR_BIT; \
		static constexpr int digits = std::numeric_limits<type>::digits; \
		static constexpr std::string_view name = NAME; \
	}

STRATAX_DEFINE_DTYPE_TRAITS(dtype::bool_, Bool, "bool");

STRATAX_DEFINE_DTYPE_TRAITS(dtype::int8, SignedInteger, "int8");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::int16, SignedInteger, "int16");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::int32, SignedInteger, "int32");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::int64, SignedInteger, "int64");

STRATAX_DEFINE_DTYPE_TRAITS(dtype::uint8, UnsignedInteger, "uint8");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::uint16, UnsignedInteger, "uint16");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::uint32, UnsignedInteger, "uint32");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::uint64, UnsignedInteger, "uint64");

STRATAX_DEFINE_DTYPE_TRAITS(dtype::float32, Floating, "float32");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::float64, Floating, "float64");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::longdouble, Floating, "longdouble");

STRATAX_DEFINE_DTYPE_TRAITS(dtype::complex64, Complex, "complex64");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::complex128, Complex, "complex128");
STRATAX_DEFINE_DTYPE_TRAITS(dtype::clongdouble, Complex, "clongdouble");

#undef STRATAX_DEFINE_DTYPE_TRAITS

} // namespace stratax::core