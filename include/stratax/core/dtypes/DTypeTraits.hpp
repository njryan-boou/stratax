#pragma once

#include <climits>
#include <cstddef>
#include <limits>
#include <string_view>
#include <type_traits>

#include <stratax/core/dtypes/Types.hpp>

namespace stratax::core {

/**
 * @brief Identifies the general category of a Stratax dtype.
 */
enum class DTypeKind
{
	Bool,
	SignedInteger,
	UnsignedInteger,
	Floating,
	Complex
};

namespace dtype_detail {

/**
 * @brief Implementation registry for supported Stratax dtypes.
 *
 * The primary template is intentionally undefined. Only explicitly
 * registered dtypes provide metadata.
 */
template<typename T>
struct DTypeTraitsImpl;

/**
 * @brief Defines metadata for a non-complex Stratax dtype.
 */
#define STRATAX_DEFINE_DTYPE_TRAITS(TYPE, KIND, NAME) \
	template<> \
	struct DTypeTraitsImpl<TYPE> \
	{ \
		using type = TYPE; \
		static constexpr DTypeKind kind = DTypeKind::KIND; \
		static constexpr std::size_t bits = sizeof(type) * CHAR_BIT; \
		static constexpr int digits = std::numeric_limits<type>::digits; \
		static constexpr std::string_view name = NAME; \
	}

/**
 * @brief Defines metadata for a ranked floating-point Stratax dtype.
 */
#define STRATAX_DEFINE_FLOAT_DTYPE_TRAITS(TYPE, RANK, NAME) \
	template<> \
	struct DTypeTraitsImpl<TYPE> \
	{ \
		using type = TYPE; \
		static constexpr DTypeKind kind = DTypeKind::Floating; \
		static constexpr std::size_t bits = sizeof(type) * CHAR_BIT; \
		static constexpr int digits = std::numeric_limits<type>::digits; \
		static constexpr std::size_t rank = RANK; \
		static constexpr std::string_view name = NAME; \
	}

/**
 * @brief Defines metadata for a complex Stratax dtype.
 *
 * Complex precision is defined by the precision of its underlying
 * floating-point component type.
 */
#define STRATAX_DEFINE_COMPLEX_DTYPE_TRAITS(TYPE, COMPONENT, RANK, NAME) \
	template<> \
	struct DTypeTraitsImpl<TYPE> \
	{ \
		using type = TYPE; \
		using component_type = COMPONENT; \
		static constexpr DTypeKind kind = DTypeKind::Complex; \
		static constexpr std::size_t bits = sizeof(type) * CHAR_BIT; \
		static constexpr int digits = std::numeric_limits<component_type>::digits; \
		static constexpr std::size_t rank = RANK; \
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

STRATAX_DEFINE_FLOAT_DTYPE_TRAITS(dtype::float32, 0, "float32");
STRATAX_DEFINE_FLOAT_DTYPE_TRAITS(dtype::float64, 1, "float64");
STRATAX_DEFINE_FLOAT_DTYPE_TRAITS(dtype::longdouble, 2, "longdouble");

STRATAX_DEFINE_COMPLEX_DTYPE_TRAITS(dtype::complex64, dtype::float32, 0, "complex64");
STRATAX_DEFINE_COMPLEX_DTYPE_TRAITS(dtype::complex128, dtype::float64, 1, "complex128");
STRATAX_DEFINE_COMPLEX_DTYPE_TRAITS(dtype::clongdouble, dtype::longdouble, 2, "clongdouble");

#undef STRATAX_DEFINE_DTYPE_TRAITS
#undef STRATAX_DEFINE_FLOAT_DTYPE_TRAITS
#undef STRATAX_DEFINE_COMPLEX_DTYPE_TRAITS

} // namespace dtype_detail

/**
 * @brief Provides compile-time metadata for a supported Stratax dtype.
 *
 * cv- and reference qualifiers are removed before looking up the dtype.
 *
 * @tparam T Type whose dtype metadata is requested.
 */
template<typename T>
struct DTypeTraits
	: dtype_detail::DTypeTraitsImpl<std::remove_cvref_t<T>>
{};

/**
 * @brief Maps a complex dtype to its underlying real component dtype.
 */
template<typename T>
struct ComplexComponent;

#define STRATAX_DEFINE_COMPLEX_COMPONENT(COMPLEX, REAL) \
	template<> \
	struct ComplexComponent<COMPLEX> \
	{ \
		using type = REAL; \
	}

STRATAX_DEFINE_COMPLEX_COMPONENT(dtype::complex64, dtype::float32);
STRATAX_DEFINE_COMPLEX_COMPONENT(dtype::complex128, dtype::float64);
STRATAX_DEFINE_COMPLEX_COMPONENT(dtype::clongdouble, dtype::longdouble);

#undef STRATAX_DEFINE_COMPLEX_COMPONENT

/**
 * @brief Underlying real dtype of a complex dtype.
 */
template<typename T>
using complex_component_t =
	typename ComplexComponent<std::remove_cvref_t<T>>::type;

/**
 * @brief Maps a real floating-point dtype to its corresponding complex dtype.
 */
template<typename T>
struct ComplexFromReal;

#define STRATAX_DEFINE_COMPLEX_FROM_REAL(REAL, COMPLEX) \
	template<> \
	struct ComplexFromReal<REAL> \
	{ \
		using type = COMPLEX; \
	}

STRATAX_DEFINE_COMPLEX_FROM_REAL(dtype::float32, dtype::complex64);
STRATAX_DEFINE_COMPLEX_FROM_REAL(dtype::float64, dtype::complex128);
STRATAX_DEFINE_COMPLEX_FROM_REAL(dtype::longdouble, dtype::clongdouble);

#undef STRATAX_DEFINE_COMPLEX_FROM_REAL

/**
 * @brief Complex dtype corresponding to a real floating-point dtype.
 */
template<typename T>
using complex_from_real_t =
	typename ComplexFromReal<std::remove_cvref_t<T>>::type;

} // namespace stratax::core
