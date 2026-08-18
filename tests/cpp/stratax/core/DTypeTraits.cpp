#include <complex>
#include <cstdint>

#include <stratax/core/DTypeTraits.hpp>

using stratax::core::DTypeKind;
using stratax::core::DTypeTraits;
using namespace stratax::core;
using namespace stratax::dtype;

static_assert(
	DTypeTraits<std::int32_t>::kind ==
	DTypeKind::SignedInteger);

static_assert(
	DTypeTraits<std::uint32_t>::kind ==
	DTypeKind::UnsignedInteger);

static_assert(
	DTypeTraits<float>::kind ==
	DTypeKind::Floating);

static_assert(
	DTypeTraits<std::complex<double>>::kind ==
	DTypeKind::Complex);

static_assert(DTypeTraits<std::int8_t>::bits == 8);
static_assert(DTypeTraits<std::uint8_t>::bits == 8);
static_assert(DTypeTraits<std::int16_t>::bits == 16);
static_assert(DTypeTraits<std::int32_t>::bits == 32);
static_assert(DTypeTraits<std::int64_t>::bits == 64);

static_assert(DTypeTraits<std::int32_t>::name == "int32");
static_assert(DTypeTraits<std::int8_t>::name == "int8");
static_assert(DTypeTraits<std::uint8_t>::name == "uint8");
static_assert(DTypeTraits<std::uint32_t>::name == "uint32");

static_assert(DTypeTraits<bool_>::kind == DTypeKind::Bool);
static_assert(DTypeTraits<bool_>::bits == sizeof(bool_) * CHAR_BIT);
static_assert(DTypeTraits<bool_>::name == "bool");

static_assert(DTypeTraits<int8>::kind == DTypeKind::SignedInteger);
static_assert(DTypeTraits<int16>::kind == DTypeKind::SignedInteger);
static_assert(DTypeTraits<int32>::kind == DTypeKind::SignedInteger);
static_assert(DTypeTraits<int64>::kind == DTypeKind::SignedInteger);
static_assert(DTypeTraits<int8>::name == "int8");
static_assert(DTypeTraits<int16>::name == "int16");
static_assert(DTypeTraits<int32>::name == "int32");
static_assert(DTypeTraits<int64>::name == "int64");

static_assert(DTypeTraits<uint8>::kind == DTypeKind::UnsignedInteger);
static_assert(DTypeTraits<uint16>::kind == DTypeKind::UnsignedInteger);
static_assert(DTypeTraits<uint32>::kind == DTypeKind::UnsignedInteger);
static_assert(DTypeTraits<uint64>::kind == DTypeKind::UnsignedInteger);
static_assert(DTypeTraits<uint8>::name == "uint8");
static_assert(DTypeTraits<uint16>::name == "uint16");
static_assert(DTypeTraits<uint32>::name == "uint32");
static_assert(DTypeTraits<uint64>::name == "uint64");

static_assert(DTypeTraits<float32>::kind == DTypeKind::Floating);
static_assert(DTypeTraits<float64>::kind == DTypeKind::Floating);
static_assert(DTypeTraits<longdouble>::kind == DTypeKind::Floating);
static_assert(DTypeTraits<float32>::name == "float32");
static_assert(DTypeTraits<float64>::name == "float64");
static_assert(DTypeTraits<longdouble>::name == "longdouble");

static_assert(DTypeTraits<complex64>::kind == DTypeKind::Complex);
static_assert(DTypeTraits<complex128>::kind == DTypeKind::Complex);
static_assert(DTypeTraits<clongdouble>::kind == DTypeKind::Complex);
static_assert(DTypeTraits<complex64>::name == "complex64");
static_assert(DTypeTraits<complex128>::name == "complex128");
static_assert(DTypeTraits<clongdouble>::name == "clongdouble");

static_assert(DTypeTraits<index_t>::kind == DTypeKind::UnsignedInteger);
static_assert(DTypeTraits<ssize_t>::kind == DTypeKind::SignedInteger);
static_assert(DTypeTraits<index_t>::bits == sizeof(index_t) * CHAR_BIT);
static_assert(DTypeTraits<ssize_t>::bits == sizeof(ssize_t) * CHAR_BIT);

static_assert(DTypeTraits<const bool_&>::kind == DTypeKind::Bool);
static_assert(DTypeTraits<const bool_&>::name == "bool");
static_assert(DTypeTraits<const int8&>::name == "int8");
static_assert(DTypeTraits<volatile complex128&&>::name == "complex128");
