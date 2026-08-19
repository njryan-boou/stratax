#include <gtest/gtest.h>

#include <concepts>

#include <stratax/core/DTypes/Types.hpp>

static_assert(std::same_as<stratax::core::index_t, std::size_t>);
static_assert(std::same_as<stratax::core::ssize_t, std::ptrdiff_t>);
static_assert(std::same_as<stratax::dtype::int8, std::int8_t>);
static_assert(std::same_as<stratax::dtype::float64, double>);
static_assert(std::same_as<
	stratax::dtype::complex128,
	std::complex<double>>);

TEST(CoreTypes, DTypesUseDedicatedNamespace)
{
	EXPECT_TRUE((std::same_as<stratax::dtype::bool_, bool>));
}
