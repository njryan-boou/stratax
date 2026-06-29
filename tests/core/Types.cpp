#include <type_traits>
#include <cstdint>

#include <stratax/core/Types.hpp>

static_assert(std::is_same_v<stratax::core::dtype::int8, std::int8_t>);
static_assert(std::is_same_v<stratax::core::dtype::int16, std::int16_t>);
static_assert(std::is_same_v<stratax::core::dtype::int32, std::int32_t>);
static_assert(std::is_same_v<stratax::core::dtype::int64, std::int64_t>);

static_assert(std::is_same_v<stratax::core::dtype::uint8, std::uint8_t>);
static_assert(std::is_same_v<stratax::core::dtype::uint16, std::uint16_t>);
static_assert(std::is_same_v<stratax::core::dtype::uint32, std::uint32_t>);
static_assert(std::is_same_v<stratax::core::dtype::uint64, std::uint64_t>);

static_assert(std::is_same_v<stratax::core::dtype::float32, float>);
static_assert(std::is_same_v<stratax::core::dtype::float64, double>);
static_assert(std::is_same_v<stratax::core::dtype::float128, long double>);

static_assert(std::is_same_v<stratax::core::dtype::index_t, std::size_t>);
static_assert(std::is_same_v<stratax::core::dtype::ssize_t, std::ptrdiff_t>);

int main() {
    return 0;
}