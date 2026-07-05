#include <type_traits>
#include <cstdint>

#include <stratax.hpp>

using namespace stratax::core::dtype;

static_assert(std::is_same_v<int8, std::int8_t>);
static_assert(std::is_same_v<int16, std::int16_t>);
static_assert(std::is_same_v<int32, std::int32_t>);
static_assert(std::is_same_v<int64, std::int64_t>);

static_assert(std::is_same_v<uint8, std::uint8_t>);
static_assert(std::is_same_v<uint16, std::uint16_t>);
static_assert(std::is_same_v<uint32, std::uint32_t>);
static_assert(std::is_same_v<uint64, std::uint64_t>);

static_assert(std::is_same_v<float32, float>);
static_assert(std::is_same_v<float64, double>);
static_assert(std::is_same_v<float128, long double>);

static_assert(std::is_same_v<index_t, std::size_t>);
static_assert(std::is_same_v<ssize_t, std::ptrdiff_t>);

int main() {
    return 0;
}