#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

#include <stratax/core/Config.hpp>

TEST(CoreConfig, constants_have_expected_types)
{
    static_assert(std::is_same_v<decltype(stratax::core::config::default_alignment), const std::size_t>);
    static_assert(std::is_same_v<decltype(stratax::core::config::bounds_checking), const bool>);
    static_assert(std::is_same_v<decltype(stratax::core::config::use_simd), const bool>);
    static_assert(std::is_same_v<decltype(stratax::core::config::use_multithreading), const bool>);
    static_assert(std::is_same_v<decltype(stratax::core::config::use_cuda), const bool>);
}

TEST(CoreConfig, default_alignment_is_nonzero)
{
    EXPECT_TRUE(stratax::core::config::default_alignment > 0);
}