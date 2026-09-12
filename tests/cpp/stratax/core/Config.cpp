#include <gtest/gtest.h>
#include <stratax/core/Config.hpp>
#include <stratax/core/Buffer.hpp>

TEST(CoreConfig, DefaultBufferAlignmentMatchesConfiguration)
{
    static_assert(stratax::core::config::default_alignment > 0);
    static_assert((stratax::core::config::default_alignment &
                  (stratax::core::config::default_alignment - 1)) == 0);
    EXPECT_EQ(stratax::core::Buffer<double>::alignment(), stratax::core::config::default_alignment);
}
