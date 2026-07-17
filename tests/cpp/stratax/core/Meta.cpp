#include <gtest/gtest.h>

#include <string_view>

#include <stratax/core/Meta.hpp>

TEST(CoreMeta, metadata_macros_are_non_empty)
{
    EXPECT_TRUE(std::string_view(STRATAX_VERSION).size() > 0);
    EXPECT_TRUE(std::string_view(STRATAX_PY_DOC).size() > 0);
    EXPECT_TRUE(std::string_view(STRATAX_AUTHOR).size() > 0);
}
