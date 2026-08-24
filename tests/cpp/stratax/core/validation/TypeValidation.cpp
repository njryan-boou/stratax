#include <gtest/gtest.h>
#include <stratax/core/validation/TypeValidation.hpp>

TEST(TypeValidation, AcceptsMatchAndDescribesMismatch)
{
    EXPECT_NO_THROW(stratax::validation::require_type(true, "int32", "numeric"));

    try
    {
        stratax::validation::require_type(false, "string", "numeric");
        FAIL() << "Expected Exceptions::TypeError";
    }
    catch (const Exceptions::TypeError& error)
    {
        EXPECT_STREQ(
            error.what(),
            "Type 'string' does not match expected type 'numeric'.");
        EXPECT_EQ(error.actual_type(), "string");
        EXPECT_EQ(error.expected_type(), "numeric");
    }
}
