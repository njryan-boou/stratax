#include <gtest/gtest.h>
#include <stratax/core/validation/DimensionValidation.hpp>

TEST(DimensionValidation, AcceptsMatchingAndNonnegativeDimensions)
{
    EXPECT_NO_THROW(stratax::validation::require_nonnegative_dimension(0));
    EXPECT_NO_THROW(stratax::validation::require_dimension(2, 2));
}

TEST(DimensionValidation, FailureCarriesMetadataAndGeneratedMessage)
{
    try
    {
        stratax::validation::require_nonnegative_dimension(-2);
        FAIL() << "Expected Exceptions::DimensionError";
    }
    catch (const Exceptions::DimensionError& error)
    {
        EXPECT_STREQ(
            error.what(),
            "Dimension -2 does not match expected dimension 0.");
        EXPECT_EQ(error.actual_dimension(), -2);
        EXPECT_EQ(error.expected_dimension(), 0);
    }
}
