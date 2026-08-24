#include <gtest/gtest.h>
#include <stratax/core/validation/IndexValidation.hpp>

#include <limits>

TEST(IndexValidation, AcceptsPositiveAndNegativeIndices)
{
    EXPECT_TRUE(stratax::validation::valid_index(0, 3));
    EXPECT_TRUE(stratax::validation::valid_index(2, 3));
    EXPECT_TRUE(stratax::validation::valid_index(-1, 3));
    EXPECT_TRUE(stratax::validation::valid_index(-3, 3));
}

TEST(IndexValidation, RejectsIndicesOutsideEitherBoundary)
{
    EXPECT_FALSE(stratax::validation::valid_index(3, 3));
    EXPECT_FALSE(stratax::validation::valid_index(-4, 3));
    EXPECT_FALSE(stratax::validation::valid_index(0, 0));
    EXPECT_FALSE(stratax::validation::valid_index(
        std::numeric_limits<std::ptrdiff_t>::min(),
        3));
}

TEST(IndexValidation, NormalizesNegativeIndices)
{
    EXPECT_EQ(stratax::validation::normalize_index(0, 3), 0);
    EXPECT_EQ(stratax::validation::normalize_index(-1, 3), 2);
    EXPECT_EQ(stratax::validation::normalize_index(-3, 3), 0);
}

TEST(IndexValidation, FailureCarriesRejectedIndexAndSize)
{
    try
    {
        static_cast<void>(stratax::validation::normalize_index(-4, 3));
        FAIL() << "Expected Exceptions::IndexError";
    }
    catch (const Exceptions::IndexError& error)
    {
        EXPECT_TRUE(error.has_index_metadata());
        EXPECT_EQ(error.index(), -4);
        EXPECT_EQ(error.size(), 3);
    }
}
