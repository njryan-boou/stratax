#include <gtest/gtest.h>

#include <stratax/indexing/Normalize.hpp>

TEST(NormalizeIndex, positives_and_zero)
{
    EXPECT_EQ(stratax::indexing::normalize_index(0, 5), 0u);
    EXPECT_EQ(stratax::indexing::normalize_index(1, 5), 1u);
    EXPECT_EQ(stratax::indexing::normalize_index(4, 5), 4u);
    EXPECT_EQ(stratax::indexing::normalize_index(2, 7), 2u);
}

TEST(NormalizeIndex, negatives_wrap_from_end)
{
    EXPECT_EQ(stratax::indexing::normalize_index(-1, 5), 4u);
    EXPECT_EQ(stratax::indexing::normalize_index(-2, 5), 3u);
    EXPECT_EQ(stratax::indexing::normalize_index(-5, 5), 0u);
    EXPECT_EQ(stratax::indexing::normalize_index(-7, 9), 2u);
}

TEST(NormalizeIndex, throws_for_out_of_bounds)
{
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(5, 5); })(), Exceptions::IndexError);
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(-6, 5); })(), Exceptions::IndexError);
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(10, 3); })(), Exceptions::IndexError);
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(-10, 3); })(), Exceptions::IndexError);
}

TEST(NormalizeIndex, size_zero_cases)
{
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(0, 0); })(), Exceptions::IndexError);
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(-1, 0); })(), Exceptions::IndexError);
}
