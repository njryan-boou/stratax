#include <gtest/gtest.h>

#include <stratax.h>

TEST(IndexingNormalize, positive_and_negative_indices)
{
    EXPECT_EQ(stratax::indexing::normalize_index(0, 5), 0u);
    EXPECT_EQ(stratax::indexing::normalize_index(4, 5), 4u);
    EXPECT_EQ(stratax::indexing::normalize_index(-1, 5), 4u);
    EXPECT_EQ(stratax::indexing::normalize_index(-5, 5), 0u);
    EXPECT_EQ(stratax::indexing::normalize_index(-2, 5), 3u);
}

TEST(IndexingNormalize, throws_for_out_of_range)
{
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(5, 5); })(), Exceptions::IndexError);
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(-6, 5); })(), Exceptions::IndexError);
    EXPECT_THROW(([]() { stratax::indexing::normalize_index(10, 3); })(), Exceptions::IndexError);
}

TEST(IndexingOffset, computes_row_major_offset)
{
    const stratax::core::Shape shape{2, 3, 4};
    const stratax::core::Strides strides(shape);

    EXPECT_EQ(stratax::indexing::offset(shape, strides, std::vector<std::size_t>{0, 0, 0}), 0u);
    EXPECT_EQ(stratax::indexing::offset(shape, strides, std::vector<std::size_t>{1, 0, 0}), 12u);
    EXPECT_EQ(stratax::indexing::offset(shape, strides, std::vector<std::size_t>{1, 2, 3}), 23u);
    EXPECT_EQ(stratax::indexing::offset(shape, strides, std::vector<std::size_t>{0, 2, 3}), 11u);
}

TEST(IndexingOffset, handles_empty_shape)
{
    const stratax::core::Shape empty_shape{};
    const stratax::core::Strides empty_strides(empty_shape);

    EXPECT_EQ(stratax::indexing::offset(empty_shape, empty_strides, std::vector<std::size_t>{}), 0u);
}
