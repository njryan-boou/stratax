#include <gtest/gtest.h>

#include <stratax.h>

TEST(Slice, construction_and_accessors)
{
    stratax::Slice slice{2, 10, 2};

    EXPECT_EQ(slice.start(), 2);
    EXPECT_EQ(slice.stop(), 10);
    EXPECT_EQ(slice.step(), 2);
    EXPECT_FALSE(slice.empty());
    EXPECT_EQ(slice.size(), 4u);
}

TEST(Slice, zero_step_is_rejected)
{
    EXPECT_THROW(([]() {
        stratax::Slice slice{0, 10, 0};
        (void)slice;
    })(), Exceptions::IndexError);
}

TEST(Slice, positive_step_size_counts)
{
    const stratax::Slice zero_to_zero{0, 0, 1};
    const stratax::Slice zero_to_one{0, 1, 1};
    const stratax::Slice zero_to_five{0, 5, 1};
    const stratax::Slice zero_to_ten_by_two{0, 10, 2};
    const stratax::Slice three_to_ten_by_three{3, 10, 3};
    const stratax::Slice negative_five_to_five_by_two{-5, 5, 2};

    EXPECT_EQ(zero_to_zero.size(), 0u);
    EXPECT_EQ(zero_to_one.size(), 1u);
    EXPECT_EQ(zero_to_five.size(), 5u);
    EXPECT_EQ(zero_to_ten_by_two.size(), 5u);
    EXPECT_EQ(three_to_ten_by_three.size(), 3u);
    EXPECT_EQ(negative_five_to_five_by_two.size(), 5u);
}

TEST(Slice, negative_step_size_counts)
{
    const stratax::Slice zero_to_zero_reverse{0, 0, -1};
    const stratax::Slice five_to_zero_reverse{5, 0, -1};
    const stratax::Slice ten_to_zero_by_two_reverse{10, 0, -2};
    const stratax::Slice nine_to_neg_one_by_two_reverse{9, -1, -2};
    const stratax::Slice ten_to_two_by_three_reverse{10, 2, -3};
    const stratax::Slice negative_one_to_negative_five_by_two_reverse{-1, -5, -2};

    EXPECT_EQ(zero_to_zero_reverse.size(), 0u);
    EXPECT_EQ(five_to_zero_reverse.size(), 5u);
    EXPECT_EQ(ten_to_zero_by_two_reverse.size(), 5u);
    EXPECT_EQ(nine_to_neg_one_by_two_reverse.size(), 5u);
    EXPECT_EQ(ten_to_two_by_three_reverse.size(), 3u);
    EXPECT_EQ(negative_one_to_negative_five_by_two_reverse.size(), 2u);
}

TEST(Slice, empty_and_equality)
{
    const stratax::Slice empty{5, 5, 1};
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.size(), 0u);

    const stratax::Slice a{0, 10, 2};
    const stratax::Slice b{0, 10, 2};
    const stratax::Slice c{0, 10, 3};
    const stratax::Slice d{1, 10, 2};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST(Slice, normalized_ranges_are_not_cached)
{
    stratax::Slice slice{2, 8, 3};

    EXPECT_EQ(slice.start(), 2);
    EXPECT_EQ(slice.stop(), 8);
    EXPECT_EQ(slice.step(), 3);

    slice = stratax::Slice{8, 2, -3};
    EXPECT_EQ(slice.start(), 8);
    EXPECT_EQ(slice.stop(), 2);
    EXPECT_EQ(slice.step(), -3);
    EXPECT_EQ(slice.size(), 2u);
}
