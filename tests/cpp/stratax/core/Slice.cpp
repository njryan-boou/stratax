#include <gtest/gtest.h>

#include <stratax.h>

using stratax::core::Slice;

TEST(CoreSlice, constructor)
{
    Slice slice(2, 5);

    EXPECT_TRUE(slice.start() == 2);
    EXPECT_TRUE(slice.stop() == 5);
    EXPECT_TRUE(slice.size() == 3);
    EXPECT_TRUE(!slice.empty());
    EXPECT_TRUE(slice.step() == 1);
}

TEST(CoreSlice, stepped_constructor)
{
    Slice slice(1, 8, 2);

    EXPECT_TRUE(slice.start() == 1);
    EXPECT_TRUE(slice.stop() == 8);
    EXPECT_TRUE(slice.step() == 2);
    EXPECT_TRUE(slice.size() == 4);
    EXPECT_TRUE(!slice.empty());
}

TEST(CoreSlice, empty_slice)
{
    Slice slice(3, 3);

    EXPECT_TRUE(slice.start() == 3);
    EXPECT_TRUE(slice.stop() == 3);
    EXPECT_TRUE(slice.size() == 0);
    EXPECT_TRUE(slice.empty());
}

TEST(CoreSlice, zero_start_slice)
{
    Slice slice(0, 4);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 4);
    EXPECT_TRUE(slice.size() == 4);
}

TEST(CoreSlice, reversed_range_is_empty_for_positive_step)
{
    Slice slice(5, 2);

    EXPECT_TRUE(slice.start() == 5);
    EXPECT_TRUE(slice.stop() == 2);
    EXPECT_TRUE(slice.step() == 1);
    EXPECT_TRUE(slice.size() == 0);
    EXPECT_TRUE(slice.empty());
}

TEST(CoreSlice, reversed_range_with_negative_step)
{
    Slice slice(5, 2, -1);

    EXPECT_TRUE(slice.start() == 5);
    EXPECT_TRUE(slice.stop() == 2);
    EXPECT_TRUE(slice.step() == -1);
    EXPECT_TRUE(slice.size() == 3);
    EXPECT_TRUE(!slice.empty());
}

TEST(CoreSlice, zero_step_throws)
{
    bool threw = false;

    try {
        Slice slice(0, 3, 0);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreSlice, equality)
{
    Slice a(1, 4);
    Slice b(1, 4);
    Slice c(2, 4);
    Slice d(1, 5);
    Slice e(1, 4, 2);

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(a != d);
    EXPECT_TRUE(a != e);
}

TEST(CoreSlice, single_element_slice)
{
    Slice slice(5, 6);

    EXPECT_TRUE(slice.start() == 5);
    EXPECT_TRUE(slice.stop() == 6);
    EXPECT_TRUE(slice.size() == 1);
    EXPECT_TRUE(!slice.empty());
    EXPECT_TRUE(slice.step() == 1);
}

TEST(CoreSlice, large_range_slice)
{
    Slice slice(0, 1000);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 1000);
    EXPECT_TRUE(slice.size() == 1000);
}

TEST(CoreSlice, step_2)
{
    Slice slice(0, 10, 2);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 10);
    EXPECT_TRUE(slice.step() == 2);
    EXPECT_TRUE(slice.size() == 5);
}

TEST(CoreSlice, step_3)
{
    Slice slice(0, 12, 3);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 12);
    EXPECT_TRUE(slice.step() == 3);
    EXPECT_TRUE(slice.size() == 4);
}

TEST(CoreSlice, step_5_partial)
{
    Slice slice(0, 20, 5);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 20);
    EXPECT_TRUE(slice.step() == 5);
    EXPECT_TRUE(slice.size() == 4);
}

TEST(CoreSlice, negative_step_basic)
{
    Slice slice(5, 0, -1);

    EXPECT_TRUE(slice.start() == 5);
    EXPECT_TRUE(slice.stop() == 0);
    EXPECT_TRUE(slice.step() == -1);
    EXPECT_TRUE(slice.size() == 5);
}

TEST(CoreSlice, negative_step_2)
{
    Slice slice(10, 0, -2);

    EXPECT_TRUE(slice.start() == 10);
    EXPECT_TRUE(slice.stop() == 0);
    EXPECT_TRUE(slice.step() == -2);
    EXPECT_TRUE(slice.size() == 5);
}

TEST(CoreSlice, negative_step_3)
{
    Slice slice(20, 0, -3);

    EXPECT_TRUE(slice.start() == 20);
    EXPECT_TRUE(slice.stop() == 0);
    EXPECT_TRUE(slice.step() == -3);
    EXPECT_TRUE(slice.size() == 7);
}

TEST(CoreSlice, negative_indices_positive_step)
{
    Slice slice(-5, -1);

    EXPECT_TRUE(slice.start() == -5);
    EXPECT_TRUE(slice.stop() == -1);
    EXPECT_TRUE(slice.step() == 1);
    EXPECT_TRUE(slice.size() == 4);
}

TEST(CoreSlice, negative_indices_negative_step)
{
    Slice slice(-1, -5, -1);

    EXPECT_TRUE(slice.start() == -1);
    EXPECT_TRUE(slice.stop() == -5);
    EXPECT_TRUE(slice.step() == -1);
    EXPECT_TRUE(slice.size() == 4);
}

TEST(CoreSlice, mixed_negative_positive_indices)
{
    Slice slice(-3, 5);

    EXPECT_TRUE(slice.start() == -3);
    EXPECT_TRUE(slice.stop() == 5);
    EXPECT_TRUE(slice.step() == 1);
    EXPECT_TRUE(slice.size() == 8);
}

TEST(CoreSlice, large_step)
{
    Slice slice(0, 100, 10);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 100);
    EXPECT_TRUE(slice.step() == 10);
    EXPECT_TRUE(slice.size() == 10);
}

TEST(CoreSlice, large_negative_step)
{
    Slice slice(100, 0, -10);

    EXPECT_TRUE(slice.start() == 100);
    EXPECT_TRUE(slice.stop() == 0);
    EXPECT_TRUE(slice.step() == -10);
    EXPECT_TRUE(slice.size() == 10);
}

TEST(CoreSlice, single_element_with_step)
{
    Slice slice(5, 6, 10);

    EXPECT_TRUE(slice.start() == 5);
    EXPECT_TRUE(slice.stop() == 6);
    EXPECT_TRUE(slice.step() == 10);
    EXPECT_TRUE(slice.size() == 1);
}

TEST(CoreSlice, off_by_one_with_step_2)
{
    Slice slice(0, 11, 2);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 11);
    EXPECT_TRUE(slice.step() == 2);
    EXPECT_TRUE(slice.size() == 6);
}

TEST(CoreSlice, off_by_one_with_step_3)
{
    Slice slice(0, 10, 3);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 10);
    EXPECT_TRUE(slice.step() == 3);
    EXPECT_TRUE(slice.size() == 4);
}

TEST(CoreSlice, start_greater_than_stop_positive_step)
{
    Slice slice(10, 5);

    EXPECT_TRUE(slice.start() == 10);
    EXPECT_TRUE(slice.stop() == 5);
    EXPECT_TRUE(slice.step() == 1);
    EXPECT_TRUE(slice.size() == 0);
    EXPECT_TRUE(slice.empty());
}

TEST(CoreSlice, start_equals_stop_with_step)
{
    Slice slice(5, 5, 2);

    EXPECT_TRUE(slice.start() == 5);
    EXPECT_TRUE(slice.stop() == 5);
    EXPECT_TRUE(slice.step() == 2);
    EXPECT_TRUE(slice.size() == 0);
    EXPECT_TRUE(slice.empty());
}

TEST(CoreSlice, negative_step_start_less_than_stop)
{
    Slice slice(2, 5, -1);

    EXPECT_TRUE(slice.start() == 2);
    EXPECT_TRUE(slice.stop() == 5);
    EXPECT_TRUE(slice.step() == -1);
    EXPECT_TRUE(slice.size() == 0);
    EXPECT_TRUE(slice.empty());
}

TEST(CoreSlice, zero_start_and_stop)
{
    Slice slice(0, 0);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 0);
    EXPECT_TRUE(slice.size() == 0);
    EXPECT_TRUE(slice.empty());
}

TEST(CoreSlice, equality_with_step)
{
    Slice a(1, 10, 2);
    Slice b(1, 10, 2);
    Slice c(1, 10, 3);

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
    EXPECT_TRUE(a != c);
}

TEST(CoreSlice, equality_start_mismatch)
{
    Slice a(0, 5);
    Slice b(1, 5);

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(!(a == b));
}

TEST(CoreSlice, equality_stop_mismatch)
{
    Slice a(0, 5);
    Slice b(0, 6);

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(!(a == b));
}

TEST(CoreSlice, equality_step_mismatch)
{
    Slice a(0, 10, 1);
    Slice b(0, 10, 2);

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(!(a == b));
}

TEST(CoreSlice, large_negative_indices)
{
    Slice slice(-1000, -500);

    EXPECT_TRUE(slice.start() == -1000);
    EXPECT_TRUE(slice.stop() == -500);
    EXPECT_TRUE(slice.size() == 500);
}

TEST(CoreSlice, step_larger_than_range)
{
    Slice slice(0, 5, 10);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 5);
    EXPECT_TRUE(slice.step() == 10);
    EXPECT_TRUE(slice.size() == 1);
}

TEST(CoreSlice, step_larger_than_range_negative)
{
    Slice slice(5, 0, -10);

    EXPECT_TRUE(slice.start() == 5);
    EXPECT_TRUE(slice.stop() == 0);
    EXPECT_TRUE(slice.step() == -10);
    EXPECT_TRUE(slice.size() == 1);
}

TEST(CoreSlice, step_exactly_equals_range)
{
    Slice slice(0, 10, 10);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 10);
    EXPECT_TRUE(slice.step() == 10);
    EXPECT_TRUE(slice.size() == 1);
}

TEST(CoreSlice, size_calculation_remainder_1)
{
    Slice slice(0, 11, 2);

    EXPECT_TRUE(slice.size() == 6);
}

TEST(CoreSlice, size_calculation_remainder_2)
{
    Slice slice(0, 13, 3);

    EXPECT_TRUE(slice.size() == 5);
}

TEST(CoreSlice, size_calculation_negative_step_remainder)
{
    Slice slice(11, 0, -2);

    EXPECT_TRUE(slice.size() == 6);
}

TEST(CoreSlice, zero_start_large_stop)
{
    Slice slice(0, 1000000);

    EXPECT_TRUE(slice.start() == 0);
    EXPECT_TRUE(slice.stop() == 1000000);
    EXPECT_TRUE(slice.size() == 1000000);
}

TEST(CoreSlice, large_start_large_stop_same)
{
    Slice slice(999999, 999999);

    EXPECT_TRUE(slice.start() == 999999);
    EXPECT_TRUE(slice.stop() == 999999);
    EXPECT_TRUE(slice.size() == 0);
    EXPECT_TRUE(slice.empty());
}

TEST(CoreSlice, single_negative_index)
{
    Slice slice(-1, 0);

    EXPECT_TRUE(slice.start() == -1);
    EXPECT_TRUE(slice.stop() == 0);
    EXPECT_TRUE(slice.size() == 1);
}

TEST(CoreSlice, multiple_equality_checks)
{
    Slice a(1, 5, 2);
    Slice b(1, 5, 2);
    Slice c(1, 5, 2);

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(b == c);
    EXPECT_TRUE(a == c);
}

TEST(CoreSlice, inequality_transitivity)
{
    Slice a(1, 5);
    Slice b(1, 6);
    Slice c(2, 6);

    EXPECT_TRUE(a != b);
    EXPECT_TRUE(b != c);
    EXPECT_TRUE(a != c);
}

