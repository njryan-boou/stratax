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

TEST(CoreSlice, reversed_range_throws)
{
    bool threw = false;

    try {
        Slice slice(5, 2);
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

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(a != d);
}

