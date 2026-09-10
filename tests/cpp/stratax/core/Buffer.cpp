#include <gtest/gtest.h>

#include <utility>

#include <stratax.h>

using namespace stratax::core;

// Constructors

TEST(BufferConstructor, DefaultConstructor)
{
    Buffer<int> buffer;

    EXPECT_EQ(buffer.size(), 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.begin(), buffer.end());
}

TEST(BufferConstructor, SizeConstructor)
{
    Buffer<int> buffer(10);

    EXPECT_EQ(buffer.size(), 10);
    EXPECT_FALSE(buffer.empty());
    EXPECT_NE(buffer.data(), nullptr);

    for (std::size_t i = 0; i < buffer.size(); ++i) 
    {
        EXPECT_EQ(buffer[i], 0);
    }

}

TEST(BufferConstructor, FillConstructor)
{
    Buffer<int> buffer(10, 5);

    EXPECT_EQ(buffer.size(), 10);
    EXPECT_FALSE(buffer.empty());
    EXPECT_NE(buffer.data(), nullptr);

    for (std::size_t i = 0; i < buffer.size(); ++i)
    {
        EXPECT_EQ(buffer[i], 5);
    }
}

TEST(BufferConstructor, ListConstructor)
{
    Buffer<int> buffer{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    EXPECT_EQ(buffer.size(), 10);
    EXPECT_FALSE(buffer.empty());
    EXPECT_NE(buffer.data(), nullptr);

    EXPECT_EQ(buffer[0], 1);
    EXPECT_EQ(buffer[1], 2);
    EXPECT_EQ(buffer[2], 3);
    EXPECT_EQ(buffer[3], 4);
    EXPECT_EQ(buffer[4], 5);
    EXPECT_EQ(buffer[5], 6);
    EXPECT_EQ(buffer[6], 7);
    EXPECT_EQ(buffer[7], 8);
    EXPECT_EQ(buffer[8], 9);
    EXPECT_EQ(buffer[9], 10);
}

TEST(BufferConstructor, ZeroSize)
{
    Buffer<int> sized(0);
    Buffer<int> filled(0, 5);
    Buffer<int> listed(std::initializer_list<int>{});

    EXPECT_TRUE(sized.empty());
    EXPECT_EQ(sized.size(), 0);
    EXPECT_EQ(sized.data(), nullptr);

    EXPECT_TRUE(filled.empty());
    EXPECT_EQ(filled.data(), nullptr);

    EXPECT_TRUE(listed.empty());
    EXPECT_EQ(listed.data(), nullptr);
}