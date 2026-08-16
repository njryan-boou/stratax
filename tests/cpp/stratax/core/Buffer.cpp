#include <gtest/gtest.h>

#include <utility>

#include <stratax.h>

using namespace stratax::core;

// Constructor Tests

TEST(BufferConstructor, DefaultConstructor)
{
    Buffer<int> buffer;

    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_TRUE(buffer.empty());
}

TEST(BufferConstructor, Size)
{
    Buffer<int> buffer(10);

    EXPECT_EQ(buffer.size(), 10);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_FALSE(buffer.empty());

    for (std::size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_EQ(buffer[i], 0);
    }
}

TEST(BufferConstructor, ListConstructor)
{
    Buffer<int> buffer({1, 2, 3, 4, 5});
    
    EXPECT_EQ(buffer.size(), 5);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_FALSE(buffer.empty());

    for (std::size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_EQ(buffer[i], i + 1);
    }
}

TEST(BufferConstructor, CopyConstructor)
{
    Buffer<int> original{1, 2, 3};
    Buffer<int> copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_NE(copy.data(), original.data());

    copy[0] = 100;
    EXPECT_EQ(copy[0], 100);
    EXPECT_EQ(original[0], 1);

}