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

// TODO: add more tests
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
