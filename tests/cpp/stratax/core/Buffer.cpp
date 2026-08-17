#include <gtest/gtest.h>

#include <utility>

#include <stratax.h>

using namespace stratax::core;

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
    Buffer<int> buffer{1, 2, 3, 4, 5};
    
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

TEST(BufferConstructor, MoveConstructor)
{
    Buffer<int> source{1, 2, 3};

    auto* original_data = source.data();

    Buffer<int> destination(std::move(source));

    EXPECT_EQ(destination.size(), 3);
    EXPECT_EQ(destination.data(), original_data);
    EXPECT_EQ(source.size(), 0);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_EQ(source.empty(), true);
}

TEST(BufferAssignment, CopyAssignment)
{
    Buffer<int> source{1, 2, 3};
    Buffer<int> destination{4, 5};

    destination = source;

    EXPECT_EQ(destination.size(), source.size());

    EXPECT_NE(destination.data(), source.data());

    for (std::size_t i = 0; i < source.size(); ++i) {
        EXPECT_EQ(destination[i], source[i]);
    }

    destination[0] = 100;

    EXPECT_EQ(destination[0], 100);
    EXPECT_EQ(source[0], 1);
}

TEST(BufferAssignment, SelfCopyAssignment)
{
    Buffer<int> buffer{1, 2, 3};

    buffer = buffer;

    EXPECT_EQ(buffer.size(), 3);
}

TEST(BufferAssignment, MoveAssignment)
{
    Buffer<int> source{1, 2, 3};
    Buffer<int> destination{4, 5};

    auto* source_data = source.data();

    destination = std::move(source);

    EXPECT_EQ(destination.data(), source_data);
    EXPECT_EQ(destination.size(), 3);

    EXPECT_EQ(destination[0], 1);
    EXPECT_EQ(destination[1], 2);
    EXPECT_EQ(destination[2], 3);

    EXPECT_EQ(source.size(), 0);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_TRUE(source.empty());
}

TEST(BufferAssignment, SelfMove)
{
    Buffer<int> buffer{1, 2, 3};

    auto* original_data = buffer.data();

    buffer = std::move(buffer);

    EXPECT_EQ(buffer.size(), 3);
    EXPECT_EQ(buffer.data(), original_data);

    EXPECT_EQ(buffer[0], 1);
    EXPECT_EQ(buffer[1], 2);
    EXPECT_EQ(buffer[2], 3);
}