#include <gtest/gtest.h>

#include <utility>

#include <stratax.h>

TEST(Buffer, default_and_empty)
{
    stratax::core::Buffer<int> buffer;

    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.begin(), buffer.end());
    EXPECT_EQ(buffer.cbegin(), buffer.cend());
    EXPECT_EQ(buffer.rbegin(), buffer.rend());
    EXPECT_EQ(buffer.crbegin(), buffer.crend());
    EXPECT_THROW(buffer.front(), Exceptions::IndexError);
    EXPECT_THROW(buffer.back(), Exceptions::IndexError);
}

TEST(Buffer, size_and_value_construction)
{
    stratax::core::Buffer<int> zeros(4);
    for (std::size_t i = 0; i < zeros.size(); ++i) {
        EXPECT_EQ(zeros[i], 0);
    }

    stratax::core::Buffer<int> filled(3, 7);
    EXPECT_EQ(filled.size(), 3u);
    EXPECT_FALSE(filled.empty());
    for (std::size_t i = 0; i < filled.size(); ++i) {
        EXPECT_EQ(filled[i], 7);
    }
}

TEST(Buffer, initializer_list_construction_and_access)
{
    stratax::core::Buffer<int> buffer{1, 2, 3, 4, 5};

    EXPECT_EQ(buffer.size(), 5u);
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer[0], 1);
    EXPECT_EQ(buffer[2], 3);
    EXPECT_EQ(buffer[4], 5);
    EXPECT_EQ(buffer.front(), 1);
    EXPECT_EQ(buffer.back(), 5);
    EXPECT_EQ(buffer.data()[1], 2);
    EXPECT_EQ(buffer.data()[3], 4);

    const auto& const_buffer = buffer;
    EXPECT_EQ(const_buffer[1], 2);
    EXPECT_EQ(const_buffer.front(), 1);
    EXPECT_EQ(const_buffer.back(), 5);
}

TEST(Buffer, iterators)
{
    stratax::core::Buffer<int> buffer{10, 20, 30, 40};

    std::size_t index = 0;
    for (auto it = buffer.begin(); it != buffer.end(); ++it) {
        EXPECT_EQ(*it, buffer[index]);
        ++index;
    }
    EXPECT_EQ(index, buffer.size());

    index = 0;
    for (auto it = buffer.cbegin(); it != buffer.cend(); ++it) {
        EXPECT_EQ(*it, buffer[index]);
        ++index;
    }
    EXPECT_EQ(index, buffer.size());

    index = buffer.size();
    for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
        --index;
        EXPECT_EQ(*it, buffer[index]);
    }
    EXPECT_EQ(index, 0u);

    stratax::core::Buffer<int> empty;
    EXPECT_EQ(empty.begin(), empty.end());
    EXPECT_EQ(empty.cbegin(), empty.cend());
    EXPECT_EQ(empty.rbegin(), empty.rend());
    EXPECT_EQ(empty.crbegin(), empty.crend());
}

TEST(Buffer, copy_and_move_operations)
{
    stratax::core::Buffer<int> original{1, 2, 3};

    stratax::core::Buffer<int> copy = original;
    EXPECT_EQ(copy.size(), original.size());
    for (std::size_t i = 0; i < copy.size(); ++i) {
        EXPECT_EQ(copy[i], original[i]);
    }

    stratax::core::Buffer<int> assigned(2, 9);
    assigned = original;
    EXPECT_EQ(assigned.size(), original.size());
    for (std::size_t i = 0; i < assigned.size(); ++i) {
        EXPECT_EQ(assigned[i], original[i]);
    }

    stratax::core::Buffer<int> moved(std::move(original));
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(moved[0], 1);
    EXPECT_EQ(moved[2], 3);

    stratax::core::Buffer<int> moved_assigned(1, 5);
    moved_assigned = std::move(moved);
    EXPECT_EQ(moved_assigned.size(), 3u);
    EXPECT_EQ(moved_assigned[1], 2);
}

TEST(Buffer, fill_and_swap)
{
    stratax::core::Buffer<int> left{1, 2, 3};
    stratax::core::Buffer<int> right{4, 5, 6, 7};

    left.fill(9);
    for (std::size_t i = 0; i < left.size(); ++i) {
        EXPECT_EQ(left[i], 9);
    }

    left.swap(right);
    EXPECT_EQ(left.size(), 4u);
    EXPECT_EQ(left[0], 4);
    EXPECT_EQ(left[3], 7);
    EXPECT_EQ(right.size(), 3u);
    EXPECT_EQ(right[0], 9);
    EXPECT_EQ(right[2], 9);
}
