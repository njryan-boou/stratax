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

TEST(BufferConstructor, List)
{
    Buffer<int> buffer{1, 2, 3, 4, 5};
    
    EXPECT_EQ(buffer.size(), 5);
    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_FALSE(buffer.empty());

    for (std::size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_EQ(buffer[i], i + 1);
    }
}

TEST(BufferConstructor, Copy)
{
    Buffer<int> original{1, 2, 3};
    Buffer<int> copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_NE(copy.data(), original.data());

    copy[0] = 100;
    EXPECT_EQ(copy[0], 100);
    EXPECT_EQ(original[0], 1);
}

TEST(BufferConstructor, Move)
{
    Buffer<int> source{1, 2, 3};

    auto* original_data = source.data();

    Buffer<int> destination(std::move(source));

    EXPECT_EQ(destination.size(), 3);
    EXPECT_EQ(destination.data(), original_data);
    EXPECT_EQ(destination[0], 1);
    EXPECT_EQ(destination[1], 2);
    EXPECT_EQ(destination[2], 3);
    EXPECT_EQ(source.size(), 0);
    EXPECT_EQ(source.data(), nullptr);
    EXPECT_TRUE(source.empty());
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

    auto* data = buffer.data();

    buffer = buffer;

    EXPECT_EQ(buffer.data(), data);
    EXPECT_EQ(buffer[0], 1);
    EXPECT_EQ(buffer[1], 2);
    EXPECT_EQ(buffer[2], 3);
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

TEST(BufferAssignment, SelfMoveAssignment)
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

TEST(BufferMeta, Front)
{
    Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_EQ(buffer.front(), 1);

    buffer.front() = 10;
    EXPECT_EQ(buffer[0], 10);

    static_assert(
        std::same_as<
            decltype(std::declval<Buffer<int>&>().front()),
            int&
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().front()),
            const int&
        >
    );
}

TEST(BufferMeta, Back)
{
    Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_EQ(buffer.back(), 4);

    buffer.back() = 10;
    EXPECT_EQ(buffer[3], 10);

    static_assert(
        std::same_as<
            decltype(std::declval<Buffer<int>&>().back()),
            int&
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().back()),
            const int&
        >
    );
}

TEST(BufferMeta, FrontEmpty)
{
    Buffer<int> buffer;

    EXPECT_THROW(buffer.front(), Exceptions::IndexError);
}

TEST(BufferMeta, BackEmpty)
{
    Buffer<int> buffer;

    EXPECT_THROW(buffer.back(), Exceptions::IndexError);
}

TEST(BufferMeta, Data)
{
    Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_NE(buffer.data(), nullptr);
    EXPECT_EQ(buffer.data()[0], 1);

    static_assert(
        std::same_as<
            decltype(std::declval<Buffer<int>&>().data()),
            int*
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().data()),
            const int*
        >
    );
}

TEST(BufferIterator, BeginEnd)
{
    Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_EQ(*buffer.begin(), 1);
    EXPECT_EQ(buffer.end() - buffer.begin(), 4);

    static_assert(
        std::same_as<
            decltype(std::declval<Buffer<int>&>().begin()),
            Buffer<int>::iterator
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<Buffer<int>&>().end()),
            Buffer<int>::iterator
        >
    );
}

TEST(BufferIterator, ConstBeginEnd)
{
    const Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_EQ(*buffer.begin(), 1);
    EXPECT_EQ(buffer.end() - buffer.begin(), 4);

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().begin()),
            Buffer<int>::const_iterator
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().end()),
            Buffer<int>::const_iterator
        >
    );
}

TEST(BufferIterator, CBeginCEnd)
{
    const Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_EQ(*buffer.cbegin(), 1);
    EXPECT_EQ(buffer.cend() - buffer.cbegin(), 4);

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().cbegin()),
            Buffer<int>::const_iterator
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().cend()),
            Buffer<int>::const_iterator
        >
    );
}

TEST(BufferIterator, Mutable)
{
    Buffer<int> buffer{1, 2, 3, 4};

    *buffer.begin() = 10;

    EXPECT_EQ(buffer.front(), 10);
}

TEST(BufferIterator, Traversal)
{
    Buffer<int> buffer{1, 2, 3, 4};

    int expected = 1;

    for (auto it = buffer.begin(); it != buffer.end(); ++it) {
        EXPECT_EQ(*it, expected++);
    }
}

TEST(BufferIterator, Reverse)
{
    Buffer<int> buffer{1, 2, 3, 4};

    int expected = 4;

    for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
        EXPECT_EQ(*it, expected--);
    }

    static_assert(
        std::same_as<
            decltype(std::declval<Buffer<int>&>().rbegin()),
            Buffer<int>::reverse_iterator
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<Buffer<int>&>().rend()),
            Buffer<int>::reverse_iterator
        >
    );
}

TEST(BufferIterator, ConstReverse)
{
    const Buffer<int> buffer{1, 2, 3, 4};

    int expected = 4;

    for (auto it = buffer.crbegin(); it != buffer.crend(); ++it) {
        EXPECT_EQ(*it, expected--);
    }

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().crbegin()),
            Buffer<int>::const_reverse_iterator
        >
    );

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().crend()),
            Buffer<int>::const_reverse_iterator
        >
    );
}

TEST(BufferIterator, Empty)
{
    Buffer<int> buffer;

    EXPECT_EQ(buffer.begin(), buffer.end());
    EXPECT_EQ(buffer.cbegin(), buffer.cend());
    EXPECT_EQ(buffer.rbegin(), buffer.rend());
    EXPECT_EQ(buffer.crbegin(), buffer.crend());
}

TEST(BufferMeta, Size)
{
    Buffer<int> empty_buffer;
    Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_EQ(empty_buffer.size(), 0);
    EXPECT_EQ(buffer.size(), 4);

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().size()),
            Buffer<int>::size_type
        >
    );
}

TEST(BufferMeta, Empty)
{
    Buffer<int> empty_buffer;
    Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_TRUE(empty_buffer.empty());
    EXPECT_FALSE(buffer.empty());

    static_assert(
        std::same_as<
            decltype(std::declval<const Buffer<int>&>().empty()),
            bool
        >
    );
}

TEST(BufferModifier, Fill)
{
    Buffer<int> buffer{1, 2, 3, 4};

    buffer.fill(10);

    for (const auto& value : buffer) {
        EXPECT_EQ(value, 10);
    }

    EXPECT_EQ(buffer.size(), 4);
}

TEST(BufferModifier, Swap)
{
    Buffer<int> lhs{1, 2, 3};
    Buffer<int> rhs{4, 5};

    auto* lhs_data = lhs.data();
    auto* rhs_data = rhs.data();

    lhs.swap(rhs);

    EXPECT_EQ(lhs.size(), 2);
    EXPECT_EQ(rhs.size(), 3);

    EXPECT_EQ(lhs.data(), rhs_data);
    EXPECT_EQ(rhs.data(), lhs_data);

    EXPECT_EQ(lhs[0], 4);
    EXPECT_EQ(lhs[1], 5);

    EXPECT_EQ(rhs[0], 1);
    EXPECT_EQ(rhs[1], 2);
    EXPECT_EQ(rhs[2], 3);
}

TEST(BufferModifier, SwapWithEmpty)
{
    Buffer<int> lhs{1, 2, 3};
    Buffer<int> rhs;

    auto* data = lhs.data();

    lhs.swap(rhs);

    EXPECT_TRUE(lhs.empty());

    EXPECT_EQ(rhs.size(), 3);
    EXPECT_EQ(rhs.data(), data);
}

TEST(BufferModifier, SelfSwap)
{
    Buffer<int> buffer{1, 2, 3};

    auto* data = buffer.data();

    buffer.swap(buffer);

    EXPECT_EQ(buffer.size(), 3);
    EXPECT_EQ(buffer.data(), data);
    EXPECT_EQ(buffer[0], 1);
    EXPECT_EQ(buffer[1], 2);
    EXPECT_EQ(buffer[2], 3);
}

TEST(BufferMeta, Alignment)
{
    static_assert(Buffer<int, 128>::alignment() == 128);

    static_assert(
        Buffer<int>::alignment() >= alignof(int)
    );

    static_assert(
        (Buffer<int>::alignment() &
        (Buffer<int>::alignment() - 1)) == 0
    );

    static_assert(
        noexcept(Buffer<int>::alignment())
    );
}

TEST(BufferMeta, MaxSize)
{
    static_assert(
        Buffer<int>::max_size() > 0
    );

    static_assert(
        Buffer<int>::max_size() ==
        std::numeric_limits<Buffer<int>::size_type>::max() / sizeof(int)
    );

    static_assert(
        noexcept(Buffer<int>::max_size())
    );
}