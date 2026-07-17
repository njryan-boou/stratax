#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <numeric>
#include <new>
#include <string>
#include <utility>

#include <stratax.h>

using namespace stratax::core;

TEST(CoreBuffer, default_constructor)
{
    Buffer<int> buffer;

    EXPECT_TRUE(buffer.size() == 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_TRUE(buffer.data() == nullptr);

    bool front_threw = false;
    bool back_threw = false;

    try {
        buffer.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    try {
        buffer.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    EXPECT_TRUE(front_threw);
    EXPECT_TRUE(back_threw);
}

TEST(CoreBuffer, zero_size_constructor)
{
    Buffer<int> buffer(0);

    EXPECT_TRUE(buffer.size() == 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_TRUE(buffer.data() == nullptr);
    EXPECT_TRUE(buffer.begin() == buffer.end());
}

TEST(CoreBuffer, size_constructor)
{
    Buffer<int> buffer(3);

    EXPECT_TRUE(buffer.size() == 3);
    EXPECT_TRUE(!buffer.empty());

    buffer[0] = 10;
    buffer[1] = 20;
    buffer[2] = 30;

    EXPECT_TRUE(buffer.front() == 10);
    EXPECT_TRUE(buffer.back() == 30);
}

TEST(CoreBuffer, fill_constructor)
{
    Buffer<int> buffer(4, 7);

    EXPECT_TRUE(buffer.size() == 4);

    for (int value : buffer) {
        EXPECT_TRUE(value == 7);
    }
}

TEST(CoreBuffer, initializer_list_constructor)
{
    Buffer<int> buffer{1, 2, 3, 4};

    EXPECT_TRUE(buffer.size() == 4);
    EXPECT_TRUE(buffer[0] == 1);
    EXPECT_TRUE(buffer[1] == 2);
    EXPECT_TRUE(buffer[2] == 3);
    EXPECT_TRUE(buffer[3] == 4);
}

TEST(CoreBuffer, empty_initializer_list_constructor)
{
    Buffer<int> buffer{};

    EXPECT_TRUE(buffer.size() == 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_TRUE(buffer.data() == nullptr);
}

TEST(CoreBuffer, iteration)
{
    Buffer<int> buffer{1, 2, 3, 4};

    const int sum = std::accumulate(buffer.begin(), buffer.end(), 0);

    EXPECT_TRUE(sum == 10);
}

TEST(CoreBuffer, const_and_reverse_iteration)
{
    const Buffer<int> buffer{1, 2, 3};

    int expected = 1;
    for (auto it = buffer.cbegin(); it != buffer.cend(); ++it) {
        EXPECT_TRUE(*it == expected);
        ++expected;
    }
    EXPECT_TRUE(expected == 4);

    expected = 3;
    for (auto it = buffer.crbegin(); it != buffer.crend(); ++it) {
        EXPECT_TRUE(*it == expected);
        --expected;
    }
    EXPECT_TRUE(expected == 0);
}

TEST(CoreBuffer, copy_constructor)
{
    Buffer<std::string> original{"alpha", "beta", "gamma"};
    Buffer<std::string> copy(original);

    EXPECT_TRUE(copy.size() == original.size());
    EXPECT_TRUE(copy[0] == "alpha");
    EXPECT_TRUE(copy[1] == "beta");
    EXPECT_TRUE(copy[2] == "gamma");

    copy[0] = "changed";

    EXPECT_TRUE(original[0] == "alpha");
    EXPECT_TRUE(copy[0] == "changed");
}

TEST(CoreBuffer, copy_assignment)
{
    Buffer<int> original{1, 2, 3};
    Buffer<int> copy;

    copy = original;

    EXPECT_TRUE(copy.size() == 3);
    EXPECT_TRUE(copy[0] == 1);
    EXPECT_TRUE(copy[1] == 2);
    EXPECT_TRUE(copy[2] == 3);
}

TEST(CoreBuffer, self_assignment)
{
    Buffer<int> buffer{1, 2, 3};

    buffer = buffer;

    EXPECT_TRUE(buffer.size() == 3);
    EXPECT_TRUE(buffer[0] == 1);
    EXPECT_TRUE(buffer[1] == 2);
    EXPECT_TRUE(buffer[2] == 3);

    Buffer<int>& same = buffer;
    buffer = std::move(same);

    EXPECT_TRUE(buffer.size() == 3);
    EXPECT_TRUE(buffer[0] == 1);
    EXPECT_TRUE(buffer[1] == 2);
    EXPECT_TRUE(buffer[2] == 3);
}

TEST(CoreBuffer, move_constructor)
{
    Buffer<int> original{5, 6, 7};
    Buffer<int> moved(std::move(original));

    EXPECT_TRUE(moved.size() == 3);
    EXPECT_TRUE(moved[0] == 5);
    EXPECT_TRUE(moved[1] == 6);
    EXPECT_TRUE(moved[2] == 7);

    EXPECT_TRUE(original.size() == 0);
    EXPECT_TRUE(original.data() == nullptr);
}

TEST(CoreBuffer, move_assignment)
{
    Buffer<int> original{8, 9};
    Buffer<int> moved;

    moved = std::move(original);

    EXPECT_TRUE(moved.size() == 2);
    EXPECT_TRUE(moved[0] == 8);
    EXPECT_TRUE(moved[1] == 9);

    EXPECT_TRUE(original.size() == 0);
    EXPECT_TRUE(original.data() == nullptr);
}

TEST(CoreBuffer, fill)
{
    Buffer<int> buffer{1, 2, 3};

    buffer.fill(42);

    for (int value : buffer) {
        EXPECT_TRUE(value == 42);
    }

    Buffer<int> empty;
    empty.fill(7);

    EXPECT_TRUE(empty.empty());
}

TEST(CoreBuffer, swap)
{
    Buffer<int> a{1, 2};
    Buffer<int> b{3, 4, 5};

    a.swap(b);

    EXPECT_TRUE(a.size() == 3);
    EXPECT_TRUE(a[0] == 3);
    EXPECT_TRUE(a[1] == 4);
    EXPECT_TRUE(a[2] == 5);

    EXPECT_TRUE(b.size() == 2);
    EXPECT_TRUE(b[0] == 1);
    EXPECT_TRUE(b[1] == 2);
}

TEST(CoreBuffer, swap_with_empty)
{
    Buffer<int> filled{1, 2};
    Buffer<int> empty;

    filled.swap(empty);

    EXPECT_TRUE(filled.empty());
    EXPECT_TRUE(filled.data() == nullptr);
    EXPECT_TRUE(empty.size() == 2);
    EXPECT_TRUE(empty[0] == 1);
    EXPECT_TRUE(empty[1] == 2);
}

TEST(CoreBuffer, alignment)
{
    Buffer<int> buffer(4);

    EXPECT_TRUE(buffer.data() != nullptr);
    EXPECT_TRUE(reinterpret_cast<std::uintptr_t>(buffer.data()) % 64 == 0);
}

TEST(CoreBuffer, uninitialized_constructor_write_before_read)
{
    Buffer<int> buffer(3, Buffer<int>::uninitialized);

    buffer[0] = 1;
    buffer[1] = 2;
    buffer[2] = 3;

    EXPECT_TRUE(buffer[0] == 1);
    EXPECT_TRUE(buffer[1] == 2);
    EXPECT_TRUE(buffer[2] == 3);
}

TEST(CoreBuffer, oversized_allocation_throws)
{
    bool threw = false;

    try {
        Buffer<int> buffer(std::numeric_limits<std::size_t>::max());
    }
    catch (const std::bad_array_new_length&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

