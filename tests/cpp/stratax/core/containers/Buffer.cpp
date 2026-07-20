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

// ============================================================================
// Const Method Coverage
// ============================================================================

TEST(CoreBuffer, const_front)
{
    const Buffer<int> buffer{1, 2, 3};

    EXPECT_TRUE(buffer.front() == 1);
}

TEST(CoreBuffer, const_back)
{
    const Buffer<int> buffer{1, 2, 3};

    EXPECT_TRUE(buffer.back() == 3);
}

TEST(CoreBuffer, const_front_throws_on_empty)
{
    const Buffer<int> buffer;

    bool threw = false;
    try {
        buffer.front();
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreBuffer, const_back_throws_on_empty)
{
    const Buffer<int> buffer;

    bool threw = false;
    try {
        buffer.back();
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreBuffer, const_operator_bracket)
{
    const Buffer<int> buffer{10, 20, 30};

    EXPECT_TRUE(buffer[0] == 10);
    EXPECT_TRUE(buffer[1] == 20);
    EXPECT_TRUE(buffer[2] == 30);
}

TEST(CoreBuffer, const_data_pointer)
{
    const Buffer<int> buffer{1, 2, 3};

    const int* ptr = buffer.data();
    EXPECT_TRUE(ptr != nullptr);
    EXPECT_TRUE(*ptr == 1);
}

TEST(CoreBuffer, const_data_pointer_empty)
{
    const Buffer<int> buffer;

    EXPECT_TRUE(buffer.data() == nullptr);
}

// ============================================================================
// Mutable Iterator Coverage
// ============================================================================

TEST(CoreBuffer, mutable_reverse_iteration)
{
    Buffer<int> buffer{1, 2, 3, 4};

    int expected = 4;
    for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
        EXPECT_TRUE(*it == expected);
        --expected;
    }
    EXPECT_TRUE(expected == 0);
}

TEST(CoreBuffer, mutable_reverse_iteration_modify)
{
    Buffer<int> buffer{1, 2, 3};

    for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
        *it *= 10;
    }

    EXPECT_TRUE(buffer[0] == 10);
    EXPECT_TRUE(buffer[1] == 20);
    EXPECT_TRUE(buffer[2] == 30);
}

TEST(CoreBuffer, forward_and_backward_iteration_consistency)
{
    Buffer<int> buffer{5, 6, 7, 8};

    // Forward iteration
    std::vector<int> forward_order;
    for (auto it = buffer.begin(); it != buffer.end(); ++it) {
        forward_order.push_back(*it);
    }

    // Backward iteration
    std::vector<int> backward_order;
    for (auto it = buffer.rbegin(); it != buffer.rend(); ++it) {
        backward_order.push_back(*it);
    }

    // Verify backward is reverse of forward
    EXPECT_TRUE(forward_order.size() == backward_order.size());
    for (std::size_t i = 0; i < forward_order.size(); ++i) {
        EXPECT_TRUE(forward_order[i] == backward_order[backward_order.size() - 1 - i]);
    }
}

// ============================================================================
// Fill Edge Cases
// ============================================================================

TEST(CoreBuffer, fill_on_size_constructed_buffer)
{
    Buffer<int> buffer(3);

    buffer.fill(99);

    EXPECT_TRUE(buffer[0] == 99);
    EXPECT_TRUE(buffer[1] == 99);
    EXPECT_TRUE(buffer[2] == 99);
}

TEST(CoreBuffer, fill_single_element)
{
    Buffer<int> buffer(1);

    buffer.fill(42);

    EXPECT_TRUE(buffer[0] == 42);
}

TEST(CoreBuffer, fill_with_complex_type)
{
    Buffer<std::string> buffer(2);

    buffer.fill("test");

    EXPECT_TRUE(buffer[0] == "test");
    EXPECT_TRUE(buffer[1] == "test");
}

// ============================================================================
// Resource Management
// ============================================================================

TEST(CoreBuffer, move_assignment_destroys_old_contents)
{
    // Verify with string that destructor is called
    Buffer<std::string> source{"a", "b"};
    Buffer<std::string> target{"x", "y", "z"};

    target = std::move(source);

    EXPECT_TRUE(target.size() == 2);
    EXPECT_TRUE(target[0] == "a");
    EXPECT_TRUE(target[1] == "b");

    EXPECT_TRUE(source.size() == 0);
    EXPECT_TRUE(source.data() == nullptr);
}

TEST(CoreBuffer, copy_assignment_creates_independent_copy)
{
    Buffer<std::string> source{"original", "data"};
    Buffer<std::string> target{"old", "content", "here"};

    target = source;

    // Verify target is now a copy
    EXPECT_TRUE(target.size() == 2);
    EXPECT_TRUE(target[0] == "original");
    EXPECT_TRUE(target[1] == "data");

    // Verify independence
    source[0] = "modified";
    EXPECT_TRUE(target[0] == "original");
    EXPECT_TRUE(source[0] == "modified");
}

// ============================================================================
// Initializer List with Complex Types
// ============================================================================

TEST(CoreBuffer, initializer_list_with_strings)
{
    Buffer<std::string> buffer{"hello", "world", "test"};

    EXPECT_TRUE(buffer.size() == 3);
    EXPECT_TRUE(buffer[0] == "hello");
    EXPECT_TRUE(buffer[1] == "world");
    EXPECT_TRUE(buffer[2] == "test");
}

TEST(CoreBuffer, initializer_list_single_element)
{
    Buffer<int> buffer{42};

    EXPECT_TRUE(buffer.size() == 1);
    EXPECT_TRUE(buffer[0] == 42);
}

// ============================================================================
// Bounds Consistency
// ============================================================================

TEST(CoreBuffer, begin_end_consistency)
{
    Buffer<int> buffer{1, 2, 3};

    EXPECT_TRUE(std::distance(buffer.begin(), buffer.end()) == 3);
}

TEST(CoreBuffer, cbegin_cend_consistency)
{
    const Buffer<int> buffer{1, 2, 3};

    EXPECT_TRUE(std::distance(buffer.cbegin(), buffer.cend()) == 3);
}

TEST(CoreBuffer, rbegin_rend_consistency)
{
    Buffer<int> buffer{1, 2, 3};

    EXPECT_TRUE(std::distance(buffer.rbegin(), buffer.rend()) == 3);
}

TEST(CoreBuffer, crbegin_crend_consistency)
{
    const Buffer<int> buffer{1, 2, 3};

    EXPECT_TRUE(std::distance(buffer.crbegin(), buffer.crend()) == 3);
}

