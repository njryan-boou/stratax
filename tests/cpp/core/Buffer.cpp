#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <new>
#include <string>
#include <utility>

#include <stratax.hpp>

using namespace stratax::core;

void test_default_constructor()
{
    Buffer<int> buffer;

    assert(buffer.size() == 0);
    assert(buffer.empty());
    assert(buffer.data() == nullptr);

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

    assert(front_threw);
    assert(back_threw);
}

void test_zero_size_constructor()
{
    Buffer<int> buffer(0);

    assert(buffer.size() == 0);
    assert(buffer.empty());
    assert(buffer.data() == nullptr);
    assert(buffer.begin() == buffer.end());
}

void test_size_constructor()
{
    Buffer<int> buffer(3);

    assert(buffer.size() == 3);
    assert(!buffer.empty());

    buffer[0] = 10;
    buffer[1] = 20;
    buffer[2] = 30;

    assert(buffer.front() == 10);
    assert(buffer.back() == 30);
}

void test_fill_constructor()
{
    Buffer<int> buffer(4, 7);

    assert(buffer.size() == 4);

    for (int value : buffer) {
        assert(value == 7);
    }
}

void test_initializer_list_constructor()
{
    Buffer<int> buffer{1, 2, 3, 4};

    assert(buffer.size() == 4);
    assert(buffer[0] == 1);
    assert(buffer[1] == 2);
    assert(buffer[2] == 3);
    assert(buffer[3] == 4);
}

void test_empty_initializer_list_constructor()
{
    Buffer<int> buffer{};

    assert(buffer.size() == 0);
    assert(buffer.empty());
    assert(buffer.data() == nullptr);
}

void test_iteration()
{
    Buffer<int> buffer{1, 2, 3, 4};

    const int sum = std::accumulate(buffer.begin(), buffer.end(), 0);

    assert(sum == 10);
}

void test_const_and_reverse_iteration()
{
    const Buffer<int> buffer{1, 2, 3};

    int expected = 1;
    for (auto it = buffer.cbegin(); it != buffer.cend(); ++it) {
        assert(*it == expected);
        ++expected;
    }
    assert(expected == 4);

    expected = 3;
    for (auto it = buffer.crbegin(); it != buffer.crend(); ++it) {
        assert(*it == expected);
        --expected;
    }
    assert(expected == 0);
}

void test_copy_constructor()
{
    Buffer<std::string> original{"alpha", "beta", "gamma"};
    Buffer<std::string> copy(original);

    assert(copy.size() == original.size());
    assert(copy[0] == "alpha");
    assert(copy[1] == "beta");
    assert(copy[2] == "gamma");

    copy[0] = "changed";

    assert(original[0] == "alpha");
    assert(copy[0] == "changed");
}

void test_copy_assignment()
{
    Buffer<int> original{1, 2, 3};
    Buffer<int> copy;

    copy = original;

    assert(copy.size() == 3);
    assert(copy[0] == 1);
    assert(copy[1] == 2);
    assert(copy[2] == 3);
}

void test_self_assignment()
{
    Buffer<int> buffer{1, 2, 3};

    buffer = buffer;

    assert(buffer.size() == 3);
    assert(buffer[0] == 1);
    assert(buffer[1] == 2);
    assert(buffer[2] == 3);

    Buffer<int>& same = buffer;
    buffer = std::move(same);

    assert(buffer.size() == 3);
    assert(buffer[0] == 1);
    assert(buffer[1] == 2);
    assert(buffer[2] == 3);
}

void test_move_constructor()
{
    Buffer<int> original{5, 6, 7};
    Buffer<int> moved(std::move(original));

    assert(moved.size() == 3);
    assert(moved[0] == 5);
    assert(moved[1] == 6);
    assert(moved[2] == 7);

    assert(original.size() == 0);
    assert(original.data() == nullptr);
}

void test_move_assignment()
{
    Buffer<int> original{8, 9};
    Buffer<int> moved;

    moved = std::move(original);

    assert(moved.size() == 2);
    assert(moved[0] == 8);
    assert(moved[1] == 9);

    assert(original.size() == 0);
    assert(original.data() == nullptr);
}

void test_fill()
{
    Buffer<int> buffer{1, 2, 3};

    buffer.fill(42);

    for (int value : buffer) {
        assert(value == 42);
    }

    Buffer<int> empty;
    empty.fill(7);

    assert(empty.empty());
}

void test_swap()
{
    Buffer<int> a{1, 2};
    Buffer<int> b{3, 4, 5};

    a.swap(b);

    assert(a.size() == 3);
    assert(a[0] == 3);
    assert(a[1] == 4);
    assert(a[2] == 5);

    assert(b.size() == 2);
    assert(b[0] == 1);
    assert(b[1] == 2);
}

void test_swap_with_empty()
{
    Buffer<int> filled{1, 2};
    Buffer<int> empty;

    filled.swap(empty);

    assert(filled.empty());
    assert(filled.data() == nullptr);
    assert(empty.size() == 2);
    assert(empty[0] == 1);
    assert(empty[1] == 2);
}

void test_alignment()
{
    Buffer<int> buffer(4);

    assert(buffer.data() != nullptr);
    assert(reinterpret_cast<std::uintptr_t>(buffer.data()) % 64 == 0);
}

void test_uninitialized_constructor_write_before_read()
{
    Buffer<int> buffer(3, Buffer<int>::uninitialized);

    buffer[0] = 1;
    buffer[1] = 2;
    buffer[2] = 3;

    assert(buffer[0] == 1);
    assert(buffer[1] == 2);
    assert(buffer[2] == 3);
}

void test_oversized_allocation_throws()
{
    bool threw = false;

    try {
        Buffer<int> buffer(std::numeric_limits<std::size_t>::max());
    }
    catch (const std::bad_array_new_length&) {
        threw = true;
    }

    assert(threw);
}

int main()
{
    test_default_constructor();
    test_zero_size_constructor();
    test_size_constructor();
    test_fill_constructor();
    test_initializer_list_constructor();
    test_empty_initializer_list_constructor();
    test_iteration();
    test_const_and_reverse_iteration();
    test_copy_constructor();
    test_copy_assignment();
    test_self_assignment();
    test_move_constructor();
    test_move_assignment();
    test_fill();
    test_swap();
    test_swap_with_empty();
    test_alignment();
    test_uninitialized_constructor_write_before_read();
    test_oversized_allocation_throws();

    return 0;
}
