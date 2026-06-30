#include <cassert>
#include <numeric>
#include <string>
#include <utility>

#include <stratax/core/Buffer.hpp>

using namespace stratax::core;

void test_default_constructor()
{
    Buffer<int> buffer;

    assert(buffer.size() == 0);
    assert(buffer.empty());
    assert(buffer.data() == nullptr);
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

void test_iteration()
{
    Buffer<int> buffer{1, 2, 3, 4};

    const int sum = std::accumulate(buffer.begin(), buffer.end(), 0);

    assert(sum == 10);
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

int main()
{
    test_default_constructor();
    test_size_constructor();
    test_fill_constructor();
    test_initializer_list_constructor();
    test_iteration();
    test_copy_constructor();
    test_copy_assignment();
    test_move_constructor();
    test_move_assignment();
    test_fill();
    test_swap();

    return 0;
}
