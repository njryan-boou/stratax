#include <cassert>

#include <stratax.hpp>

using stratax::core::Slice;

void test_constructor()
{
    Slice slice(2, 5);

    assert(slice.start() == 2);
    assert(slice.stop() == 5);
    assert(slice.size() == 3);
    assert(!slice.empty());
}

void test_empty_slice()
{
    Slice slice(3, 3);

    assert(slice.start() == 3);
    assert(slice.stop() == 3);
    assert(slice.size() == 0);
    assert(slice.empty());
}

void test_zero_start_slice()
{
    Slice slice(0, 4);

    assert(slice.start() == 0);
    assert(slice.stop() == 4);
    assert(slice.size() == 4);
}

void test_reversed_range_throws()
{
    bool threw = false;

    try {
        Slice slice(5, 2);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    assert(threw);
}

void test_equality()
{
    Slice a(1, 4);
    Slice b(1, 4);
    Slice c(2, 4);
    Slice d(1, 5);

    assert(a == b);
    assert(!(a != b));
    assert(a != c);
    assert(a != d);
}

int main()
{
    test_constructor();
    test_empty_slice();
    test_zero_start_slice();
    test_reversed_range_throws();
    test_equality();

    return 0;
}
