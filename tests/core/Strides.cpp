#include <cassert>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include <stratax.hpp>

using namespace stratax::core;

void test_default_constructor()
{
    Strides strides;

    assert(strides.size() == 0);
    assert(strides.rank() == 0);
    assert(strides.empty());
    assert(strides.data() == nullptr);
    assert(strides.begin() == strides.end());

    bool front_threw = false;
    try {
        strides.front();
    }
    catch (const Exceptions::IndexError&) {
        front_threw = true;
    }

    assert(front_threw);

    bool back_threw = false;
    try {
        strides.back();
    }
    catch (const Exceptions::IndexError&) {
        back_threw = true;
    }

    assert(back_threw);
}

void test_shape_constructor()
{
    Shape shape{2, 3, 4};
    Strides strides(shape);

    assert(strides.size() == 3);
    assert(strides.rank() == 3);
    assert(!strides.empty());

    assert(strides(0) == 12);
    assert(strides(1) == 4);
    assert(strides(2) == 1);

    assert(strides.front() == 12);
    assert(strides.back() == 1);
}

void test_vector_shape_constructor()
{
    Shape shape{5};
    Strides strides(shape);

    assert(strides.size() == 1);
    assert(strides.rank() == 1);
    assert(strides(0) == 1);
}

void test_empty_shape_constructor()
{
    Shape shape;
    Strides strides(shape);

    assert(strides.size() == 0);
    assert(strides.rank() == 0);
    assert(strides.empty());
}

void test_shape_with_one_dimensions()
{
    Strides strides(Shape{1, 3, 1});

    assert(strides.rank() == 3);
    assert(strides(0) == 3);
    assert(strides(1) == 1);
    assert(strides(2) == 1);
}

void test_shape_with_zero_dimension()
{
    Strides strides(Shape{2, 0, 4});

    assert(strides.rank() == 3);
    assert(strides(0) == 0);
    assert(strides(1) == 4);
    assert(strides(2) == 1);
}

void test_shape_overflow_throws()
{
    bool threw = false;

    try {
        Strides strides(Shape{2, std::numeric_limits<std::size_t>::max(), 2});
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    assert(threw);
}

void test_at()
{
    Shape shape{2, 3, 4};
    Strides strides(shape);

    assert(strides.at(0) == 12);
    assert(strides.at(1) == 4);
    assert(strides.at(2) == 1);

    bool threw = false;

    try {
        strides.at(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    assert(threw);
}

void test_iteration()
{
    Shape shape{2, 3, 4};
    Strides strides(shape);

    const std::size_t expected[] = {12, 4, 1};
    std::size_t i = 0;

    for (std::size_t stride : strides) {
        assert(stride == expected[i]);
        ++i;
    }

    assert(i == 3);
}

void test_const_iteration()
{
    const Strides strides(Shape{2, 3, 4});

    const std::size_t expected[] = {12, 4, 1};
    std::size_t i = 0;

    for (auto it = strides.cbegin(); it != strides.cend(); ++it) {
        assert(*it == expected[i]);
        ++i;
    }

    assert(i == 3);
}

void test_reverse_iteration()
{
    Strides strides(Shape{2, 3, 4});

    const std::size_t expected[] = {1, 4, 12};
    std::size_t i = 0;

    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
        assert(*it == expected[i]);
        ++i;
    }

    assert(i == 3);
}

void test_stream_output()
{
    std::ostringstream empty;
    empty << Strides{};
    assert(empty.str() == "()");

    std::ostringstream vector;
    vector << Strides(Shape{5});
    assert(vector.str() == "(1,)");

    std::ostringstream tensor;
    tensor << Strides(Shape{2, 3, 4});
    assert(tensor.str() == "(12, 4, 1)");
}

void test_equality()
{
    Strides a(Shape{2, 3, 4});
    Strides b(Shape{2, 3, 4});
    Strides c(Shape{3, 4});
    Strides empty_a;
    Strides empty_b;

    assert(a == b);
    assert(!(a != b));

    assert(a != c);
    assert(!(a == c));

    assert(empty_a == empty_b);
    assert(!(empty_a != empty_b));
}

void test_copy_constructor()
{
    Strides original(Shape{2, 3, 4});
    Strides copy(original);

    assert(copy == original);
    assert(copy(0) == 12);
    assert(copy(1) == 4);
    assert(copy(2) == 1);
}

void test_copy_assignment()
{
    Strides original(Shape{2, 3, 4});
    Strides copy;

    copy = original;

    assert(copy == original);
    assert(copy.rank() == 3);

    copy = copy;
    assert(copy == original);
    assert(copy.rank() == 3);
}

void test_move_constructor()
{
    Strides original(Shape{2, 3, 4});
    Strides moved(std::move(original));

    assert(moved.rank() == 3);
    assert(moved(0) == 12);
    assert(moved(1) == 4);
    assert(moved(2) == 1);
}

void test_move_assignment()
{
    Strides original(Shape{2, 3, 4});
    Strides moved;

    moved = std::move(original);

    assert(moved.rank() == 3);
    assert(moved(0) == 12);
    assert(moved(1) == 4);
    assert(moved(2) == 1);

    Strides& same = moved;
    moved = std::move(same);
    assert(moved.rank() == 3);
    assert(moved(0) == 12);
    assert(moved(1) == 4);
    assert(moved(2) == 1);
}

void test_swap()
{
    Strides a(Shape{2, 3, 4});
    Strides b(Shape{5, 6});

    a.swap(b);

    assert(a.rank() == 2);
    assert(a(0) == 6);
    assert(a(1) == 1);

    assert(b.rank() == 3);
    assert(b(0) == 12);
    assert(b(1) == 4);
    assert(b(2) == 1);
}

void test_swap_with_empty()
{
    Strides populated(Shape{2, 3, 4});
    Strides empty;

    populated.swap(empty);

    assert(populated.empty());
    assert(empty.rank() == 3);
    assert(empty(0) == 12);
    assert(empty(1) == 4);
    assert(empty(2) == 1);
}

int main()
{
    test_default_constructor();
    test_shape_constructor();
    test_vector_shape_constructor();
    test_empty_shape_constructor();
    test_shape_with_one_dimensions();
    test_shape_with_zero_dimension();
    test_shape_overflow_throws();
    test_at();
    test_iteration();
    test_const_iteration();
    test_reverse_iteration();
    test_stream_output();
    test_equality();
    test_copy_constructor();
    test_copy_assignment();
    test_move_constructor();
    test_move_assignment();
    test_swap();
    test_swap_with_empty();

    return 0;
}
