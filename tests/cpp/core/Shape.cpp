#include <cassert>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <stratax.hpp>

using namespace stratax::core;

void test_default_constructor()
{
    Shape shape;

    assert(shape.rank() == 0);
    assert(shape.empty());
    assert(shape.elements() == 0);
    assert(shape.begin() == shape.end());
}

void test_initializer_list_constructor()
{
    Shape shape{3, 224, 224};

    assert(shape.rank() == 3);
    assert(shape.elements() == 150528);
    assert(shape(0) == 3);
    assert(shape(1) == 224);
    assert(shape(2) == 224);
}

void test_zero_dimension_elements()
{
    Shape scalar_zero{0};
    Shape matrix_zero{0, 3};
    Shape later_zero{std::numeric_limits<std::size_t>::max(), 0};

    assert(scalar_zero.rank() == 1);
    assert(scalar_zero.elements() == 0);
    assert(matrix_zero.rank() == 2);
    assert(matrix_zero.elements() == 0);
    assert(later_zero.rank() == 2);
    assert(later_zero.elements() == 0);
}

void test_elements_overflow_throws()
{
    Shape shape{2, std::numeric_limits<std::size_t>::max(), 2};

    bool threw = false;
    try {
        [[maybe_unused]] const std::size_t elements = shape.elements();
    }
    catch (const Exceptions::DimensionError&) {
        threw = true;
    }

    assert(threw);
}

void test_buffer_constructors()
{
    Buffer<std::size_t> dims{2, 3, 4};
    Shape copied(dims);

    assert(copied.rank() == 3);
    assert(copied.elements() == 24);
    assert(copied(0) == 2);
    assert(copied(1) == 3);
    assert(copied(2) == 4);

    Shape moved(Buffer<std::size_t>{5, 6});

    assert(moved.rank() == 2);
    assert(moved.elements() == 30);
    assert(moved(0) == 5);
    assert(moved(1) == 6);
}

void test_index_operater()
{
    Shape shape{3, 224, 224};

    assert(shape(0) == 3);
    assert(shape(1) == 224);
    assert(shape(2) == 224);

    bool threw = false;

    try {
        shape(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    assert(threw);
}

void test_equality()
{
    Shape a{3, 224, 224};
    Shape b{3, 224, 224};
    Shape c{224, 224};
    Shape d{224, 3, 224};
    Shape empty_a;
    Shape empty_b;

    assert(a == b);
    assert(!(a != b));

    assert(a != c);
    assert(!(a == c));

    assert(a != d);
    assert(!(a == d));

    assert(empty_a == empty_b);
    assert(!(empty_a != empty_b));
}

void test_iteration()
{
    Shape shape{3, 224, 224};

    std::size_t expected[] = {3, 224, 224};
    std::size_t i = 0;

    for (std::size_t dim : shape) {
        assert(dim == expected[i]);
        ++i;
    }

    assert(i == 3);
}

void test_output()
{
    Shape shape{3, 224, 224};
    Shape rank_one{3};
    Shape empty;

    std::ostringstream out;
    out << shape;

    assert(out.str() == "(3, 224, 224)");

    std::ostringstream rank_one_out;
    rank_one_out << rank_one;

    assert(rank_one_out.str() == "(3,)");

    std::ostringstream empty_out;
    empty_out << empty;

    assert(empty_out.str() == "()");
}

void test_copy_assignment()
{
    Shape original{2, 3, 4};
    Shape copy;

    copy = original;

    assert(copy == original);
    assert(copy.rank() == 3);
    assert(copy.elements() == 24);

    copy = copy;
    assert(copy == original);
}

void test_move_assignment()
{
    Shape original{5, 6};
    Shape moved;

    moved = std::move(original);

    assert(moved.rank() == 2);
    assert(moved.elements() == 30);
    assert(moved(0) == 5);
    assert(moved(1) == 6);

    Shape& same = moved;
    moved = std::move(same);
    assert(moved.rank() == 2);
    assert(moved.elements() == 30);
}

void test_swap()
{
    Shape a{2, 3, 4};
    Shape b{5};

    a.swap(b);

    assert(a.rank() == 1);
    assert(a(0) == 5);
    assert(b.rank() == 3);
    assert(b(0) == 2);
    assert(b(1) == 3);
    assert(b(2) == 4);
}

int main()
{
    test_default_constructor();
    test_initializer_list_constructor();
    test_zero_dimension_elements();
    test_elements_overflow_throws();
    test_buffer_constructors();
    test_index_operater();
    test_equality();
    test_iteration();
    test_output();
    test_copy_assignment();
    test_move_assignment();
    test_swap();

    return 0;
}
