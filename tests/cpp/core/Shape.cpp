#include <gtest/gtest.h>
#include <concepts>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <stratax.h>

using namespace stratax::core;

static_assert(std::constructible_from<Shape, int>);
static_assert(std::constructible_from<Shape, int, int, int>);
static_assert(!std::constructible_from<Shape, bool>);
static_assert(!std::constructible_from<Shape, char>);
static_assert(!std::constructible_from<Shape, double>);

TEST(CoreShape, default_constructor)
{
    Shape shape;

    EXPECT_TRUE(shape.rank() == 0);
    EXPECT_TRUE(shape.empty());
    EXPECT_TRUE(shape.elements() == 0);
    EXPECT_TRUE(shape.begin() == shape.end());
}

TEST(CoreShape, initializer_list_constructor)
{
    Shape shape{3, 224, 224};

    EXPECT_TRUE(shape.rank() == 3);
    EXPECT_TRUE(shape.elements() == 150528);
    EXPECT_TRUE(shape(0) == 3);
    EXPECT_TRUE(shape(1) == 224);
    EXPECT_TRUE(shape(2) == 224);
}

TEST(CoreShape, zero_dimension_elements)
{
    Shape scalar_zero{0};
    Shape matrix_zero{0, 3};
    Shape later_zero{std::numeric_limits<long long>::max(), 0};

    EXPECT_TRUE(scalar_zero.rank() == 1);
    EXPECT_TRUE(scalar_zero.elements() == 0);
    EXPECT_TRUE(matrix_zero.rank() == 2);
    EXPECT_TRUE(matrix_zero.elements() == 0);
    EXPECT_TRUE(later_zero.rank() == 2);
    EXPECT_TRUE(later_zero.elements() == 0);
}

TEST(CoreShape, initializer_list_constructor_rejects_bool_and_char_like_types)
{
    // Compile-time checks above cover bool, char-like, and floating-point input.
    Shape shape{0};
    EXPECT_TRUE(shape.rank() == 1);
    EXPECT_TRUE(shape.elements() == 0);
}

TEST(CoreShape, elements_overflow_throws)
{
    Shape shape{2, std::numeric_limits<long long>::max(), 2};

    EXPECT_THROW(
        [[maybe_unused]] const std::size_t elements = shape.elements(),
        Exceptions::StrataxError);
}

TEST(CoreShape, buffer_constructors)
{
    Buffer<std::size_t> dims{2, 3, 4};
    Shape copied(dims);

    EXPECT_TRUE(copied.rank() == 3);
    EXPECT_TRUE(copied.elements() == 24);
    EXPECT_TRUE(copied(0) == 2);
    EXPECT_TRUE(copied(1) == 3);
    EXPECT_TRUE(copied(2) == 4);

    Shape moved(Buffer<std::size_t>{5, 6});

    EXPECT_TRUE(moved.rank() == 2);
    EXPECT_TRUE(moved.elements() == 30);
    EXPECT_TRUE(moved(0) == 5);
    EXPECT_TRUE(moved(1) == 6);
}

TEST(CoreShape, index_operater)
{
    Shape shape{3, 224, 224};

    EXPECT_TRUE(shape(0) == 3);
    EXPECT_TRUE(shape(1) == 224);
    EXPECT_TRUE(shape(2) == 224);

    bool threw = false;

    try {
        shape(3);
    }
    catch (const Exceptions::IndexError&) {
        threw = true;
    }

    EXPECT_TRUE(threw);
}

TEST(CoreShape, equality)
{
    Shape a{3, 224, 224};
    Shape b{3, 224, 224};
    Shape c{224, 224};
    Shape d{224, 3, 224};
    Shape empty_a;
    Shape empty_b;

    EXPECT_TRUE(a == b);
    EXPECT_TRUE(!(a != b));

    EXPECT_TRUE(a != c);
    EXPECT_TRUE(!(a == c));

    EXPECT_TRUE(a != d);
    EXPECT_TRUE(!(a == d));

    EXPECT_TRUE(empty_a == empty_b);
    EXPECT_TRUE(!(empty_a != empty_b));
}

TEST(CoreShape, iteration)
{
    Shape shape{3, 224, 224};

    std::size_t expected[] = {3, 224, 224};
    std::size_t i = 0;

    for (std::size_t dim : shape) {
        EXPECT_TRUE(dim == expected[i]);
        ++i;
    }

    EXPECT_TRUE(i == 3);
}

TEST(CoreShape, output)
{
    Shape shape{3, 224, 224};
    Shape rank_one{3};
    Shape empty;

    std::ostringstream out;
    out << shape;

    EXPECT_TRUE(out.str() == "(3, 224, 224)");

    std::ostringstream rank_one_out;
    rank_one_out << rank_one;

    EXPECT_TRUE(rank_one_out.str() == "(3,)");

    std::ostringstream empty_out;
    empty_out << empty;

    EXPECT_TRUE(empty_out.str() == "()");
}

TEST(CoreShape, copy_assignment)
{
    Shape original{2, 3, 4};
    Shape copy;

    copy = original;

    EXPECT_TRUE(copy == original);
    EXPECT_TRUE(copy.rank() == 3);
    EXPECT_TRUE(copy.elements() == 24);

    copy = copy;
    EXPECT_TRUE(copy == original);
}

TEST(CoreShape, move_assignment)
{
    Shape original{5, 6};
    Shape moved;

    moved = std::move(original);

    EXPECT_TRUE(moved.rank() == 2);
    EXPECT_TRUE(moved.elements() == 30);
    EXPECT_TRUE(moved(0) == 5);
    EXPECT_TRUE(moved(1) == 6);

    Shape& same = moved;
    moved = std::move(same);
    EXPECT_TRUE(moved.rank() == 2);
    EXPECT_TRUE(moved.elements() == 30);
}

TEST(CoreShape, swap)
{
    Shape a{2, 3, 4};
    Shape b{5};

    a.swap(b);

    EXPECT_TRUE(a.rank() == 1);
    EXPECT_TRUE(a(0) == 5);
    EXPECT_TRUE(b.rank() == 3);
    EXPECT_TRUE(b(0) == 2);
    EXPECT_TRUE(b(1) == 3);
    EXPECT_TRUE(b(2) == 4);
}

