#include <gtest/gtest.h>
#include <concepts>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include <stratax.h>

using stratax::core::Buffer;
using stratax::core::Shape;

static_assert(!std::constructible_from<Shape, int>);
static_assert(!std::constructible_from<Shape, int, int, int>);
static_assert(std::constructible_from<Shape, std::initializer_list<std::size_t>>);
static_assert(!std::constructible_from<Shape, bool>);
static_assert(!std::constructible_from<Shape, char>);
static_assert(!std::constructible_from<Shape, double>);

TEST(CoreShape, DefaultConstructor)
{
    Shape shape;

    EXPECT_EQ(shape.rank(), 0);
    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.elements(), 0);
    EXPECT_EQ(shape.begin(), shape.end());
}

TEST(CoreShape, InitializerListConstructor)
{
    Shape shape{3, 224, 224};

    EXPECT_EQ(shape.rank(), 3);
    EXPECT_FALSE(shape.empty());
    EXPECT_EQ(shape.elements(), 150528);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 224);
    EXPECT_EQ(shape[2], 224);
    EXPECT_EQ(shape(0), 3);
    EXPECT_EQ(shape(1), 224);
    EXPECT_EQ(shape(2), 224);
}

TEST(CoreShape, ZeroDimensionElements)
{
    Shape scalar_zero{0};
    Shape matrix_zero{0, 3};
    Shape later_zero{std::numeric_limits<std::size_t>::max(), 0};

    EXPECT_EQ(scalar_zero.rank(), 1);
    EXPECT_EQ(scalar_zero.elements(), 0);
    EXPECT_EQ(matrix_zero.rank(), 2);
    EXPECT_EQ(matrix_zero.elements(), 0);
    EXPECT_EQ(later_zero.rank(), 2);
    EXPECT_EQ(later_zero.elements(), 0);
}

TEST(CoreShape, ElementsOverflowThrows)
{
    Shape shape{2, std::numeric_limits<std::size_t>::max(), 2};

    EXPECT_THROW(
        [[maybe_unused]] const std::size_t elements = shape.elements(),
        Exceptions::StrataxError);
}

TEST(CoreShape, BufferAndVectorConstructors)
{
    Buffer<std::size_t> dims{2, 3, 4};
    Shape copied(dims);

    EXPECT_EQ(copied.rank(), 3);
    EXPECT_EQ(copied.elements(), 24);
    EXPECT_EQ(copied[0], 2);
    EXPECT_EQ(copied[1], 3);
    EXPECT_EQ(copied[2], 4);

    Shape moved(Buffer<std::size_t>{5, 6});
    EXPECT_EQ(moved.rank(), 2);
    EXPECT_EQ(moved.elements(), 30);
    EXPECT_EQ(moved[0], 5);
    EXPECT_EQ(moved[1], 6);

    std::vector<std::size_t> vec_dims{7, 8, 9};
    Shape from_vec(vec_dims);
    EXPECT_EQ(from_vec.rank(), 3);
    EXPECT_EQ(from_vec.elements(), 504);
    EXPECT_EQ(from_vec[0], 7);
    EXPECT_EQ(from_vec[1], 8);
    EXPECT_EQ(from_vec[2], 9);

    vec_dims[0] = 1;
    EXPECT_EQ(from_vec[0], 7);
}

TEST(CoreShape, AllowZeroTagConstructors)
{
    Shape from_list(std::initializer_list<std::size_t>{0, 5, 0, 3}, Shape::allow_zero);
    EXPECT_EQ(from_list.rank(), 4);
    EXPECT_EQ(from_list[0], 0);
    EXPECT_EQ(from_list[1], 5);
    EXPECT_EQ(from_list[2], 0);
    EXPECT_EQ(from_list[3], 3);
    EXPECT_EQ(from_list.elements(), 0);

    Buffer<std::size_t> dims{1, 0, 2};
    Shape from_buffer(dims, Shape::allow_zero);
    EXPECT_EQ(from_buffer.rank(), 3);
    EXPECT_EQ(from_buffer[1], 0);
    EXPECT_EQ(from_buffer.elements(), 0);
}

TEST(CoreShape, AtProvidesCheckedSignedIndexing)
{
    Shape shape{3, 224, 224};

    EXPECT_EQ(shape.at(0), 3);
    EXPECT_EQ(shape.at(1), 224);
    EXPECT_EQ(shape.at(2), 224);
    EXPECT_EQ(shape.at(-1), 224);
    EXPECT_EQ(shape.at(-2), 224);
    EXPECT_EQ(shape.at(-3), 3);

    EXPECT_THROW(shape.at(3), Exceptions::IndexError);
    EXPECT_THROW(shape.at(-4), Exceptions::IndexError);

    const Shape const_shape{3, 224, 224};
    EXPECT_EQ(const_shape.at(-1), 224);
}

TEST(CoreShape, EqualityAndInequality)
{
    Shape a{3, 224, 224};
    Shape b{3, 224, 224};
    Shape c{224, 224};
    Shape d{224, 3, 224};
    Shape empty_a;
    Shape empty_b;

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);

    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a == c);

    EXPECT_TRUE(a != d);
    EXPECT_FALSE(a == d);

    EXPECT_TRUE(empty_a == empty_b);
    EXPECT_FALSE(empty_a != empty_b);
}

TEST(CoreShape, Iteration)
{
    Shape shape{3, 224, 224};

    const std::size_t expected[] = {3, 224, 224};
    std::size_t i = 0;

    for (std::size_t dim : shape)
    {
        EXPECT_EQ(dim, expected[i]);
        ++i;
    }

    EXPECT_EQ(i, 3);
}

TEST(CoreShape, ReverseIteration)
{
    const Shape shape{10, 20, 30};

    const std::size_t expected[] = {30, 20, 10};
    std::size_t i = 0;

    for (auto it = shape.rbegin(); it != shape.rend(); ++it)
    {
        EXPECT_EQ(*it, expected[i]);
        ++i;
    }

    EXPECT_EQ(i, 3);
}

TEST(CoreShape, EmptyIterators)
{
    Shape shape;
    const Shape const_shape;

    EXPECT_EQ(shape.begin(), shape.end());
    EXPECT_EQ(shape.cbegin(), shape.cend());
    EXPECT_EQ(shape.rbegin(), shape.rend());
    EXPECT_EQ(shape.crbegin(), shape.crend());
    EXPECT_EQ(const_shape.begin(), const_shape.end());
    EXPECT_EQ(const_shape.rbegin(), const_shape.rend());
}

TEST(CoreShape, CopyMoveAssignAndSwap)
{
    Shape original{2, 3, 4};
    Shape copy;
    copy = original;

    EXPECT_EQ(copy.rank(), 3);
    EXPECT_EQ(copy.elements(), 24);
    EXPECT_TRUE(copy == original);

    copy = copy;
    EXPECT_TRUE(copy == original);

    Shape moved;
    moved = std::move(copy);
    EXPECT_EQ(moved.rank(), 3);
    EXPECT_EQ(moved[0], 2);
    EXPECT_EQ(moved[1], 3);
    EXPECT_EQ(moved[2], 4);

    Shape a{2, 3, 4};
    Shape b{5};
    a.swap(b);

    EXPECT_EQ(a.rank(), 1);
    EXPECT_EQ(a[0], 5);
    EXPECT_EQ(b.rank(), 3);
    EXPECT_EQ(b[0], 2);
    EXPECT_EQ(b[1], 3);
    EXPECT_EQ(b[2], 4);
}

TEST(CoreShape, StreamOutput)
{
    Shape shape{3, 224, 224};
    Shape rank_one{3};
    Shape empty;

    std::ostringstream out;
    out << shape;
    EXPECT_EQ(out.str(), "(3, 224, 224)");

    std::ostringstream rank_one_out;
    rank_one_out << rank_one;
    EXPECT_EQ(rank_one_out.str(), "(3,)");

    std::ostringstream empty_out;
    empty_out << empty;
    EXPECT_EQ(empty_out.str(), "()");

    std::ostringstream zeros_out;
    zeros_out << Shape{0, 5, 0};
    EXPECT_EQ(zeros_out.str(), "(0, 5, 0)");
}
