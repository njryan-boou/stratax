#include <gtest/gtest.h>

#include <concepts>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include <stratax.h>

using namespace stratax::core;

TEST(ShapeConstructor, DefaultConstructor)
{
    const Shape shape;

    EXPECT_EQ(shape.rank(), 0);
    EXPECT_EQ(shape.elements(), 0);
    EXPECT_TRUE(shape.empty());
}

TEST(ShapeConstructor, InitializerList)
{
    const Shape shape{2, 3, 4};

    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    EXPECT_EQ(shape.elements(), 24);
    EXPECT_FALSE(shape.empty());
}

TEST(ShapeConstructor, Vector)
{
    const std::vector<std::size_t> dimensions{5, 6, 7};
    const Shape shape(dimensions);

    EXPECT_EQ(shape.rank(), dimensions.size());

    for (std::size_t i = 0; i < dimensions.size(); ++i) {
        EXPECT_EQ(shape[i], dimensions[i]);
    }
}

TEST(ShapeConstructor, EmptyVector)
{
    const std::vector<std::size_t> dimensions;
    const Shape shape(dimensions);

    EXPECT_TRUE(shape.empty());
    EXPECT_EQ(shape.rank(), 0);
    EXPECT_EQ(shape.elements(), 0);
}

TEST(ShapeConstructor, Copy)
{
    const Shape original{2, 3, 4};
    const Shape copy(original);

    EXPECT_EQ(copy, original);
    EXPECT_EQ(copy.rank(), original.rank());
    EXPECT_EQ(copy.elements(), original.elements());
}

TEST(ShapeConstructor, Move)
{
    Shape source{2, 3, 4};
    Shape destination(std::move(source));

    EXPECT_EQ(destination, (Shape{2, 3, 4}));
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.rank(), 0);
}

TEST(ShapeAssignment, CopyAssignment)
{
    const Shape source{2, 3, 4};
    Shape destination{5, 6};

    destination = source;

    EXPECT_EQ(destination, source);
    EXPECT_EQ(destination.rank(), 3);
}

TEST(ShapeAssignment, SelfCopyAssignment)
{
    Shape shape{2, 3, 4};

    shape = shape;

    EXPECT_EQ(shape, (Shape{2, 3, 4}));
}

TEST(ShapeAssignment, MoveAssignment)
{
    Shape source{2, 3, 4};
    Shape destination{5, 6};

    destination = std::move(source);

    EXPECT_EQ(destination, (Shape{2, 3, 4}));
    EXPECT_TRUE(source.empty());
}

TEST(ShapeAssignment, SelfMoveAssignment)
{
    Shape shape{2, 3, 4};

    shape = std::move(shape);

    EXPECT_EQ(shape, (Shape{2, 3, 4}));
}

TEST(ShapeMeta, Rank)
{
    const Shape empty_shape;
    const Shape shape{2, 3, 4};

    EXPECT_EQ(empty_shape.rank(), 0);
    EXPECT_EQ(shape.rank(), 3);

    static_assert(std::same_as<decltype(shape.rank()), std::size_t>);
    static_assert(noexcept(shape.rank()));
}

TEST(ShapeMeta, Strides)
{
    EXPECT_EQ(Shape({2, 3, 4}).strides(), Shape({12, 4, 1}));
    EXPECT_EQ(Shape({5}).strides(), Shape({1}));
    EXPECT_EQ(Shape({}).strides(), Shape({}));
}

TEST(ShapeMeta, StridesPreserveZeroSizedLayout)
{
    EXPECT_EQ(Shape({2, 0, 4}).strides(), Shape({0, 4, 1}));
}

TEST(ShapeMeta, StridesRejectOverflow)
{
    const Shape shape{1, std::numeric_limits<std::size_t>::max(), 2};

    EXPECT_THROW(static_cast<void>(shape.strides()), Exceptions::DimensionError);
}

TEST(ShapeMeta, Empty)
{
    const Shape empty_shape;
    const Shape shape{2};

    EXPECT_TRUE(empty_shape.empty());
    EXPECT_FALSE(shape.empty());

    static_assert(std::same_as<decltype(shape.empty()), bool>);
    static_assert(noexcept(shape.empty()));
}

TEST(ShapeMeta, Elements)
{
    EXPECT_EQ(Shape({2, 3, 4}).elements(), 24);
    EXPECT_EQ(Shape({1, 1, 1}).elements(), 1);
    EXPECT_EQ(Shape({7}).elements(), 7);
    EXPECT_EQ(Shape().elements(), 0);
}

TEST(ShapeMeta, ZeroDimensionHasZeroElements)
{
    EXPECT_EQ(Shape({0}).elements(), 0);
    EXPECT_EQ(Shape({0, 3, 4}).elements(), 0);
    EXPECT_EQ(Shape({2, 0, 4}).elements(), 0);
    EXPECT_EQ(Shape({2, 3, 0}).elements(), 0);
}

TEST(ShapeMeta, ElementCountOverflow)
{
    const Shape shape{
        std::numeric_limits<std::size_t>::max(),
        2
    };

    EXPECT_THROW(static_cast<void>(shape.elements()), Exceptions::DimensionError);
}

TEST(ShapeAccess, Subscript)
{
    const Shape shape{2, 3, 4};

    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);

    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>()[0]),
            const std::size_t&
        >
    );
    static_assert(noexcept(std::declval<const Shape&>()[0]));
}

TEST(ShapeAccess, AtPositiveIndices)
{
    const Shape shape{2, 3, 4};

    EXPECT_EQ(shape.at(0), 2);
    EXPECT_EQ(shape.at(1), 3);
    EXPECT_EQ(shape.at(2), 4);
}

TEST(ShapeAccess, AtNegativeIndices)
{
    const Shape shape{2, 3, 4};

    EXPECT_EQ(shape.at(-3), 2);
    EXPECT_EQ(shape.at(-2), 3);
    EXPECT_EQ(shape.at(-1), 4);
}

TEST(ShapeAccess, AtOutOfRange)
{
    const Shape shape{2, 3, 4};

    EXPECT_THROW(static_cast<void>(shape.at(-4)), Exceptions::IndexError);
    EXPECT_THROW(static_cast<void>(shape.at(3)), Exceptions::IndexError);
}

TEST(ShapeAccess, AtEmpty)
{
    const Shape shape;

    EXPECT_THROW(static_cast<void>(shape.at(0)), Exceptions::IndexError);
    EXPECT_THROW(static_cast<void>(shape.at(-1)), Exceptions::IndexError);
}

TEST(ShapeIterator, BeginEnd)
{
    const Shape shape{2, 3, 4};

    EXPECT_EQ(*shape.begin(), 2);
    EXPECT_EQ(shape.end() - shape.begin(), 3);

    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().begin()),
            Shape::const_iterator
        >
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().end()),
            Shape::const_iterator
        >
    );
}

TEST(ShapeIterator, CBeginCEnd)
{
    const Shape shape{2, 3, 4};

    EXPECT_EQ(*shape.cbegin(), 2);
    EXPECT_EQ(shape.cend() - shape.cbegin(), 3);

    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().cbegin()),
            Shape::const_iterator
        >
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().cend()),
            Shape::const_iterator
        >
    );
}

TEST(ShapeIterator, ForwardTraversal)
{
    const Shape shape{2, 3, 4};
    const std::vector<std::size_t> expected{2, 3, 4};

    std::size_t index = 0;
    for (std::size_t dimension : shape) {
        EXPECT_EQ(dimension, expected[index]);
        ++index;
    }

    EXPECT_EQ(index, expected.size());
}

TEST(ShapeIterator, ReverseTraversal)
{
    const Shape shape{2, 3, 4};
    const std::vector<std::size_t> expected{4, 3, 2};

    std::size_t index = 0;
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        EXPECT_EQ(*it, expected[index]);
        ++index;
    }

    EXPECT_EQ(index, expected.size());

    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().rbegin()),
            Shape::const_reverse_iterator
        >
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().rend()),
            Shape::const_reverse_iterator
        >
    );
}

TEST(ShapeIterator, ConstReverseTraversal)
{
    const Shape shape{2, 3, 4};

    EXPECT_EQ(*shape.crbegin(), 4);
    EXPECT_EQ(shape.crend() - shape.crbegin(), 3);

    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().crbegin()),
            Shape::const_reverse_iterator
        >
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Shape&>().crend()),
            Shape::const_reverse_iterator
        >
    );
}

TEST(ShapeIterator, Empty)
{
    const Shape shape;

    EXPECT_EQ(shape.begin(), shape.end());
    EXPECT_EQ(shape.cbegin(), shape.cend());
    EXPECT_EQ(shape.rbegin(), shape.rend());
    EXPECT_EQ(shape.crbegin(), shape.crend());
}

TEST(ShapeComparison, Equal)
{
    EXPECT_EQ(Shape({2, 3, 4}), Shape({2, 3, 4}));
    EXPECT_EQ(Shape(), Shape());
}

TEST(ShapeComparison, DifferentDimensions)
{
    EXPECT_NE(Shape({2, 3}), Shape({3, 2}));
}

TEST(ShapeComparison, DifferentRanks)
{
    EXPECT_NE(Shape({2, 3}), Shape({2, 3, 1}));
    EXPECT_NE(Shape(), Shape({1}));
}

TEST(ShapeModifier, Swap)
{
    Shape lhs{2, 3};
    Shape rhs{5, 7, 11};

    lhs.swap(rhs);

    EXPECT_EQ(lhs, (Shape{5, 7, 11}));
    EXPECT_EQ(rhs, (Shape{2, 3}));
}

TEST(ShapeModifier, SwapWithEmpty)
{
    Shape lhs{2, 3};
    Shape rhs;

    lhs.swap(rhs);

    EXPECT_TRUE(lhs.empty());
    EXPECT_EQ(rhs, (Shape{2, 3}));
}

TEST(ShapeModifier, SelfSwap)
{
    Shape shape{2, 3, 4};

    shape.swap(shape);

    EXPECT_EQ(shape, (Shape{2, 3, 4}));
}

TEST(ShapeStream, MultipleDimensions)
{
    std::ostringstream stream;

    stream << Shape{2, 3, 4};

    EXPECT_EQ(stream.str(), "(2, 3, 4)");
}

TEST(ShapeStream, SingleDimension)
{
    std::ostringstream stream;

    stream << Shape{5};

    EXPECT_EQ(stream.str(), "(5,)");
}

TEST(ShapeStream, Empty)
{
    std::ostringstream stream;

    stream << Shape{};

    EXPECT_EQ(stream.str(), "()");
}

TEST(ShapeStream, ReturnsOutputStream)
{
    std::ostringstream stream;

    std::ostream& result = (stream << Shape{2, 3});

    EXPECT_EQ(&result, &stream);
}
