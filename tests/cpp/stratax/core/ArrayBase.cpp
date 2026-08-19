#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <limits>
#include <utility>
#include <vector>

#include <stratax.h>

using namespace stratax::core;

namespace {

template<typename T>
class TestArray : public ArrayBase<T>
{
public:
    explicit TestArray(const Shape& shape)
        : ArrayBase<T>(shape)
    {}

    TestArray(const Shape& shape, const T& value)
        : ArrayBase<T>(shape, value)
    {}

    TestArray(const Shape& shape, Buffer<T>&& buffer)
        : ArrayBase<T>(shape, std::move(buffer))
    {}

    using ArrayBase<T>::normalized_flat_offset;

    void swap(TestArray& other) noexcept
    {
        ArrayBase<T>::swap(other);
    }
};

} // namespace

TEST(ArrayBaseConstructor, Shape)
{
    const TestArray<int> array(Shape{2, 3});

    EXPECT_EQ(array.size(), 6);
    EXPECT_EQ(array.rank(), 2);
    EXPECT_EQ(array.shape(), (Shape{2, 3}));
    EXPECT_EQ(array.strides(), (Shape{3, 1}));
    EXPECT_FALSE(array.empty());

    for (int value : array) {
        EXPECT_EQ(value, 0);
    }
}

TEST(ArrayBaseConstructor, EmptyShape)
{
    const TestArray<int> array(Shape{});

    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.rank(), 0);
    EXPECT_TRUE(array.empty());
    EXPECT_EQ(array.data(), nullptr);
    EXPECT_TRUE(array.shape().empty());
    EXPECT_TRUE(array.strides().empty());
}

TEST(ArrayBaseConstructor, ZeroDimension)
{
    const TestArray<int> array(Shape{2, 0, 4});

    EXPECT_EQ(array.size(), 0);
    EXPECT_EQ(array.rank(), 3);
    EXPECT_TRUE(array.empty());
    EXPECT_EQ(array.shape(), (Shape{2, 0, 4}));
    EXPECT_EQ(array.strides(), (Shape{0, 4, 1}));
}

TEST(ArrayBaseConstructor, FillValue)
{
    const TestArray<int> array(Shape{2, 3}, 7);

    ASSERT_EQ(array.size(), 6);
    for (int value : array) {
        EXPECT_EQ(value, 7);
    }
}

TEST(ArrayBaseConstructor, AdoptBuffer)
{
    Buffer<int> buffer{1, 2, 3, 4, 5, 6};
    auto* original_data = buffer.data();

    const TestArray<int> array(Shape{2, 3}, std::move(buffer));

    EXPECT_EQ(array.data(), original_data);
    EXPECT_EQ(array.shape(), (Shape{2, 3}));
    EXPECT_EQ(array.strides(), (Shape{3, 1}));
    EXPECT_TRUE(buffer.empty());
}

TEST(ArrayBaseConstructor, RejectsMismatchedBuffer)
{
    Buffer<int> buffer{1, 2, 3};

    EXPECT_THROW(
        static_cast<void>(TestArray<int>(Shape{2, 2}, std::move(buffer))),
        Exceptions::ShapeError
    );
}

TEST(ArrayBaseConstructor, RejectsElementCountOverflow)
{
    const Shape shape{std::numeric_limits<std::size_t>::max(), 2};

    EXPECT_THROW(static_cast<void>(TestArray<int>(shape)), Exceptions::DimensionError);
}

TEST(ArrayBaseConstructor, Copy)
{
    const TestArray<int> original(Shape{2, 3}, 7);
    TestArray<int> copy(original);

    EXPECT_EQ(copy.shape(), original.shape());
    EXPECT_EQ(copy.strides(), original.strides());
    EXPECT_NE(copy.data(), original.data());

    copy.front() = 99;
    EXPECT_EQ(copy.front(), 99);
    EXPECT_EQ(original.front(), 7);
}

TEST(ArrayBaseConstructor, Move)
{
    TestArray<int> source(Shape{2, 3}, 7);
    auto* original_data = source.data();

    TestArray<int> destination(std::move(source));

    EXPECT_EQ(destination.data(), original_data);
    EXPECT_EQ(destination.shape(), (Shape{2, 3}));
    EXPECT_EQ(destination.strides(), (Shape{3, 1}));
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.rank(), 0);
    EXPECT_TRUE(source.shape().empty());
    EXPECT_TRUE(source.strides().empty());
}

TEST(ArrayBaseAssignment, CopyAssignment)
{
    const TestArray<int> source(Shape{2, 3}, 7);
    TestArray<int> destination(Shape{2}, 4);

    destination = source;

    EXPECT_EQ(destination.shape(), source.shape());
    EXPECT_EQ(destination.strides(), source.strides());
    EXPECT_NE(destination.data(), source.data());
    EXPECT_EQ(destination.front(), 7);
}

TEST(ArrayBaseAssignment, SelfCopyAssignment)
{
    TestArray<int> array(Shape{2, 3}, 7);
    auto* original_data = array.data();

    array = array;

    EXPECT_EQ(array.data(), original_data);
    EXPECT_EQ(array.shape(), (Shape{2, 3}));
    EXPECT_EQ(array.front(), 7);
}

TEST(ArrayBaseAssignment, MoveAssignment)
{
    TestArray<int> source(Shape{2, 3}, 7);
    TestArray<int> destination(Shape{2}, 4);
    auto* source_data = source.data();

    destination = std::move(source);

    EXPECT_EQ(destination.data(), source_data);
    EXPECT_EQ(destination.shape(), (Shape{2, 3}));
    EXPECT_EQ(destination.strides(), (Shape{3, 1}));
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.rank(), 0);
}

TEST(ArrayBaseAssignment, SelfMoveAssignment)
{
    TestArray<int> array(Shape{2, 3}, 7);
    auto* original_data = array.data();

    array = std::move(array);

    EXPECT_EQ(array.data(), original_data);
    EXPECT_EQ(array.shape(), (Shape{2, 3}));
    EXPECT_EQ(array.front(), 7);
}

TEST(ArrayBaseTypes, Aliases)
{
    using Array = TestArray<int>;

    static_assert(std::same_as<Array::value_type, int>);
    static_assert(std::same_as<Array::size_type, std::size_t>);
    static_assert(std::same_as<Array::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<Array::reference, int&>);
    static_assert(std::same_as<Array::const_reference, const int&>);
    static_assert(std::same_as<Array::pointer, int*>);
    static_assert(std::same_as<Array::const_pointer, const int*>);
    static_assert(std::same_as<Array::iterator, int*>);
    static_assert(std::same_as<Array::const_iterator, const int*>);
}

TEST(ArrayBaseMeta, Size)
{
    const TestArray<int> empty_array(Shape{});
    const TestArray<int> array(Shape{2, 3});

    EXPECT_EQ(empty_array.size(), 0);
    EXPECT_EQ(array.size(), 6);
    static_assert(noexcept(array.size()));
}

TEST(ArrayBaseMeta, Empty)
{
    const TestArray<int> empty_array(Shape{});
    const TestArray<int> array(Shape{2, 3});

    EXPECT_TRUE(empty_array.empty());
    EXPECT_FALSE(array.empty());
    static_assert(noexcept(array.empty()));
}

TEST(ArrayBaseMeta, Rank)
{
    const TestArray<int> scalar(Shape{});
    const TestArray<int> array(Shape{2, 3, 4});

    EXPECT_EQ(scalar.rank(), 0);
    EXPECT_EQ(array.rank(), 3);
    static_assert(noexcept(array.rank()));
}

TEST(ArrayBaseMeta, ShapeAndStrides)
{
    const TestArray<int> array(Shape{2, 3, 4});

    EXPECT_EQ(array.shape(), (Shape{2, 3, 4}));
    EXPECT_EQ(array.strides(), (Shape{12, 4, 1}));

    static_assert(
        std::same_as<decltype(array.shape()), const Shape&>
    );
    static_assert(
        std::same_as<decltype(array.strides()), const Shape&>
    );
}

TEST(ArrayBaseAccess, Data)
{
    TestArray<int> array(Shape{2, 3}, 7);
    const TestArray<int>& const_array = array;

    ASSERT_NE(array.data(), nullptr);
    array.data()[0] = 99;
    EXPECT_EQ(array[0], 99);
    EXPECT_EQ(const_array.data()[0], 99);

    static_assert(
        std::same_as<decltype(std::declval<TestArray<int>&>().data()), int*>
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const TestArray<int>&>().data()),
            const int*
        >
    );
}

TEST(ArrayBaseAccess, Front)
{
    TestArray<int> array(Shape{2, 3}, 7);

    EXPECT_EQ(array.front(), 7);
    array.front() = 99;
    EXPECT_EQ(array[0], 99);

    static_assert(
        std::same_as<decltype(std::declval<TestArray<int>&>().front()), int&>
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const TestArray<int>&>().front()),
            const int&
        >
    );
}

TEST(ArrayBaseAccess, Back)
{
    TestArray<int> array(Shape{2, 3}, 7);

    EXPECT_EQ(array.back(), 7);
    array.back() = 99;
    EXPECT_EQ(array[array.size() - 1], 99);
}

TEST(ArrayBaseAccess, FrontEmpty)
{
    TestArray<int> array(Shape{});

    EXPECT_THROW(static_cast<void>(array.front()), Exceptions::IndexError);
}

TEST(ArrayBaseAccess, BackEmpty)
{
    TestArray<int> array(Shape{});

    EXPECT_THROW(static_cast<void>(array.back()), Exceptions::IndexError);
}

TEST(ArrayBaseAccess, Subscript)
{
    TestArray<int> array(Shape{3}, 0);
    const TestArray<int>& const_array = array;

    array[0] = 1;
    array[1] = 2;
    array[2] = 3;

    EXPECT_EQ(const_array[0], 1);
    EXPECT_EQ(const_array[1], 2);
    EXPECT_EQ(const_array[2], 3);
    static_assert(noexcept(array[0]));
}

TEST(ArrayBaseAccess, AtPositiveIndices)
{
    TestArray<int> array(Shape{3}, 0);
    array[0] = 1;
    array[1] = 2;
    array[2] = 3;

    EXPECT_EQ(array.at(0), 1);
    EXPECT_EQ(array.at(1), 2);
    EXPECT_EQ(array.at(2), 3);
}

TEST(ArrayBaseAccess, AtNegativeIndices)
{
    TestArray<int> array(Shape{3}, 0);
    array[0] = 1;
    array[1] = 2;
    array[2] = 3;

    EXPECT_EQ(array.at(-3), 1);
    EXPECT_EQ(array.at(-2), 2);
    EXPECT_EQ(array.at(-1), 3);
}

TEST(ArrayBaseAccess, AtOutOfRange)
{
    const TestArray<int> array(Shape{3}, 0);

    EXPECT_THROW(static_cast<void>(array.at(-4)), Exceptions::IndexError);
    EXPECT_THROW(static_cast<void>(array.at(3)), Exceptions::IndexError);
}

TEST(ArrayBaseAccess, AtEmpty)
{
    const TestArray<int> array(Shape{});

    EXPECT_THROW(static_cast<void>(array.at(0)), Exceptions::IndexError);
    EXPECT_THROW(static_cast<void>(array.at(-1)), Exceptions::IndexError);
}

TEST(ArrayBaseIterator, BeginEnd)
{
    TestArray<int> array(Shape{3}, 7);

    EXPECT_EQ(*array.begin(), 7);
    EXPECT_EQ(array.end() - array.begin(), 3);
    *array.begin() = 99;
    EXPECT_EQ(array.front(), 99);
}

TEST(ArrayBaseIterator, ConstBeginEnd)
{
    const TestArray<int> array(Shape{3}, 7);

    EXPECT_EQ(*array.begin(), 7);
    EXPECT_EQ(array.end() - array.begin(), 3);
    EXPECT_EQ(*array.cbegin(), 7);
    EXPECT_EQ(array.cend() - array.cbegin(), 3);
}

TEST(ArrayBaseIterator, ForwardTraversal)
{
    TestArray<int> array(Shape{3}, 0);
    array[0] = 1;
    array[1] = 2;
    array[2] = 3;

    int expected = 1;
    for (int value : array) {
        EXPECT_EQ(value, expected);
        ++expected;
    }
}

TEST(ArrayBaseIterator, ReverseTraversal)
{
    TestArray<int> array(Shape{3}, 0);
    array[0] = 1;
    array[1] = 2;
    array[2] = 3;

    int expected = 3;
    for (auto it = array.rbegin(); it != array.rend(); ++it) {
        EXPECT_EQ(*it, expected);
        --expected;
    }

    *array.rbegin() = 99;
    EXPECT_EQ(array.back(), 99);
}

TEST(ArrayBaseIterator, ConstReverseTraversal)
{
    const TestArray<int> array(Shape{3}, 7);

    EXPECT_EQ(*array.rbegin(), 7);
    EXPECT_EQ(array.rend() - array.rbegin(), 3);
    EXPECT_EQ(*array.crbegin(), 7);
    EXPECT_EQ(array.crend() - array.crbegin(), 3);
}

TEST(ArrayBaseIterator, Empty)
{
    TestArray<int> array(Shape{});
    const TestArray<int>& const_array = array;

    EXPECT_EQ(array.begin(), array.end());
    EXPECT_EQ(array.rbegin(), array.rend());
    EXPECT_EQ(const_array.begin(), const_array.end());
    EXPECT_EQ(const_array.cbegin(), const_array.cend());
    EXPECT_EQ(const_array.rbegin(), const_array.rend());
    EXPECT_EQ(const_array.crbegin(), const_array.crend());
}

TEST(ArrayBaseModifier, Fill)
{
    TestArray<int> array(Shape{2, 3}, 1);

    array.fill(9);

    for (int value : array) {
        EXPECT_EQ(value, 9);
    }
    EXPECT_EQ(array.shape(), (Shape{2, 3}));
}

TEST(ArrayBaseModifier, Swap)
{
    TestArray<int> lhs(Shape{2, 3}, 1);
    TestArray<int> rhs(Shape{4}, 2);
    auto* lhs_data = lhs.data();
    auto* rhs_data = rhs.data();

    lhs.swap(rhs);

    EXPECT_EQ(lhs.data(), rhs_data);
    EXPECT_EQ(rhs.data(), lhs_data);
    EXPECT_EQ(lhs.shape(), (Shape{4}));
    EXPECT_EQ(rhs.shape(), (Shape{2, 3}));
    EXPECT_EQ(lhs.strides(), (Shape{1}));
    EXPECT_EQ(rhs.strides(), (Shape{3, 1}));
    EXPECT_EQ(lhs.front(), 2);
    EXPECT_EQ(rhs.front(), 1);
}

TEST(ArrayBaseModifier, SwapWithEmpty)
{
    TestArray<int> lhs(Shape{2, 3}, 1);
    TestArray<int> rhs(Shape{});

    lhs.swap(rhs);

    EXPECT_TRUE(lhs.empty());
    EXPECT_TRUE(lhs.shape().empty());
    EXPECT_EQ(rhs.shape(), (Shape{2, 3}));
    EXPECT_EQ(rhs.front(), 1);
}

TEST(ArrayBaseModifier, SelfSwap)
{
    TestArray<int> array(Shape{2, 3}, 7);
    auto* original_data = array.data();

    array.swap(array);

    EXPECT_EQ(array.data(), original_data);
    EXPECT_EQ(array.shape(), (Shape{2, 3}));
    EXPECT_EQ(array.front(), 7);
}

TEST(ArrayBaseIndexing, NormalizedFlatOffsetPositive)
{
    const TestArray<int> array(Shape{2, 3, 4});
    const std::array<std::ptrdiff_t, 3> indices{1, 2, 3};

    EXPECT_EQ(array.normalized_flat_offset(indices), 23);
}

TEST(ArrayBaseIndexing, NormalizedFlatOffsetNegative)
{
    const TestArray<int> array(Shape{2, 3, 4});
    const std::array<std::ptrdiff_t, 3> indices{-1, -1, -1};

    EXPECT_EQ(array.normalized_flat_offset(indices), 23);
}

TEST(ArrayBaseIndexing, NormalizedFlatOffsetRankMismatch)
{
    const TestArray<int> array(Shape{2, 3, 4});
    const std::array<std::ptrdiff_t, 2> indices{1, 2};

    EXPECT_THROW(
        static_cast<void>(array.normalized_flat_offset(indices)),
        Exceptions::IndexError
    );
}

TEST(ArrayBaseIndexing, NormalizedFlatOffsetComponentOutOfRange)
{
    const TestArray<int> array(Shape{2, 3, 4});
    const std::array<std::ptrdiff_t, 3> indices{1, 3, 0};

    EXPECT_THROW(
        static_cast<void>(array.normalized_flat_offset(indices)),
        Exceptions::IndexError
    );
}

TEST(ArrayBaseIndexing, NormalizedFlatOffsetCustomMessages)
{
    const TestArray<int> array(Shape{2, 3, 4});
    const std::array<std::ptrdiff_t, 2> wrong_rank{1, 2};
    const std::array<std::ptrdiff_t, 3> bad_component{1, 3, 0};

    try {
        static_cast<void>(array.normalized_flat_offset(
            wrong_rank,
            "custom rank message"
        ));
        FAIL() << "Expected rank mismatch to throw";
    } catch (const Exceptions::IndexError& error) {
        EXPECT_STREQ(error.what(), "custom rank message");
    }

    try {
        static_cast<void>(array.normalized_flat_offset(
            bad_component,
            "unused rank message",
            "custom component message"
        ));
        FAIL() << "Expected component mismatch to throw";
    } catch (const Exceptions::IndexError& error) {
        EXPECT_STREQ(error.what(), "custom component message");
    }
}
