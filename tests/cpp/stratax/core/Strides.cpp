#include <gtest/gtest.h>

#include <concepts>
#include <limits>
#include <sstream>
#include <utility>

#include <stratax.h>

using namespace stratax::core;

TEST(StridesConstructor, DefaultConstructor)
{
    const Strides strides;

    EXPECT_EQ(strides.rank(), 0);
    EXPECT_EQ(strides.data(), nullptr);
    EXPECT_TRUE(strides.empty());
}

TEST(StridesConstructor, Shape)
{
    const Strides strides(Shape{2, 3, 4});

    ASSERT_EQ(strides.rank(), 3);
    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);
    EXPECT_FALSE(strides.empty());
}

TEST(StridesConstructor, SingleDimension)
{
    const Strides strides(Shape{5});

    ASSERT_EQ(strides.rank(), 1);
    EXPECT_EQ(strides[0], 1);
}

TEST(StridesConstructor, EmptyShape)
{
    const Strides strides(Shape{});

    EXPECT_TRUE(strides.empty());
    EXPECT_EQ(strides.rank(), 0);
    EXPECT_EQ(strides.data(), nullptr);
}

TEST(StridesConstructor, ZeroDimension)
{
    const Strides first(Shape{0, 3, 4});
    const Strides middle(Shape{2, 0, 4});
    const Strides last(Shape{2, 3, 0});

    EXPECT_EQ(first, (Strides{Shape{0, 3, 4}}));
    EXPECT_EQ(first[0], 12);
    EXPECT_EQ(first[1], 4);
    EXPECT_EQ(first[2], 1);

    EXPECT_EQ(middle[0], 0);
    EXPECT_EQ(middle[1], 4);
    EXPECT_EQ(middle[2], 1);

    EXPECT_EQ(last[0], 0);
    EXPECT_EQ(last[1], 0);
    EXPECT_EQ(last[2], 1);
}

TEST(StridesConstructor, ThrowsWhenStrideOverflows)
{
    const Shape shape{
        2,
        std::numeric_limits<std::size_t>::max(),
        2
    };

    EXPECT_THROW(static_cast<void>(Strides(shape)), Exceptions::DimensionError);
}

TEST(StridesConstructor, Copy)
{
    const Strides original(Shape{2, 3, 4});
    Strides copy(original);

    EXPECT_EQ(copy, original);
    EXPECT_NE(copy.data(), original.data());

    copy.front() = 99;
    EXPECT_EQ(copy.front(), 99);
    EXPECT_EQ(original.front(), 12);
}

TEST(StridesConstructor, Move)
{
    Strides source(Shape{2, 3, 4});
    auto* original_data = source.data();

    Strides destination(std::move(source));

    EXPECT_EQ(destination.data(), original_data);
    EXPECT_EQ(destination, (Strides{Shape{2, 3, 4}}));
    EXPECT_TRUE(source.empty());
    EXPECT_EQ(source.data(), nullptr);
}

TEST(StridesAssignment, CopyAssignment)
{
    const Strides source(Shape{2, 3, 4});
    Strides destination(Shape{5, 6});

    destination = source;

    EXPECT_EQ(destination, source);
    EXPECT_NE(destination.data(), source.data());
}

TEST(StridesAssignment, SelfCopyAssignment)
{
    Strides strides(Shape{2, 3, 4});
    auto* original_data = strides.data();

    strides = strides;

    EXPECT_EQ(strides.data(), original_data);
    EXPECT_EQ(strides, (Strides{Shape{2, 3, 4}}));
}

TEST(StridesAssignment, MoveAssignment)
{
    Strides source(Shape{2, 3, 4});
    Strides destination(Shape{5, 6});
    auto* source_data = source.data();

    destination = std::move(source);

    EXPECT_EQ(destination.data(), source_data);
    EXPECT_EQ(destination, (Strides{Shape{2, 3, 4}}));
    EXPECT_TRUE(source.empty());
}

TEST(StridesAssignment, SelfMoveAssignment)
{
    Strides strides(Shape{2, 3, 4});
    auto* original_data = strides.data();

    strides = std::move(strides);

    EXPECT_EQ(strides.data(), original_data);
    EXPECT_EQ(strides, (Strides{Shape{2, 3, 4}}));
}

TEST(StridesTypes, Aliases)
{
    static_assert(std::same_as<Strides::value_type, std::size_t>);
    static_assert(std::same_as<Strides::size_type, std::size_t>);
    static_assert(std::same_as<Strides::difference_type, std::ptrdiff_t>);
    static_assert(std::same_as<Strides::reference, std::size_t&>);
    static_assert(std::same_as<Strides::const_reference, const std::size_t&>);
    static_assert(std::same_as<Strides::pointer, std::size_t*>);
    static_assert(std::same_as<Strides::const_pointer, const std::size_t*>);
}

TEST(StridesMeta, Rank)
{
    const Strides empty_strides;
    const Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(empty_strides.rank(), 0);
    EXPECT_EQ(strides.rank(), 3);

    static_assert(std::same_as<decltype(strides.rank()), Strides::size_type>);
    static_assert(noexcept(strides.rank()));
}

TEST(StridesMeta, Empty)
{
    const Strides empty_strides;
    const Strides strides(Shape{2});

    EXPECT_TRUE(empty_strides.empty());
    EXPECT_FALSE(strides.empty());

    static_assert(std::same_as<decltype(strides.empty()), bool>);
    static_assert(noexcept(strides.empty()));
}

TEST(StridesAccess, Data)
{
    Strides strides(Shape{2, 3, 4});
    const Strides& const_strides = strides;

    ASSERT_NE(strides.data(), nullptr);
    EXPECT_EQ(strides.data()[0], 12);

    strides.data()[0] = 99;
    EXPECT_EQ(strides[0], 99);

    static_assert(
        std::same_as<decltype(std::declval<Strides&>().data()), Strides::pointer>
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>().data()),
            Strides::const_pointer
        >
    );
    EXPECT_EQ(const_strides.data()[0], 99);
}

TEST(StridesAccess, Front)
{
    Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(strides.front(), 12);
    strides.front() = 99;
    EXPECT_EQ(strides[0], 99);

    static_assert(
        std::same_as<decltype(std::declval<Strides&>().front()), Strides::reference>
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>().front()),
            Strides::const_reference
        >
    );
}

TEST(StridesAccess, Back)
{
    Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(strides.back(), 1);
    strides.back() = 99;
    EXPECT_EQ(strides[2], 99);

    static_assert(
        std::same_as<decltype(std::declval<Strides&>().back()), Strides::reference>
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>().back()),
            Strides::const_reference
        >
    );
}

TEST(StridesAccess, FrontEmpty)
{
    Strides strides;

    EXPECT_THROW(static_cast<void>(strides.front()), Exceptions::IndexError);
}

TEST(StridesAccess, BackEmpty)
{
    Strides strides;

    EXPECT_THROW(static_cast<void>(strides.back()), Exceptions::IndexError);
}

TEST(StridesAccess, Subscript)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(strides[0], 12);
    EXPECT_EQ(strides[1], 4);
    EXPECT_EQ(strides[2], 1);

    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>()[0]),
            Strides::const_reference
        >
    );
    static_assert(noexcept(std::declval<const Strides&>()[0]));
}

TEST(StridesAccess, AtPositiveIndices)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(strides.at(0), 12);
    EXPECT_EQ(strides.at(1), 4);
    EXPECT_EQ(strides.at(2), 1);
}

TEST(StridesAccess, AtNegativeIndices)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(strides.at(-3), 12);
    EXPECT_EQ(strides.at(-2), 4);
    EXPECT_EQ(strides.at(-1), 1);
}

TEST(StridesAccess, AtOutOfRange)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_THROW(static_cast<void>(strides.at(-4)), Exceptions::IndexError);
    EXPECT_THROW(static_cast<void>(strides.at(3)), Exceptions::IndexError);
}

TEST(StridesAccess, AtEmpty)
{
    const Strides strides;

    EXPECT_THROW(static_cast<void>(strides.at(0)), Exceptions::IndexError);
    EXPECT_THROW(static_cast<void>(strides.at(-1)), Exceptions::IndexError);
}

TEST(StridesIterator, BeginEnd)
{
    Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(*strides.begin(), 12);
    EXPECT_EQ(strides.end() - strides.begin(), 3);
    *strides.begin() = 99;
    EXPECT_EQ(strides.front(), 99);

    static_assert(
        std::same_as<decltype(std::declval<Strides&>().begin()), Strides::iterator>
    );
    static_assert(
        std::same_as<decltype(std::declval<Strides&>().end()), Strides::iterator>
    );
}

TEST(StridesIterator, ConstBeginEnd)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(*strides.begin(), 12);
    EXPECT_EQ(strides.end() - strides.begin(), 3);

    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>().begin()),
            Strides::const_iterator
        >
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>().end()),
            Strides::const_iterator
        >
    );
}

TEST(StridesIterator, CBeginCEnd)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(*strides.cbegin(), 12);
    EXPECT_EQ(strides.cend() - strides.cbegin(), 3);
}

TEST(StridesIterator, ForwardTraversal)
{
    const Strides strides(Shape{2, 3, 4});
    const Strides::value_type expected[]{12, 4, 1};

    Strides::size_type index = 0;
    for (Strides::value_type stride : strides) {
        EXPECT_EQ(stride, expected[index]);
        ++index;
    }

    EXPECT_EQ(index, strides.rank());
}

TEST(StridesIterator, ReverseTraversal)
{
    Strides strides(Shape{2, 3, 4});
    const Strides::value_type expected[]{1, 4, 12};

    Strides::size_type index = 0;
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
        EXPECT_EQ(*it, expected[index]);
        ++index;
    }

    *strides.rbegin() = 99;
    EXPECT_EQ(strides.back(), 99);
}

TEST(StridesIterator, ConstReverseTraversal)
{
    const Strides strides(Shape{2, 3, 4});

    EXPECT_EQ(*strides.rbegin(), 1);
    EXPECT_EQ(strides.rend() - strides.rbegin(), 3);
    EXPECT_EQ(*strides.crbegin(), 1);
    EXPECT_EQ(strides.crend() - strides.crbegin(), 3);

    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>().crbegin()),
            Strides::const_reverse_iterator
        >
    );
    static_assert(
        std::same_as<
            decltype(std::declval<const Strides&>().crend()),
            Strides::const_reverse_iterator
        >
    );
}

TEST(StridesIterator, Empty)
{
    Strides strides;
    const Strides& const_strides = strides;

    EXPECT_EQ(strides.begin(), strides.end());
    EXPECT_EQ(strides.rbegin(), strides.rend());
    EXPECT_EQ(const_strides.begin(), const_strides.end());
    EXPECT_EQ(const_strides.cbegin(), const_strides.cend());
    EXPECT_EQ(const_strides.rbegin(), const_strides.rend());
    EXPECT_EQ(const_strides.crbegin(), const_strides.crend());
}

TEST(StridesComparison, Equal)
{
    EXPECT_EQ(Strides(Shape{2, 3, 4}), Strides(Shape{2, 3, 4}));
    EXPECT_EQ(Strides(), Strides());
}

TEST(StridesComparison, DifferentValues)
{
    EXPECT_NE(Strides(Shape{2, 3, 4}), Strides(Shape{2, 4, 3}));
}

TEST(StridesComparison, DifferentRanks)
{
    EXPECT_NE(Strides(Shape{2, 3}), Strides(Shape{2, 3, 1}));
    EXPECT_NE(Strides(), Strides(Shape{1}));
}

TEST(StridesModifier, Swap)
{
    Strides lhs(Shape{2, 3});
    Strides rhs(Shape{5, 7, 11});
    auto* lhs_data = lhs.data();
    auto* rhs_data = rhs.data();

    lhs.swap(rhs);

    EXPECT_EQ(lhs, (Strides{Shape{5, 7, 11}}));
    EXPECT_EQ(rhs, (Strides{Shape{2, 3}}));
    EXPECT_EQ(lhs.data(), rhs_data);
    EXPECT_EQ(rhs.data(), lhs_data);
}

TEST(StridesModifier, SwapWithEmpty)
{
    Strides lhs(Shape{2, 3});
    Strides rhs;

    lhs.swap(rhs);

    EXPECT_TRUE(lhs.empty());
    EXPECT_EQ(rhs, (Strides{Shape{2, 3}}));
}

TEST(StridesModifier, SelfSwap)
{
    Strides strides(Shape{2, 3, 4});
    auto* original_data = strides.data();

    strides.swap(strides);

    EXPECT_EQ(strides.data(), original_data);
    EXPECT_EQ(strides, (Strides{Shape{2, 3, 4}}));
}

TEST(StridesStream, MultipleDimensions)
{
    std::ostringstream stream;

    stream << Strides(Shape{2, 3, 4});

    EXPECT_EQ(stream.str(), "(12, 4, 1)");
}

TEST(StridesStream, SingleDimension)
{
    std::ostringstream stream;

    stream << Strides(Shape{5});

    EXPECT_EQ(stream.str(), "(1,)");
}

TEST(StridesStream, Empty)
{
    std::ostringstream stream;

    stream << Strides{};

    EXPECT_EQ(stream.str(), "()");
}

TEST(StridesStream, ReturnsOutputStream)
{
    std::ostringstream stream;

    std::ostream& result = (stream << Strides(Shape{2, 3}));

    EXPECT_EQ(&result, &stream);
}
