#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <stratax.h>

using namespace stratax::core;

TEST(SliceConstructor, DefaultStep)
{
	const Slice slice{2, 8};

	EXPECT_EQ(slice.start(), 2);
	EXPECT_EQ(slice.stop(), 8);
	EXPECT_EQ(slice.step(), 1);
	EXPECT_EQ(slice.size(), 6);
}

TEST(SliceConstructor, PositiveStep)
{
	const Slice slice{2, 10, 2};

	EXPECT_EQ(slice.start(), 2);
	EXPECT_EQ(slice.stop(), 10);
	EXPECT_EQ(slice.step(), 2);
	EXPECT_EQ(slice.size(), 4);
	EXPECT_FALSE(slice.empty());
}

TEST(SliceConstructor, NegativeStep)
{
	const Slice slice{10, 2, -2};

	EXPECT_EQ(slice.start(), 10);
	EXPECT_EQ(slice.stop(), 2);
	EXPECT_EQ(slice.step(), -2);
	EXPECT_EQ(slice.size(), 4);
	EXPECT_FALSE(slice.empty());
}

TEST(SliceConstructor, PreservesNegativeBounds)
{
	const Slice slice{-8, -2, 2};

	EXPECT_EQ(slice.start(), -8);
	EXPECT_EQ(slice.stop(), -2);
	EXPECT_EQ(slice.step(), 2);
	EXPECT_EQ(slice.size(), 3);
}

TEST(SliceConstructor, RejectsZeroStep)
{
	EXPECT_THROW(static_cast<void>(Slice(0, 10, 0)), Exceptions::IndexError);
}

TEST(SliceConstructor, ZeroStepErrorMessage)
{
	try {
		static_cast<void>(Slice(0, 10, 0));
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(error.what(), "Slice step cannot be zero.");
	}
}

TEST(SliceConstructor, Copy)
{
	const Slice original{2, 10, 2};
	const Slice copy(original);

	EXPECT_EQ(copy, original);
	EXPECT_EQ(copy.start(), 2);
	EXPECT_EQ(copy.stop(), 10);
	EXPECT_EQ(copy.step(), 2);
}

TEST(SliceConstructor, Move)
{
	Slice source{2, 10, 2};
	const Slice destination(std::move(source));

	EXPECT_EQ(destination, Slice(2, 10, 2));
	EXPECT_EQ(source, Slice(2, 10, 2));
}

TEST(SliceAssignment, CopyAssignment)
{
	const Slice source{2, 10, 2};
	Slice destination{0, 4};

	destination = source;

	EXPECT_EQ(destination, source);
}

TEST(SliceAssignment, SelfCopyAssignment)
{
	Slice slice{2, 10, 2};

	slice = slice;

	EXPECT_EQ(slice, Slice(2, 10, 2));
}

TEST(SliceAssignment, MoveAssignment)
{
	Slice source{2, 10, 2};
	Slice destination{0, 4};

	destination = std::move(source);

	EXPECT_EQ(destination, Slice(2, 10, 2));
	EXPECT_EQ(source, Slice(2, 10, 2));
}

TEST(SliceAssignment, SelfMoveAssignment)
{
	Slice slice{2, 10, 2};

	slice = std::move(slice);

	EXPECT_EQ(slice, Slice(2, 10, 2));
}

TEST(SliceTypes, Aliases)
{
	static_assert(std::same_as<Slice::size_type, std::size_t>);
	static_assert(std::same_as<Slice::difference_type, std::ptrdiff_t>);
	static_assert(std::is_copy_constructible_v<Slice>);
	static_assert(std::is_move_constructible_v<Slice>);
	static_assert(std::is_copy_assignable_v<Slice>);
	static_assert(std::is_move_assignable_v<Slice>);
	static_assert(!std::default_initializable<Slice>);
}

TEST(SliceAccess, AccessorTypesAndExceptionGuarantees)
{
	const Slice slice{2, 10, 2};

	static_assert(std::same_as<decltype(slice.start()), Slice::difference_type>);
	static_assert(std::same_as<decltype(slice.stop()), Slice::difference_type>);
	static_assert(std::same_as<decltype(slice.step()), Slice::difference_type>);
	static_assert(std::same_as<decltype(slice.size()), Slice::size_type>);
	static_assert(std::same_as<decltype(slice.empty()), bool>);
	static_assert(noexcept(slice.start()));
	static_assert(noexcept(slice.stop()));
	static_assert(noexcept(slice.step()));
	static_assert(noexcept(slice.size()));
	static_assert(noexcept(slice.empty()));
}

TEST(SliceSize, PositiveUnitStep)
{
	EXPECT_EQ(Slice(0, 1).size(), 1);
	EXPECT_EQ(Slice(0, 5).size(), 5);
	EXPECT_EQ(Slice(-5, 5).size(), 10);
	EXPECT_EQ(Slice(-5, -1).size(), 4);
}

TEST(SliceSize, PositiveStepExactDivision)
{
	EXPECT_EQ(Slice(0, 10, 2).size(), 5);
	EXPECT_EQ(Slice(1, 10, 3).size(), 3);
	EXPECT_EQ(Slice(-6, 6, 3).size(), 4);
}

TEST(SliceSize, PositiveStepRoundsUp)
{
	EXPECT_EQ(Slice(0, 10, 3).size(), 4);
	EXPECT_EQ(Slice(2, 11, 3).size(), 3);
	EXPECT_EQ(Slice(-5, 5, 4).size(), 3);
}

TEST(SliceSize, NegativeUnitStep)
{
	EXPECT_EQ(Slice(5, 0, -1).size(), 5);
	EXPECT_EQ(Slice(5, -5, -1).size(), 10);
	EXPECT_EQ(Slice(-1, -5, -1).size(), 4);
}

TEST(SliceSize, NegativeStepExactDivision)
{
	EXPECT_EQ(Slice(10, 0, -2).size(), 5);
	EXPECT_EQ(Slice(10, 1, -3).size(), 3);
	EXPECT_EQ(Slice(6, -6, -3).size(), 4);
}

TEST(SliceSize, NegativeStepRoundsUp)
{
	EXPECT_EQ(Slice(10, 0, -3).size(), 4);
	EXPECT_EQ(Slice(10, 2, -3).size(), 3);
	EXPECT_EQ(Slice(5, -5, -4).size(), 3);
}

TEST(SliceSize, EqualBoundsAreEmpty)
{
	EXPECT_EQ(Slice(0, 0, 1).size(), 0);
	EXPECT_EQ(Slice(5, 5, 3).size(), 0);
	EXPECT_EQ(Slice(0, 0, -1).size(), 0);
	EXPECT_EQ(Slice(-5, -5, -3).size(), 0);
}

TEST(SliceSize, PositiveStepInWrongDirectionIsEmpty)
{
	EXPECT_TRUE(Slice(5, 0, 1).empty());
	EXPECT_TRUE(Slice(5, 0, 3).empty());
	EXPECT_TRUE(Slice(-1, -5, 2).empty());
}

TEST(SliceSize, NegativeStepInWrongDirectionIsEmpty)
{
	EXPECT_TRUE(Slice(0, 5, -1).empty());
	EXPECT_TRUE(Slice(0, 5, -3).empty());
	EXPECT_TRUE(Slice(-5, -1, -2).empty());
}

TEST(SliceComparison, EqualStoredValues)
{
	const Slice lhs{0, 10, 2};
	const Slice rhs{0, 10, 2};

	EXPECT_EQ(lhs, rhs);
	EXPECT_FALSE(lhs != rhs);
	static_assert(noexcept(lhs == rhs));
}

TEST(SliceComparison, DifferentStart)
{
	EXPECT_NE(Slice(0, 10, 2), Slice(1, 10, 2));
}

TEST(SliceComparison, DifferentStop)
{
	EXPECT_NE(Slice(0, 10, 2), Slice(0, 11, 2));
}

TEST(SliceComparison, DifferentStep)
{
	EXPECT_NE(Slice(0, 10, 2), Slice(0, 10, 3));
}

TEST(SliceComparison, ComparesRawRepresentation)
{
	const Slice first{0, 5, 2};
	const Slice same_indices_different_stop{0, 6, 2};

	EXPECT_EQ(first.size(), same_indices_different_stop.size());
	EXPECT_NE(first, same_indices_different_stop);
}
