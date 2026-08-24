#include <gtest/gtest.h>

#include <stratax/exceptions/Exceptions.hpp>

#include <type_traits>

static_assert(std::is_base_of_v<
	std::runtime_error,
	Exceptions::StrataxError>);
static_assert(std::is_base_of_v<
	Exceptions::StrataxError,
	Exceptions::IndexError>);
static_assert(std::is_base_of_v<
	Exceptions::StrataxError,
	Exceptions::RankError>);
static_assert(std::is_base_of_v<
	Exceptions::StrataxError,
	Exceptions::BroadcastError>);

TEST(Exceptions, PreserveMessages)
{
	const Exceptions::IndexError error("Index is out of bounds.");
	EXPECT_STREQ(error.what(), "Index is out of bounds.");
}

TEST(Exceptions, RankErrorPreservesMessage)
{
	const Exceptions::RankError error("Rank does not match.");
	EXPECT_STREQ(error.what(), "Rank does not match.");
}

TEST(Exceptions, CatchSpecializedErrorsThroughBase)
{
	EXPECT_THROW(
		throw Exceptions::ShapeError("Invalid shape."),
		Exceptions::StrataxError);
}
