#include <gtest/gtest.h>

#include <stratax/exceptions/Exceptions.hpp>

TEST(ExceptionStructure, SpecializedErrorsShareTheSimpleBase)
{
	EXPECT_THROW(
		throw Exceptions::OverflowError("Overflow."),
		Exceptions::StrataxError);
}
