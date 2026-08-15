#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include <stratax.h>

TEST(Shape, construction_and_metadata)
{
	stratax::Shape shape{2, 3, 4};

	EXPECT_EQ(shape.rank(), 3);
	EXPECT_EQ(shape[0], 2);
	EXPECT_EQ(shape[1], 3);
	EXPECT_EQ(shape[2], 4);
	EXPECT_FALSE(shape.empty());
	EXPECT_EQ(shape.elements(), 24);
	EXPECT_EQ(shape.at(-3), 2);
	EXPECT_EQ(shape.at(-1), 4);
	EXPECT_THROW(shape.at(-4), Exceptions::IndexError);
	EXPECT_THROW(shape.at(3), Exceptions::IndexError);
}

TEST(Shape, comparison_and_copy)
{
	stratax::Shape a{2, 3};
	stratax::Shape b{2, 3};
	stratax::Shape c{3, 2};
	stratax::Shape d{2, 3, 1};

	EXPECT_TRUE(a == b);
	EXPECT_FALSE(a == c);
	EXPECT_FALSE(a == d);
	EXPECT_FALSE(a != b);
	EXPECT_TRUE(a != c);

	const auto copy = a;
	EXPECT_EQ(copy, a);
	EXPECT_EQ(copy.rank(), a.rank());
	EXPECT_EQ(copy.elements(), a.elements());
}

TEST(Shape, iteration)
{
	stratax::Shape shape{2, 3, 4};

	std::size_t index = 0;
	for (auto it = shape.begin(); it != shape.end(); ++it)
	{
		EXPECT_EQ(*it, shape[index]);
		++index;
	}
	EXPECT_EQ(index, shape.rank());

	const auto& const_shape = shape;
	index = 0;
	for (auto it = const_shape.cbegin(); it != const_shape.cend(); ++it)
	{
		EXPECT_EQ(*it, const_shape[index]);
		++index;
	}
	EXPECT_EQ(index, const_shape.rank());

	index = shape.rank();
	for (auto it = shape.rbegin(); it != shape.rend(); ++it)
	{
		--index;
		EXPECT_EQ(*it, shape[index]);
	}
	EXPECT_EQ(index, 0);

	stratax::Shape empty;
	EXPECT_TRUE(empty.empty());
	EXPECT_EQ(empty.begin(), empty.end());
	EXPECT_EQ(empty.cbegin(), empty.cend());
	EXPECT_EQ(empty.rbegin(), empty.rend());
}

TEST(Shape, empty_shape)
{
	stratax::Shape empty;
	EXPECT_TRUE(empty.empty());
	EXPECT_EQ(empty.rank(), 0);
	EXPECT_EQ(empty.elements(), 0);
	EXPECT_EQ(empty.begin(), empty.end());
	EXPECT_THROW(empty.at(0), Exceptions::IndexError);
}

TEST(Shape, stream_output)
{
	std::ostringstream os;
	os << stratax::Shape{2, 3};
	EXPECT_EQ(os.str(), "(2, 3)");

	std::ostringstream os2;
	os2 << stratax::Shape{5};
	EXPECT_EQ(os2.str(), "(5,)");

	std::ostringstream os3;
	os3 << stratax::Shape{};
	EXPECT_EQ(os3.str(), "()");
}

TEST(Shape, swap)
{
	stratax::Shape a{2, 3};
	stratax::Shape b{5, 7, 11};

	a.swap(b);
	EXPECT_EQ(a, (stratax::Shape{5, 7, 11}));
	EXPECT_EQ(b, (stratax::Shape{2, 3}));
}
