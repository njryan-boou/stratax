#include <gtest/gtest.h>
#include <stratax.h>

using namespace stratax::container;

TEST(Vector, methods)
{
	Vector<int> vector(6, 1);

	// Metadata
	{
		EXPECT_EQ(vector.size(), 6);
		EXPECT_FALSE(vector.empty());
		EXPECT_EQ(vector.rank(), 1);
		EXPECT_EQ(vector.shape(), (stratax::Shape{6}));
		EXPECT_EQ(vector.strides(), stratax::Strides(vector.shape()));
	}

	// Data access
	{
		EXPECT_NE(vector.data(), nullptr);
		vector.data()[1] = 42;
		vector.data()[0] = 10;
		vector.data()[vector.size() - 1] = 20;
		EXPECT_EQ(vector.data()[1], 42);
		EXPECT_EQ(vector.front(), 10);
		EXPECT_EQ(vector.back(), 20);
	}

	// Const access
	{
		const Vector<int>& const_vector = vector;
		EXPECT_NE(const_vector.data(), nullptr);
		EXPECT_EQ(const_vector.data()[1], 42);
		EXPECT_EQ(const_vector.front(), 10);
		EXPECT_EQ(const_vector.back(), 20);
	}

	// Empty vector
	{
		Vector<int> empty(0);
		EXPECT_EQ(empty.size(), 0);
		EXPECT_TRUE(empty.empty());
		EXPECT_THROW(empty.front(), Exceptions::IndexError);
		EXPECT_THROW(empty.back(), Exceptions::IndexError);
	}
}

TEST(Vector, operators)
{
	Vector<int> vector(6, 0);

	// Mutable flat access
	{
		vector[0] = 5;
		vector[1] = 10;
		vector(vector.size() - 1) = 20;
		EXPECT_EQ(vector[0], 5);
		EXPECT_EQ(vector[1], 10);
		EXPECT_EQ(vector(vector.size() - 1), 20);
	}

	// Const flat access
	{
		const Vector<int>& const_vector = vector;
		EXPECT_EQ(const_vector[0], 5);
		EXPECT_EQ(const_vector[1], 10);
		EXPECT_EQ(const_vector(const_vector.size() - 1), 20);
	}

	// Checked access
	{
		EXPECT_EQ(vector.at(0), 5);
		EXPECT_EQ(vector.at(-1), 20);
		EXPECT_THROW(vector.at(6), Exceptions::IndexError);
		EXPECT_THROW(vector.at(-7), Exceptions::IndexError);
	}
}

TEST(Vector, iterators)
{
	Vector<int> vector(6, 0);

	for (std::size_t i = 0; i < vector.size(); ++i)
	{
		vector[i] = static_cast<int>(i);
	}

	// Forward iterators
	{
		std::size_t index = 0;
		for (auto it = vector.begin(); it != vector.end(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index));
			++index;
		}
		EXPECT_EQ(index, vector.size());

		const Vector<int>& const_vector = vector;
		index = 0;
		for (auto it = const_vector.cbegin(); it != const_vector.cend(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index));
			++index;
		}
		EXPECT_EQ(index, const_vector.size());
	}

	// Reverse iterators
	{
		std::size_t index = vector.size();
		for (auto it = vector.rbegin(); it != vector.rend(); ++it)
		{
			--index;
			EXPECT_EQ(*it, static_cast<int>(index));
		}
		EXPECT_EQ(index, 0);
	}

	// Empty vector
	{
		Vector<int> empty(0);
		EXPECT_EQ(empty.begin(), empty.end());
		EXPECT_EQ(empty.cbegin(), empty.cend());
		EXPECT_EQ(empty.rbegin(), empty.rend());
	}
}

TEST(Vector, modifiers)
{
	Vector<int> vector(6, 0);
	Vector<int> other(6, 1);

	// Fill
	{
		vector.fill(7);
		for (std::size_t i = 0; i < vector.size(); ++i)
		{
			EXPECT_EQ(vector[i], 7);
		}
	}

	// Swap
	{
		vector.swap(other);
		for (std::size_t i = 0; i < vector.size(); ++i)
		{
			EXPECT_EQ(vector[i], 1);
			EXPECT_EQ(other[i], 7);
		}
	}
}
