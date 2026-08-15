#include <gtest/gtest.h>
#include <stratax.h>

using namespace stratax::container;

TEST(Vector, methods)
{
	Vector<int> vector(5, 0);

	{
		EXPECT_EQ(vector.size(), 5);
		EXPECT_FALSE(vector.empty());
		EXPECT_EQ(vector.rank(), 1);
		EXPECT_EQ(vector.shape(), (stratax::Shape{5}));
		EXPECT_EQ(vector.strides(), stratax::Strides(vector.shape()));
	}

	{
		EXPECT_NE(vector.data(), nullptr);
		vector.data()[0] = 10;
		vector.data()[2] = 30;
		vector.data()[4] = 50;
		EXPECT_EQ(vector.data()[0], 10);
		EXPECT_EQ(vector.front(), 10);
		EXPECT_EQ(vector.back(), 50);
	}

	{
		const Vector<int>& const_vector = vector;
		EXPECT_NE(const_vector.data(), nullptr);
		EXPECT_EQ(const_vector.data()[2], 30);
		EXPECT_EQ(const_vector.front(), 10);
		EXPECT_EQ(const_vector.back(), 50);
	}

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
	Vector<int> vector(5, 0);

	{
		vector[0] = 11;
		vector[4] = 55;
		EXPECT_EQ(vector[0], 11);
		EXPECT_EQ(vector[4], 55);
	}

	{
		EXPECT_EQ(vector.at(0), 11);
		EXPECT_EQ(vector.at(-1), 55);
		EXPECT_EQ(vector.at(-5), 11);
		EXPECT_THROW(vector.at(5), Exceptions::IndexError);
		EXPECT_THROW(vector.at(-6), Exceptions::IndexError);
	}

	{
		const Vector<int>& const_vector = vector;
		EXPECT_EQ(const_vector.at(2), 0);
		EXPECT_EQ(const_vector.at(-2), 0);
	}
}

TEST(Vector, iterators)
{
	Vector<int> vector(5, 0);
	for (std::size_t i = 0; i < vector.size(); ++i)
	{
		vector[i] = static_cast<int>(i + 1);
	}

	{
		std::size_t index = 0;
		for (auto it = vector.begin(); it != vector.end(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index + 1));
			++index;
		}
		EXPECT_EQ(index, vector.size());
	}

	{
		const Vector<int>& const_vector = vector;
		std::size_t index = 0;
		for (auto it = const_vector.cbegin(); it != const_vector.cend(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index + 1));
			++index;
		}
		EXPECT_EQ(index, const_vector.size());
	}

	{
		std::size_t index = vector.size();
		for (auto it = vector.rbegin(); it != vector.rend(); ++it)
		{
			--index;
			EXPECT_EQ(*it, static_cast<int>(index + 1));
		}
		EXPECT_EQ(index, 0);
	}

	{
		Vector<int> empty(0);
		EXPECT_EQ(empty.begin(), empty.end());
		EXPECT_EQ(empty.cbegin(), empty.cend());
		EXPECT_EQ(empty.rbegin(), empty.rend());
	}
}

TEST(Vector, constructors)
{
	Vector<int> size_ctor(4);
	EXPECT_EQ(size_ctor.size(), 4);
	for (std::size_t i = 0; i < size_ctor.size(); ++i)
	{
		EXPECT_EQ(size_ctor[i], 0);
	}

	Vector<int> value_ctor(3, 7);
	EXPECT_EQ(value_ctor.size(), 3);
	for (std::size_t i = 0; i < value_ctor.size(); ++i)
	{
		EXPECT_EQ(value_ctor[i], 7);
	}

	EXPECT_THROW(([]() { Vector<int> invalid_shape(stratax::Shape{2, 3}); })(), Exceptions::DimensionError);

	Vector<int> shape_ctor(stratax::Shape{4});
	EXPECT_EQ(shape_ctor.size(), 4);
	EXPECT_EQ(shape_ctor.rank(), 1);
	EXPECT_EQ(shape_ctor.shape(), (stratax::Shape{4}));

	Vector<int> init_list_ctor{1, 2, 3, 4};
	EXPECT_EQ(init_list_ctor.size(), 4);
	EXPECT_EQ(init_list_ctor.shape(), (stratax::Shape{4}));
	EXPECT_EQ(init_list_ctor[0], 1);
	EXPECT_EQ(init_list_ctor[3], 4);

	Vector<int> default_ctor;
	EXPECT_EQ(default_ctor.size(), 0);
	EXPECT_TRUE(default_ctor.empty());
}

TEST(Vector, modifiers)
{
	Vector<int> vector(5, 0);
	Vector<int> other(5, 1);

	{
		vector.fill(7);
		for (std::size_t i = 0; i < vector.size(); ++i)
		{
			EXPECT_EQ(vector[i], 7);
		}
	}

	{
		vector.swap(other);
		for (std::size_t i = 0; i < vector.size(); ++i)
		{
			EXPECT_EQ(vector[i], 1);
			EXPECT_EQ(other[i], 7);
		}
	}
}
