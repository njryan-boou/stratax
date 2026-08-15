#include <gtest/gtest.h>

#include <array>
#include <vector>

#include <stratax.h>

namespace {

class TestArray : public stratax::core::ArrayBase<int>
{
public:
	using stratax::core::ArrayBase<int>::at;
	using stratax::core::ArrayBase<int>::back;
	using stratax::core::ArrayBase<int>::begin;
	using stratax::core::ArrayBase<int>::cbegin;
	using stratax::core::ArrayBase<int>::cend;
	using stratax::core::ArrayBase<int>::data;
	using stratax::core::ArrayBase<int>::empty;
	using stratax::core::ArrayBase<int>::end;
	using stratax::core::ArrayBase<int>::fill;
	using stratax::core::ArrayBase<int>::front;
	using stratax::core::ArrayBase<int>::operator[];
	using stratax::core::ArrayBase<int>::rbegin;
	using stratax::core::ArrayBase<int>::rend;
	using stratax::core::ArrayBase<int>::rank;
	using stratax::core::ArrayBase<int>::shape;
	using stratax::core::ArrayBase<int>::size;
	using stratax::core::ArrayBase<int>::strides;

	TestArray(const stratax::core::Shape& shape)
		: stratax::core::ArrayBase<int>(shape)
	{}

	TestArray(const stratax::core::Shape& shape, const int& value)
		: stratax::core::ArrayBase<int>(shape, value)
	{}

	TestArray(const stratax::core::Shape& shape, stratax::core::Buffer<int>&& buffer)
		: stratax::core::ArrayBase<int>(shape, std::move(buffer))
	{}

	std::size_t flat_offset_for(const std::vector<std::size_t>& indices) const
	{
		return this->flat_offset(indices);
	}

	std::size_t normalized_flat_offset_for(const std::vector<std::ptrdiff_t>& indices) const
	{
		return this->normalized_flat_offset(
			indices,
			"ArrayBase rank mismatch.",
			"ArrayBase component is out of bounds.");
	}
};

} // namespace

TEST(ArrayBase, metadata)
{
	TestArray array(stratax::Shape{2, 3});

	EXPECT_EQ(array.size(), 6);
	EXPECT_FALSE(array.empty());
	EXPECT_EQ(array.rank(), 2);
	EXPECT_EQ(array.shape(), (stratax::Shape{2, 3}));
	EXPECT_EQ(array.strides(), stratax::Strides(array.shape()));
	EXPECT_NE(array.data(), nullptr);
}

TEST(ArrayBase, data_and_checked_access)
{
	TestArray array(stratax::Shape{2, 3}, 0);
	array[0] = 10;
	array[5] = 20;
	array.data()[2] = 30;

	EXPECT_EQ(array[0], 10);
	EXPECT_EQ(array[5], 20);
	EXPECT_EQ(array.data()[2], 30);
	EXPECT_EQ(array.front(), 10);
	EXPECT_EQ(array.back(), 20);

	const TestArray& const_array = array;
	EXPECT_EQ(const_array.front(), 10);
	EXPECT_EQ(const_array.back(), 20);
	EXPECT_EQ(const_array.at(-6), 10);
	EXPECT_EQ(const_array.at(-1), 20);
	EXPECT_THROW(const_array.at(-7), Exceptions::IndexError);
	EXPECT_THROW(const_array.at(6), Exceptions::IndexError);
}

TEST(ArrayBase, iterators)
{
	TestArray array(stratax::Shape{2, 3}, 0);
	for (std::size_t i = 0; i < array.size(); ++i)
	{
		array[i] = static_cast<int>(i + 1);
	}

	std::size_t index = 0;
	for (auto it = array.begin(); it != array.end(); ++it)
	{
		EXPECT_EQ(*it, static_cast<int>(index + 1));
		++index;
	}
	EXPECT_EQ(index, array.size());

	const TestArray& const_array = array;
	index = 0;
	for (auto it = const_array.cbegin(); it != const_array.cend(); ++it)
	{
		EXPECT_EQ(*it, static_cast<int>(index + 1));
		++index;
	}
	EXPECT_EQ(index, const_array.size());

	index = array.size();
	for (auto it = array.rbegin(); it != array.rend(); ++it)
	{
		--index;
		EXPECT_EQ(*it, static_cast<int>(index + 1));
	}
	EXPECT_EQ(index, 0);

	TestArray empty(stratax::Shape{});
	EXPECT_EQ(empty.begin(), empty.end());
	EXPECT_EQ(empty.cbegin(), empty.cend());
	EXPECT_EQ(empty.rbegin(), empty.rend());
}

TEST(ArrayBase, flat_offset_and_normalized_offset)
{
	TestArray array(stratax::Shape{2, 3}, 0);
	for (std::size_t i = 0; i < array.size(); ++i)
	{
		array[i] = static_cast<int>(i + 1);
	}

	EXPECT_EQ(array.flat_offset_for({0, 0}), 0u);
	EXPECT_EQ(array.flat_offset_for({1, 2}), 5u);
	EXPECT_EQ(array.flat_offset_for({1, 1}), 4u);

	EXPECT_EQ(array.normalized_flat_offset_for({-2, -1}), 5u);
	EXPECT_THROW(array.normalized_flat_offset_for({-3, 0}), Exceptions::IndexError);
	EXPECT_THROW(array.normalized_flat_offset_for({0, 3}), Exceptions::IndexError);
	EXPECT_THROW(array.normalized_flat_offset_for({1}), Exceptions::IndexError);
}

TEST(ArrayBase, constructors_and_fill)
{
	TestArray default_array(stratax::Shape{2, 3});
	for (std::size_t i = 0; i < default_array.size(); ++i)
	{
		EXPECT_EQ(default_array[i], 0);
	}

	TestArray value_array(stratax::Shape{2, 3}, 7);
	for (std::size_t i = 0; i < value_array.size(); ++i)
	{
		EXPECT_EQ(value_array[i], 7);
	}

	stratax::core::Buffer<int> buffer{1, 2, 3, 4, 5, 6};
	TestArray buffer_array(stratax::Shape{2, 3}, std::move(buffer));
	EXPECT_EQ(buffer_array.size(), 6);
	EXPECT_EQ(buffer_array[0], 1);
	EXPECT_EQ(buffer_array[5], 6);

	stratax::core::Buffer<int> bad_buffer{1, 2, 3};
	EXPECT_THROW(([]() { TestArray bad(stratax::Shape{2, 3}, stratax::core::Buffer<int>{1, 2, 3}); })(), Exceptions::ShapeError);

	TestArray fill_array(stratax::Shape{2, 3}, 0);
	fill_array.fill(11);
	for (std::size_t i = 0; i < fill_array.size(); ++i)
	{
		EXPECT_EQ(fill_array[i], 11);
	}
}
