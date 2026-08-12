#include <gtest/gtest.h>
#include <stratax.h>

using namespace stratax::container;

TEST(Tensor, methods)
{
	Tensor<int> tensor({2, 3, 4}, 1);

	// Metadata
	{
		EXPECT_EQ(tensor.size(), 24);
		EXPECT_FALSE(tensor.empty());
		EXPECT_EQ(tensor.rank(), 3);
		EXPECT_EQ(tensor.shape(), (stratax::Shape{2, 3, 4}));
		EXPECT_EQ(tensor.strides(), stratax::Strides(tensor.shape()));
	}

	// Data access
	{
		EXPECT_NE(tensor.data(), nullptr);
		tensor.data()[1] = 42;
		tensor.data()[0] = 10;
		tensor.data()[tensor.size() - 1] = 20;
		EXPECT_EQ(tensor.data()[1], 42);
		EXPECT_EQ(tensor.front(), 10);
		EXPECT_EQ(tensor.back(), 20);
	}

	// Const access
	{
		const Tensor<int>& const_tensor = tensor;
		EXPECT_NE(const_tensor.data(), nullptr);
		EXPECT_EQ(const_tensor.data()[1], 42);
		EXPECT_EQ(const_tensor.front(), 10);
		EXPECT_EQ(const_tensor.back(), 20);
	}

	// Empty tensor
	{
		Tensor<int> empty(stratax::Shape{});
		EXPECT_EQ(empty.size(), 0);
		EXPECT_TRUE(empty.empty());
		EXPECT_THROW(empty.front(), Exceptions::IndexError);
		EXPECT_THROW(empty.back(), Exceptions::IndexError);
	}
}

TEST(Tensor, operators)
{
	Tensor<int> tensor({2, 3, 4}, 0);

	// Mutable multidimensional access
	{
		tensor(1, 2, 3) = 21;
		std::vector<std::size_t> idx{1, 2, 3};
		EXPECT_EQ(tensor(1, 2, 3), 21);
		EXPECT_EQ(tensor(idx), 21);
		EXPECT_THROW(tensor(2, 0, 0), Exceptions::IndexError);
		EXPECT_THROW(tensor(std::vector<std::size_t>{1, 2}), Exceptions::DimensionError);
	}

	// Const access
	{
		const Tensor<int>& const_tensor = tensor;
		std::vector<std::size_t> idx{1, 2, 3};
		EXPECT_EQ(const_tensor(1, 2, 3), 21);
		EXPECT_EQ(const_tensor(idx), 21);
	}

	// Checked access
	{
		EXPECT_EQ(tensor.at(1, 2, 3), 21);
		EXPECT_EQ(tensor.at(-1, -1, -1), 21);
		EXPECT_THROW(tensor.at(0, 0), Exceptions::IndexError);
		EXPECT_THROW(tensor.at(2, 0, 0), Exceptions::IndexError);
		EXPECT_THROW(tensor.at(0, 0, 4), Exceptions::IndexError);
	}
}

TEST(Tensor, iterators)
{
	Tensor<int> tensor({2, 3, 4}, 0);

	for (std::size_t i = 0; i < tensor.size(); ++i)
	{
		tensor[i] = static_cast<int>(i);
	}

	// Forward iterators
	{
		std::size_t index = 0;
		for (auto it = tensor.begin(); it != tensor.end(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index));
			++index;
		}
		EXPECT_EQ(index, tensor.size());

		const Tensor<int>& const_tensor = tensor;
		index = 0;
		for (auto it = const_tensor.cbegin(); it != const_tensor.cend(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index));
			++index;
		}
		EXPECT_EQ(index, const_tensor.size());
	}

	// Reverse iterators
	{
		std::size_t index = tensor.size();
		for (auto it = tensor.rbegin(); it != tensor.rend(); ++it)
		{
			--index;
			EXPECT_EQ(*it, static_cast<int>(index));
		}
		EXPECT_EQ(index, 0);
	}

	// Empty tensor
	{
		Tensor<int> empty(stratax::Shape{});
		EXPECT_EQ(empty.begin(), empty.end());
		EXPECT_EQ(empty.cbegin(), empty.cend());
		EXPECT_EQ(empty.rbegin(), empty.rend());
	}
}

TEST(Tensor, modifiers)
{
	Tensor<int> tensor({2, 3, 4}, 0);
	Tensor<int> other({2, 3, 4}, 1);

	// Fill
	{
		tensor.fill(7);
		for (std::size_t i = 0; i < tensor.size(); ++i)
		{
			EXPECT_EQ(tensor[i], 7);
		}
	}

	// Swap
	{
		tensor.swap(other);
		for (std::size_t i = 0; i < tensor.size(); ++i)
		{
			EXPECT_EQ(tensor[i], 1);
			EXPECT_EQ(other[i], 7);
		}
	}
}
