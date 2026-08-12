#include <gtest/gtest.h>
#include <stratax.h>

using namespace stratax::container;

TEST(Matrix, methods)
{
	Matrix<int> matrix(2, 3, 1);

	// Metadata
	{
		EXPECT_EQ(matrix.size(), 6);
		EXPECT_FALSE(matrix.empty());
		EXPECT_EQ(matrix.rank(), 2);
		EXPECT_EQ(matrix.rows(), 2);
		EXPECT_EQ(matrix.cols(), 3);
		EXPECT_EQ(matrix.shape(), (stratax::Shape{2, 3}));
		EXPECT_EQ(matrix.strides(), stratax::Strides(matrix.shape()));
	}

	// Data access
	{
		EXPECT_NE(matrix.data(), nullptr);
		matrix.data()[1] = 42;
		matrix.data()[0] = 10;
		matrix.data()[matrix.size() - 1] = 20;
		EXPECT_EQ(matrix.data()[1], 42);
		EXPECT_EQ(matrix.front(), 10);
		EXPECT_EQ(matrix.back(), 20);
	}

	// Const access
	{
		const Matrix<int>& const_matrix = matrix;
		EXPECT_NE(const_matrix.data(), nullptr);
		EXPECT_EQ(const_matrix.data()[1], 42);
		EXPECT_EQ(const_matrix.front(), 10);
		EXPECT_EQ(const_matrix.back(), 20);
	}

	// Empty matrix
	{
		Matrix<int> empty(0, 0);
		EXPECT_EQ(empty.size(), 0);
		EXPECT_TRUE(empty.empty());
		EXPECT_THROW(empty.front(), Exceptions::IndexError);
		EXPECT_THROW(empty.back(), Exceptions::IndexError);
	}
}

TEST(Matrix, operators)
{
	Matrix<int> matrix(2, 3, 0);

	// Mutable flat access
	{
		matrix[0] = 5;
		matrix[1] = 10;
		matrix(matrix.size() - 1) = 20;
		EXPECT_EQ(matrix[0], 5);
		EXPECT_EQ(matrix[1], 10);
		EXPECT_EQ(matrix(matrix.size() - 1), 20);
	}

	// Mutable 2D access
	{
		matrix(0, 0) = 11;
		matrix(1, 2) = 21;
		EXPECT_EQ(matrix(0, 0), 11);
		EXPECT_EQ(matrix(1, 2), 21);
		EXPECT_THROW(matrix(2, 0), Exceptions::IndexError);
		EXPECT_THROW(matrix(0, 3), Exceptions::IndexError);
	}

	// Const access
	{
		const Matrix<int>& const_matrix = matrix;
		EXPECT_EQ(const_matrix[0], 11);
		EXPECT_EQ(const_matrix(const_matrix.size() - 1), 21);
		EXPECT_EQ(const_matrix(1, 2), 21);
	}

	// Checked access
	{
		EXPECT_EQ(matrix.at(0, 0), 11);
		EXPECT_EQ(matrix.at(-1, -1), 21);
		EXPECT_THROW(matrix.at(2, 0), Exceptions::IndexError);
		EXPECT_THROW(matrix.at(0, 3), Exceptions::IndexError);
		EXPECT_THROW(matrix.at(-3, 0), Exceptions::IndexError);
	}
}

TEST(Matrix, iterators)
{
	Matrix<int> matrix(2, 3, 0);

	for (std::size_t i = 0; i < matrix.size(); ++i)
	{
		matrix[i] = static_cast<int>(i);
	}

	// Forward iterators
	{
		std::size_t index = 0;
		for (auto it = matrix.begin(); it != matrix.end(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index));
			++index;
		}
		EXPECT_EQ(index, matrix.size());

		const Matrix<int>& const_matrix = matrix;
		index = 0;
		for (auto it = const_matrix.cbegin(); it != const_matrix.cend(); ++it)
		{
			EXPECT_EQ(*it, static_cast<int>(index));
			++index;
		}
		EXPECT_EQ(index, const_matrix.size());
	}

	// Reverse iterators
	{
		std::size_t index = matrix.size();
		for (auto it = matrix.rbegin(); it != matrix.rend(); ++it)
		{
			--index;
			EXPECT_EQ(*it, static_cast<int>(index));
		}
		EXPECT_EQ(index, 0);
	}

	// Empty matrix
	{
		Matrix<int> empty(0, 0);
		EXPECT_EQ(empty.begin(), empty.end());
		EXPECT_EQ(empty.cbegin(), empty.cend());
		EXPECT_EQ(empty.rbegin(), empty.rend());
	}
}

TEST(Matrix, modifiers)
{
	Matrix<int> matrix(2, 3, 0);
	Matrix<int> other(2, 3, 1);

	// Fill
	{
		matrix.fill(7);
		for (std::size_t i = 0; i < matrix.size(); ++i)
		{
			EXPECT_EQ(matrix[i], 7);
		}
	}

	// Swap
	{
		matrix.swap(other);
		for (std::size_t i = 0; i < matrix.size(); ++i)
		{
			EXPECT_EQ(matrix[i], 1);
			EXPECT_EQ(other[i], 7);
		}
	}
}
