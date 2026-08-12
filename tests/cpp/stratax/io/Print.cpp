#include <gtest/gtest.h>
#include <sstream>

#include <stratax/io/Print.hpp>

#include <stratax.h>

using namespace stratax::container;

TEST(IoPrint, HeaderIsAvailable) { EXPECT_TRUE(true); }

TEST(IoPrint, VectorOutput)
{
	Vector<int> vector{1, 2, 3};
	std::ostringstream out;

	out << vector;

	EXPECT_EQ(out.str(), "[1, 2, 3]");
}

TEST(IoPrint, MatrixOutput)
{
	Matrix<int> matrix{{1, 2}, {3, 4}};
	std::ostringstream out;

	out << matrix;

	EXPECT_EQ(out.str(), "[\n    [1, 2]\n    [3, 4]\n]");
}

TEST(IoPrint, TensorOutput)
{
	Tensor<int> tensor({2, 2, 2}, 1);
	std::ostringstream out;

	out << tensor;

	EXPECT_EQ(
		out.str(),
		"[\n"
		"    [\n"
		"        [1, 1],\n"
		"        [1, 1]\n"
		"    ],\n"
		"    [\n"
		"        [1, 1],\n"
		"        [1, 1]\n"
		"    ]\n"
		"]");
}

TEST(IoPrint, EmptyMatrixOutput)
{
	Matrix<int> matrix(0, 0);
	std::ostringstream out;

	out << matrix;

	EXPECT_EQ(out.str(), "[]");
}
