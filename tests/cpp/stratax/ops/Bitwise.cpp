#include <gtest/gtest.h>

#include <concepts>
#include <type_traits>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(BitwiseUnary, Not)
{
	const Vector<unsigned int> source{0U, 1U, 0x0FU};
	const auto result = ~source;

	static_assert(std::same_as<
		std::remove_cv_t<decltype(result)>, Vector<unsigned int>>);
	EXPECT_EQ(result.shape(), source.shape());
	EXPECT_EQ(result[0], ~0U);
	EXPECT_EQ(result[1], ~1U);
	EXPECT_EQ(result[2], ~0x0FU);
}

TEST(BitwiseArray, AndOrXor)
{
	const Vector<unsigned int> lhs{0b1100U, 0b1010U, 0b1111U};
	const Vector<unsigned int> rhs{0b1010U, 0b1100U, 0b0101U};
	const auto anded = lhs & rhs;
	const auto ored = lhs | rhs;
	const auto xored = lhs ^ rhs;

	EXPECT_EQ(std::vector<unsigned int>(anded.begin(), anded.end()),
		(std::vector<unsigned int>{0b1000U, 0b1000U, 0b0101U}));
	EXPECT_EQ(std::vector<unsigned int>(ored.begin(), ored.end()),
		(std::vector<unsigned int>{0b1110U, 0b1110U, 0b1111U}));
	EXPECT_EQ(std::vector<unsigned int>(xored.begin(), xored.end()),
		(std::vector<unsigned int>{0b0110U, 0b0110U, 0b1010U}));
}

TEST(BitwiseArray, PerElementShifts)
{
	const Vector<unsigned int> values{1U, 2U, 16U};
	const Vector<unsigned int> counts{1U, 2U, 3U};
	const auto left = values << counts;
	const auto right = values >> counts;

	EXPECT_EQ(std::vector<unsigned int>(left.begin(), left.end()),
		(std::vector<unsigned int>{2U, 8U, 128U}));
	EXPECT_EQ(std::vector<unsigned int>(right.begin(), right.end()),
		(std::vector<unsigned int>{2U, 8U, 128U}));
}

TEST(BitwiseArray, PreservesMatrixAndTensorShapes)
{
	const Matrix<int> matrix{{1, 2}, {3, 4}};
	const Tensor<int> tensor(Shape{2, 1, 2}, 3);

	EXPECT_EQ((matrix | 1).shape(), Shape({2, 2}));
	EXPECT_EQ((tensor ^ 1).shape(), Shape({2, 1, 2}));
}

TEST(BitwiseArray, ReturnsIndependentStorage)
{
	Vector<int> lhs{1, 2, 3};
	const Vector<int> rhs{3, 3, 3};
	auto result = lhs & rhs;

	EXPECT_NE(result.data(), lhs.data());
	EXPECT_NE(result.data(), rhs.data());
	result[0] = 99;
	EXPECT_EQ(lhs[0], 1);
}

TEST(BitwiseArray, BroadcastsCompatibleShapes)
{
	const Matrix<int> lhs(2, 1);
	const Matrix<int> rhs(2, 3);

	EXPECT_EQ((lhs & rhs).shape(), Shape({2, 3}));
	EXPECT_EQ((lhs | rhs).shape(), Shape({2, 3}));
	EXPECT_EQ((lhs ^ rhs).shape(), Shape({2, 3}));
	EXPECT_EQ((lhs << rhs).shape(), Shape({2, 3}));
	EXPECT_EQ((lhs >> rhs).shape(), Shape({2, 3}));
}

TEST(BitwiseArray, RejectsIncompatibleShapes)
{
	const Vector<int> lhs{1, 2};
	const Vector<int> rhs{1, 2, 3};

	try {
		static_cast<void>(lhs & rhs);
		FAIL() << "Expected Exceptions::BroadcastError";
	} catch (const Exceptions::BroadcastError& error) {
		EXPECT_STREQ(error.what(), "Shapes are not broadcastable");
	}
}

TEST(BitwiseScalar, RejectsNegativeShift)
{
    const Vector<unsigned int> source{4U, 8U, 15U};

    EXPECT_THROW(
        source << -1,
        Exceptions::StrataxError
    );

    EXPECT_THROW(
        source >> -1,
        Exceptions::StrataxError
    );
}

TEST(BitwiseScalar, RejectsShiftEqualToBitWidth)
{
    const Vector<unsigned int> source{4U, 8U, 15U};

    constexpr auto width =
        std::numeric_limits<unsigned int>::digits;

    EXPECT_THROW(
        source << width,
        Exceptions::StrataxError
    );

    EXPECT_THROW(
        source >> width,
        Exceptions::StrataxError
    );
}

TEST(BitwiseScalar, ScalarOnLeftPreservesShiftOrder)
{
	const Vector<unsigned int> source{1U, 2U, 3U};

	const auto left = 8U << source;
	const auto right = 64U >> source;

	EXPECT_EQ(std::vector<unsigned int>(left.begin(), left.end()),
		(std::vector<unsigned int>{16U, 32U, 64U}));
	EXPECT_EQ(std::vector<unsigned int>(right.begin(), right.end()),
		(std::vector<unsigned int>{32U, 16U, 8U}));
}

TEST(BitwiseScalar, CommutativeScalarFormsMatch)
{
	const Vector<int> source{1, 2, 3};

	EXPECT_TRUE((3 & source) == (source & 3));
	EXPECT_TRUE((3 | source) == (source | 3));
	EXPECT_TRUE((3 ^ source) == (source ^ 3));
}

TEST(BitwiseCompoundArray, AllOperatorsReturnLeftReference)
{
	Vector<unsigned int> value{8U, 16U};
	const Vector<unsigned int> mask{15U, 31U};
	const Vector<unsigned int> counts{1U, 2U};

	EXPECT_EQ(&(value &= mask), &value);
	EXPECT_EQ(&(value |= mask), &value);
	EXPECT_EQ(&(value ^= mask), &value);
	EXPECT_EQ(&(value <<= counts), &value);
	EXPECT_EQ(&(value >>= counts), &value);
}

TEST(BitwiseCompoundScalar, AllOperatorsReturnLeftReference)
{
	Vector<unsigned int> value{8U, 16U};

	EXPECT_EQ(&(value &= 31U), &value);
	EXPECT_EQ(&(value |= 1U), &value);
	EXPECT_EQ(&(value ^= 3U), &value);
	EXPECT_EQ(&(value <<= 1U), &value);
	EXPECT_EQ(&(value >>= 1U), &value);
}

TEST(BitwiseCompound, ShapeFailureLeavesLeftUnchanged)
{
	Vector<int> lhs{1, 2, 3};
	const std::vector<int> original(lhs.begin(), lhs.end());
	const Vector<int> rhs{1, 2};

	EXPECT_THROW(lhs &= rhs, Exceptions::BroadcastError);
	EXPECT_EQ(lhs.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(lhs.begin(), lhs.end()), original);
}

TEST(BitwiseEmpty, OperationsPreserveEmptyShape)
{
	const Tensor<int> lhs(Shape{2, 0, 3});
	const Tensor<int> rhs(Shape{2, 0, 3});

	EXPECT_EQ((~lhs).shape(), lhs.shape());
	EXPECT_EQ((lhs & rhs).shape(), lhs.shape());
	EXPECT_EQ((lhs | 1).shape(), lhs.shape());
	EXPECT_EQ((1 ^ lhs).shape(), lhs.shape());
}
