#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(GlobalReduction, SumAndProduct)
{
	const Vector<int> source{1, 2, 3, 4};

	EXPECT_EQ(reduction::sum(source), 10);
	EXPECT_EQ(reduction::prod(source), 24);
}

TEST(GlobalReduction, Extrema)
{
	const Matrix<int> source{{4, -2, 7}, {7, 3, -2}};

	EXPECT_EQ(reduction::max(source), 7);
	EXPECT_EQ(reduction::min(source), -2);
}

TEST(GlobalReduction, ExtremumIndicesChooseFirstTie)
{
	const Matrix<int> source{{4, -2, 7}, {7, 3, -2}};

	EXPECT_EQ(reduction::argmax(source), 2);
	EXPECT_EQ(reduction::argmin(source), 1);
}

TEST(GlobalReduction, StatisticsUsePopulationDefinition)
{
	const Vector<double> source{1.0, 2.0, 3.0, 4.0};

	EXPECT_DOUBLE_EQ(reduction::mean(source), 2.5);
	EXPECT_DOUBLE_EQ(reduction::var(source), 1.25);
	EXPECT_DOUBLE_EQ(reduction::std(source), std::sqrt(1.25));
}

TEST(GlobalReduction, SingleElementStatistics)
{
	const Tensor<int> source(Shape{1, 1, 1}, 8);

	EXPECT_DOUBLE_EQ(reduction::mean(source), 8.0);
	EXPECT_DOUBLE_EQ(reduction::var(source), 0.0);
	EXPECT_DOUBLE_EQ(reduction::std(source), 0.0);
}

TEST(AxisReduction, SumAcrossMatrixAxes)
{
	const Matrix<int> source{{1, 2, 3}, {4, 5, 6}};
	const auto columns = reduction::sum(source, 0);
	const auto rows = reduction::sum(source, 1);

	EXPECT_EQ(columns.shape(), Shape({3}));
	EXPECT_EQ(rows.shape(), Shape({2}));
	EXPECT_EQ(std::vector<int>(columns.begin(), columns.end()),
		(std::vector<int>{5, 7, 9}));
	EXPECT_EQ(std::vector<int>(rows.begin(), rows.end()),
		(std::vector<int>{6, 15}));
}

TEST(AxisReduction, ProductSupportsNegativeAxis)
{
	const Matrix<int> source{{1, 2, 3}, {4, 5, 6}};
	const auto result = reduction::prod(source, -1);

	EXPECT_EQ(result.shape(), Shape({2}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{6, 120}));
}

TEST(AxisReduction, ExtremaAcrossMatrixAxes)
{
	const Matrix<int> source{{3, 1, 4}, {2, 7, 0}};
	const auto maxima = reduction::max(source, 0);
	const auto minima = reduction::min(source, 1);

	EXPECT_EQ(std::vector<int>(maxima.begin(), maxima.end()),
		(std::vector<int>{3, 7, 4}));
	EXPECT_EQ(std::vector<int>(minima.begin(), minima.end()),
		(std::vector<int>{1, 0}));
}

TEST(AxisReduction, ExtremumIndicesAreRelativeToAxisSlice)
{
	const Matrix<int> source{{3, 1, 4}, {2, 7, 0}};
	const auto maxima = reduction::argmax(source, 1);
	const auto minima = reduction::argmin(source, 0);

	EXPECT_EQ(std::vector<std::size_t>(maxima.begin(), maxima.end()),
		(std::vector<std::size_t>{2, 1}));
	EXPECT_EQ(std::vector<std::size_t>(minima.begin(), minima.end()),
		(std::vector<std::size_t>{1, 0, 1}));
}

TEST(AxisReduction, StatisticsAcrossRows)
{
	const Matrix<int> source{{1, 2, 3}, {2, 4, 6}};
	const auto means = reduction::mean(source, 1);
	const auto variances = reduction::var(source, 1);
	const auto deviations = reduction::std(source, 1);

	EXPECT_EQ(means.shape(), Shape({2}));
	EXPECT_DOUBLE_EQ(means[0], 2.0);
	EXPECT_DOUBLE_EQ(means[1], 4.0);
	EXPECT_DOUBLE_EQ(variances[0], 2.0 / 3.0);
	EXPECT_DOUBLE_EQ(variances[1], 8.0 / 3.0);
	EXPECT_DOUBLE_EQ(deviations[0], std::sqrt(2.0 / 3.0));
	EXPECT_DOUBLE_EQ(deviations[1], std::sqrt(8.0 / 3.0));
}

TEST(AxisReduction, KeepdimsPreservesRank)
{
	const Matrix<int> source{{1, 2, 3}, {4, 5, 6}};
	const auto columns = reduction::sum(source, 0, true);
	const auto rows = reduction::mean(source, 1, true);

	EXPECT_EQ(columns.shape(), Shape({1, 3}));
	EXPECT_EQ(rows.shape(), Shape({2, 1}));
	EXPECT_EQ(std::vector<int>(columns.begin(), columns.end()),
		(std::vector<int>{5, 7, 9}));
	EXPECT_DOUBLE_EQ(rows[0], 2.0);
	EXPECT_DOUBLE_EQ(rows[1], 5.0);
}

TEST(AxisReduction, ThreeDimensionalResultShapeAndValues)
{
	Tensor<int> source(Shape{2, 2, 2});
	for (std::size_t i = 0; i < source.size(); ++i) {
		source[i] = static_cast<int>(i + 1);
	}
	const auto result = reduction::sum(source, 1);

	EXPECT_EQ(result.shape(), Shape({2, 2}));
	EXPECT_EQ(std::vector<int>(result.begin(), result.end()),
		(std::vector<int>{4, 6, 12, 14}));
}

TEST(AxisReduction, RankOneResultUsesSingleElementTensor)
{
	const Vector<int> source{1, 2, 3, 4};
	const auto result = reduction::sum(source, 0);

	EXPECT_EQ(result.shape(), Shape({1}));
	EXPECT_EQ(result[0], 10);
}

TEST(AxisReduction, RejectsOutOfRangeAxes)
{
	const Matrix<int> source{{1, 2}, {3, 4}};

	EXPECT_THROW(reduction::sum(source, 2), Exceptions::AxisError);
	EXPECT_THROW(reduction::sum(source, -3), Exceptions::AxisError);
}

TEST(AxisReduction, AxisErrorMessage)
{
	const Vector<int> source{1, 2, 3};

	try {
		static_cast<void>(reduction::sum(source, 1));
		FAIL() << "Expected Exceptions::AxisError";
	} catch (const Exceptions::AxisError& error) {
		EXPECT_STREQ(error.what(), "axis is out of range.");
	}
}

TEST(EmptyGlobalReduction, SumUsesAdditiveIdentity)
{
	const Vector<int> source;

	EXPECT_EQ(reduction::sum(source), 0);
}

TEST(EmptyGlobalReduction, ProductUsesMultiplicativeIdentity)
{
	const Tensor<double> source(Shape{2, 0, 3});

	EXPECT_EQ(reduction::prod(source), 1.0);
}

TEST(EmptyGlobalReduction, ExtremaThrowIndexError)
{
	const Matrix<int> source(0, 3);

	EXPECT_THROW(reduction::max(source), Exceptions::IndexError);
	EXPECT_THROW(reduction::min(source), Exceptions::IndexError);
}

TEST(EmptyGlobalReduction, ExtremaErrorMessages)
{
	const Vector<int> source;

	try {
		reduction::max(source);
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(error.what(), "Maximum is undefined for an empty array.");
	}

	try {
		reduction::min(source);
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(error.what(), "Minimum is undefined for an empty array.");
	}
}

TEST(EmptyGlobalReduction, ExtremumIndicesThrowIndexError)
{
	const Tensor<int> source(Shape{0});

	EXPECT_THROW(reduction::argmax(source), Exceptions::IndexError);
	EXPECT_THROW(reduction::argmin(source), Exceptions::IndexError);
}

TEST(EmptyGlobalReduction, ExtremumIndexErrorMessages)
{
	const Vector<int> source;

	try {
		reduction::argmax(source);
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(error.what(), "Argmax is undefined for an empty array.");
	}

	try {
		reduction::argmin(source);
		FAIL() << "Expected Exceptions::IndexError";
	} catch (const Exceptions::IndexError& error) {
		EXPECT_STREQ(error.what(), "Argmin is undefined for an empty array.");
	}
}

TEST(EmptyGlobalReduction, StatisticsThrowZeroDivisionError)
{
	const Vector<double> source;

	EXPECT_THROW(reduction::mean(source), Exceptions::ZeroDivisionError);
	EXPECT_THROW(reduction::var(source), Exceptions::ZeroDivisionError);
	EXPECT_THROW(reduction::std(source), Exceptions::ZeroDivisionError);
}

TEST(EmptyGlobalReduction, StatisticErrorMessages)
{
	const Vector<double> source;

	try {
		reduction::mean(source);
		FAIL() << "Expected Exceptions::ZeroDivisionError";
	} catch (const Exceptions::ZeroDivisionError& error) {
		EXPECT_STREQ(error.what(), "Mean is undefined for an empty array.");
	}

	try {
		reduction::var(source);
		FAIL() << "Expected Exceptions::ZeroDivisionError";
	} catch (const Exceptions::ZeroDivisionError& error) {
		EXPECT_STREQ(error.what(), "Variance is undefined for an empty array.");
	}
}

TEST(EmptyAxisReduction, SumAndProductUseIdentities)
{
	const Tensor<int> source(Shape{0, 3});
	const auto sums = reduction::sum(source, 0);
	const auto products = reduction::prod(source, 0);

	EXPECT_EQ(sums.shape(), Shape({3}));
	EXPECT_EQ(products.shape(), Shape({3}));
	EXPECT_EQ(std::vector<int>(sums.begin(), sums.end()),
		(std::vector<int>{0, 0, 0}));
	EXPECT_EQ(std::vector<int>(products.begin(), products.end()),
		(std::vector<int>{1, 1, 1}));
}

TEST(EmptyAxisReduction, UndefinedReductionsThrow)
{
	const Tensor<int> source(Shape{0, 3});

	EXPECT_THROW(reduction::max(source, 0), Exceptions::IndexError);
	EXPECT_THROW(reduction::min(source, 0), Exceptions::IndexError);
	EXPECT_THROW(reduction::argmax(source, 0), Exceptions::IndexError);
	EXPECT_THROW(reduction::argmin(source, 0), Exceptions::IndexError);
	EXPECT_THROW(reduction::mean(source, 0), Exceptions::ZeroDivisionError);
	EXPECT_THROW(reduction::var(source, 0), Exceptions::ZeroDivisionError);
	EXPECT_THROW(reduction::std(source, 0), Exceptions::ZeroDivisionError);
}

TEST(EmptyAxisReduction, EmptyOutputReturnsWithoutReducing)
{
	const Tensor<int> source(Shape{2, 0, 3});
	const auto maxima = reduction::max(source, 0);
	const auto means = reduction::mean(source, 0);
	const auto sums = reduction::sum(source, 0);

	EXPECT_TRUE(maxima.empty());
	EXPECT_EQ(maxima.shape(), Shape({0, 3}));
	EXPECT_TRUE(means.empty());
	EXPECT_EQ(means.shape(), Shape({0, 3}));
	EXPECT_TRUE(sums.empty());
	EXPECT_EQ(sums.shape(), Shape({0, 3}));
}

TEST(EmptyAxisReduction, KeepdimsPreservesEmptyOutputShape)
{
	const Tensor<int> source(Shape{2, 0, 3});
	const auto result = reduction::argmin(source, 2, true);

	EXPECT_TRUE(result.empty());
	EXPECT_EQ(result.shape(), Shape({2, 0, 1}));
}

TEST(EmptyAxisReduction, RankOneIdentitiesProduceSingleElementTensor)
{
	const Vector<int> source;
	const auto sums = reduction::sum(source, 0);
	const auto products = reduction::prod(source, 0);

	EXPECT_EQ(sums.shape(), Shape({1}));
	EXPECT_EQ(products.shape(), Shape({1}));
	EXPECT_EQ(sums[0], 0);
	EXPECT_EQ(products[0], 1);
}

TEST(EmptyAxisReduction, RankOneUndefinedReductionsThrow)
{
	const Vector<int> source;

	EXPECT_THROW(reduction::max(source, 0), Exceptions::IndexError);
	EXPECT_THROW(reduction::argmax(source, -1), Exceptions::IndexError);
	EXPECT_THROW(reduction::mean(source, 0, true), Exceptions::ZeroDivisionError);
}
