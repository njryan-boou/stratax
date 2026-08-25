#include <gtest/gtest.h>

#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Slice.hpp>
#include <stratax/indexing/Slicing.hpp>
#include <stratax/io/Print.hpp>

#include <cstddef>
#include <sstream>

using stratax::container::Matrix;
using stratax::core::Slice;
using stratax::indexing::slice;

TEST(ArrayViewPrint, PrintsStridedRankTwoViewInLogicalOrder)
{
    Matrix<int> source(3, 4);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        source[index] = static_cast<int>(index);
    }

    const auto view = slice(
        source,
        Slice{0, 3, 2},
        Slice{1, 4, 2});

    std::ostringstream stream;
    stream << view;

    EXPECT_EQ(stream.str(), "[\n    [1, 3]\n    [9, 11]\n]");
}
