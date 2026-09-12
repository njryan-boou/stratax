#include <stratax.h>
#include <stratax.h>
#include <gtest/gtest.h>

TEST(PublicApi, UmbrellaHeaderSupportsContainerAlgorithmsAndViews)
{
    stratax::container::Vector<double> values{1, 2, 3, 4};
    auto matrix = stratax::manipulation::reshape(values, stratax::core::Shape{2, 2});
    auto row_sums = reduction::sum(matrix, 1);
    EXPECT_EQ(row_sums[0], 3);
    EXPECT_EQ(row_sums[1], 7);
    auto view = stratax::indexing::slice(values, stratax::core::Slice{1, 4, 2});
    EXPECT_EQ(reduction::sum(view), 6);
    view[0] = 20;
    EXPECT_EQ(values[1], 20);
    EXPECT_EQ(matrix[1], 2);
}
