#include <gtest/gtest.h>
#include <stratax/core/Iterator.hpp>

TEST(CoreIterator, ReverseIteratorTraversesBackwards)
{
    int values[]{1, 2, 3};
    stratax::core::reverse_iterator_t<int*> first(values + 3);
    stratax::core::reverse_iterator_t<int*> last(values);
    EXPECT_EQ(last - first, 3);
    EXPECT_EQ(*first, 3);
    EXPECT_EQ(first[2], 1);
}
