#include <stratax.h>
#include <iostream>

using namespace stratax::core;

std::size_t get_higher_rank(const Shape& shape1, const Shape& shape2)
{
    return std::max({shape1.rank(), shape2.rank()});
}

Shape pad_shape(const Shape& shape, const std::size_t& target_rank)
{
    std::size_t rank_diff = target_rank - shape.rank();
    std::vector<std::size_t> result;

    for (std::size_t i = 0; i < shape.rank(); i++)
    {
        result.push_back(shape(i));
    }

    for (std::size_t i = 0; i < rank_diff; i++)
    {
        result.insert(result.begin(), 1);
    }

    return Shape{result};
}

bool validate_broadcast(const Shape& shape1, const Shape& shape2)
{
    std::size_t higher_rank = get_higher_rank(shape1, shape2);
    Shape pad_shape1 = pad_shape(shape1, higher_rank);
    Shape pad_shape2 = pad_shape(shape2, higher_rank);
    for (std::size_t i = 0; i < higher_rank; ++i)
    {
        std::size_t left = pad_shape1(i);
        std::size_t right = pad_shape2(i);

        if (left == right)
        {
            continue;
        }

        else if (right == 1)
        {
            continue;
        }

        else if (left == 1)
        {
            continue;
        }

        else
        {
            return false;
        }
    }

    return true;
}

Shape broadcasted_shape(const Shape& shape1, const Shape& shape2)
{
    std::size_t higher_rank = get_higher_rank(shape1, shape2);
    Shape pad_shape1 = pad_shape(shape1, higher_rank);
    Shape pad_shape2 = pad_shape(shape2, higher_rank);

    if (!validate_broadcast(shape1, shape2))
    {
        throw Exceptions::BroadcastError("Shapes are not broadcastable");
    }
    
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < higher_rank; ++i)
    {
        std::size_t left = pad_shape1(i);
        std::size_t right = pad_shape2(i);

        if (left == right)
        {
            result.push_back(left);
        }

        else
        {
            result.push_back(std::max({left, right}));
        }

    }

    Shape result_shape(result);
    return result_shape;
}
int main()
{
    stratax::Vector<int> a;
    Shape shape1{3, 1};
    Shape shape2{1, 4};
    auto x = broadcasted_shape(shape1, shape2);
    std::cout << x;
}