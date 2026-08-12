#include <gtest/gtest.h>
#include <stratax/core/ArrayBase.hpp>
#include <stratax.h>

using namespace stratax::container;
using namespace stratax::core;

TEST(ArrayBase, methods)
{
    Tensor<int> tensor({2, 3, 4});

    // Metadata
    {
        EXPECT_EQ(tensor.size(), 2 * 3 * 4);
        EXPECT_FALSE(tensor.empty());
        EXPECT_EQ(tensor.rank(), 3);

        EXPECT_EQ(
            tensor.shape(),
            (stratax::Shape{2, 3, 4})
        );

        EXPECT_EQ(
            tensor.strides(),
            stratax::Strides(tensor.shape())
        );
    }

    // Data access
    {
        EXPECT_NE(tensor.data(), nullptr);

        tensor.data()[1] = 42;

        EXPECT_EQ(tensor.data()[1], 42);
    }

    // Front and back
    {
        tensor.data()[0] = 10;
        tensor.data()[tensor.size() - 1] = 20;

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

    // Empty array
    {
        Tensor<int> empty(stratax::Shape{});

        EXPECT_EQ(empty.size(), 0);
        EXPECT_TRUE(empty.empty());

        EXPECT_THROW(empty.front(), Exceptions::IndexError);
        EXPECT_THROW(empty.back(), Exceptions::IndexError);
    }
}

TEST(ArrayBase, operators)
{
    Tensor<int> tensor({2, 3, 4});

    // Mutable access
    {
        tensor[0] = 5;
        tensor[1] = 10;
        tensor[tensor.size() - 1] = 20;

        EXPECT_EQ(tensor[0], 5);
        EXPECT_EQ(tensor[1], 10);
        EXPECT_EQ(tensor[tensor.size() - 1], 20);
    }

    // Mutable call operator access
    {
        tensor[0] = 11;
        tensor[1] = 12;
        tensor[tensor.size() - 1] = 21;

        EXPECT_EQ(tensor[0], 11);
        EXPECT_EQ(tensor[1], 12);
        EXPECT_EQ(tensor[tensor.size() - 1], 21);
    }

    // Const access
    {
        const Tensor<int>& const_tensor = tensor;

        EXPECT_EQ(const_tensor[0], 11);
        EXPECT_EQ(const_tensor[1], 12);
        EXPECT_EQ(const_tensor[const_tensor.size() - 1], 21);
    }

    // Const call operator access
    {
        const Tensor<int>& const_tensor = tensor;

        EXPECT_EQ(const_tensor[0], 11);
        EXPECT_EQ(const_tensor[1], 12);
        EXPECT_EQ(const_tensor[const_tensor.size() - 1], 21);
    }
}

TEST(ArrayBase, iterators)
{
    Tensor<int> tensor({2, 3, 4});

    // Fill the tensor with values
    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        tensor[i] = static_cast<int>(i);
    }

    // Test mutable iterators
    {
        std::size_t index = 0;
        for (auto it = tensor.begin(); it != tensor.end(); ++it)
        {
            EXPECT_EQ(*it, static_cast<int>(index));
            ++index;
        }
        EXPECT_EQ(index, tensor.size());
    }

    // Test const iterators
    {
        const Tensor<int>& const_tensor = tensor;
        std::size_t index = 0;
        for (auto it = const_tensor.begin(); it != const_tensor.end(); ++it)
        {
            EXPECT_EQ(*it, static_cast<int>(index));
            ++index;
        }
        EXPECT_EQ(index, const_tensor.size());
    }

    // Test reverse iterators
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

TEST(ArrayBase, modifiers)
{
    Tensor<int> tensor({2, 3, 4});
    Tensor<int> other({2, 3, 4});

    // Fill the tensor with values
    for (std::size_t i = 0; i < tensor.size(); ++i)
    {
        tensor[i] = static_cast<int>(i);
    }

    // Fill the other tensor with a different value
    other.fill(42);

    // Test fill
    {
        tensor.fill(7);
        for (std::size_t i = 0; i < tensor.size(); ++i)
        {
            EXPECT_EQ(tensor[i], 7);
        }
    }

    // Test swap
    {
        tensor.swap(other);

        for (std::size_t i = 0; i < tensor.size(); ++i)
        {
            EXPECT_EQ(tensor[i], 42);
            EXPECT_EQ(other[i], 7);
        }
    }
}


