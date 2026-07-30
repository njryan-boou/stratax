#include <gtest/gtest.h>

#include <stratax.h>

#include <type_traits>

TEST(PublicApi, top_level_array_type_aliases_are_available)
{
    static_assert(std::is_same_v<stratax::Shape, stratax::core::Shape>);
    static_assert(std::is_same_v<stratax::Slice, stratax::core::Slice>);
    static_assert(std::is_same_v<stratax::Strides, stratax::core::Strides>);
    static_assert(std::is_same_v<stratax::Vector<double>, stratax::container::Vector<double>>);
    static_assert(std::is_same_v<stratax::Matrix<double>, stratax::container::Matrix<double>>);
    static_assert(std::is_same_v<stratax::Tensor<double>, stratax::container::Tensor<double>>);

    stratax::Vector<double> vector{1.0, 2.0, 3.0};
    EXPECT_EQ(vector.size(), 3);
}

TEST(PublicApi, top_level_common_algorithms_are_available)
{
    const auto tensor = stratax::zeros<double>(stratax::Shape{2, 3});
    const auto identity = stratax::identity<double>(3);
    const auto flat = stratax::flatten(identity);
    const auto reshaped = stratax::reshape(flat, stratax::Shape{3, 3});

    EXPECT_EQ(tensor.shape(), (stratax::Shape{2, 3}));
    EXPECT_EQ(identity(0, 0), 1.0);
    EXPECT_EQ(reshaped.shape(), (stratax::Shape{3, 3}));
    EXPECT_EQ(stratax::sum(identity), 3.0);
}

TEST(PublicApi, module_namespaces_are_available)
{
    const auto tensor = stratax::creation::full<int>(stratax::arrays::Shape{2, 2}, 4);
    const auto vector = stratax::transforms::flatten(tensor);
    const auto matrix = stratax::conversions::to_matrix(tensor);
    const auto first_row = stratax::slicing::slice(matrix, stratax::Slice{0, 1}, stratax::Slice{0, 2});

    EXPECT_EQ(stratax::reductions::sum(vector), 16);
    EXPECT_EQ(first_row.shape(), (stratax::Shape{1, 2}));
}

TEST(PublicApi, errors_namespace_alias_is_available)
{
    static_assert(std::is_same_v<stratax::ShapeError, stratax::errors::ShapeError>);
    static_assert(std::is_same_v<stratax::AxisError, stratax::errors::AxisError>);
}
