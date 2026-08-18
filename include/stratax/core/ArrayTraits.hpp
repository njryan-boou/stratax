#pragma once

#include <type_traits>

#include <stratax/containers/Vector.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>

namespace stratax::core {

template<typename A, typename T>
struct RebindArray;

template<typename From, typename To>
struct RebindArray<container::Vector<From>, To>
{
	using type = container::Vector<To>;
};

template<typename From, typename To>
struct RebindArray<container::Matrix<From>, To>
{
	using type = container::Matrix<To>;
};

template<typename From, typename To>
struct RebindArray<container::Tensor<From>, To>
{
	using type = container::Tensor<To>;
};

template<typename A, typename T>
using rebind_array_t =
	typename RebindArray<std::remove_cvref_t<A>, T>::type;

} // namespace stratax::core