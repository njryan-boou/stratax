#pragma once

#include <type_traits>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

namespace stratax::core {

/**
 * @brief Rebinds a Stratax array container to a different value type.
 *
 * The container family is preserved while its stored value type is replaced.
 *
 * @tparam A Original array type.
 * @tparam T New value type.
 */
template<typename A, typename T>
struct RebindArray;

/**
 * @brief Rebinds a Vector to a different value type.
 */
template<typename From, typename To>
struct RebindArray<container::Vector<From>, To>
{
	using type = container::Vector<To>;
};

/**
 * @brief Rebinds a Matrix to a different value type.
 */
template<typename From, typename To>
struct RebindArray<container::Matrix<From>, To>
{
	using type = container::Matrix<To>;
};

/**
 * @brief Rebinds a Tensor to a different value type.
 */
template<typename From, typename To>
struct RebindArray<container::Tensor<From>, To>
{
	using type = container::Tensor<To>;
};

/**
 * @brief Convenience alias for rebinding an array to a new value type.
 *
 * cv- and reference qualifiers are removed from the array type before
 * rebinding.
 */
template<typename A, typename T>
using rebind_array_t =
	typename RebindArray<
		std::remove_cvref_t<A>,
		std::remove_cvref_t<T>
	>::type;

/**
 * @brief Determines the result container family for two Stratax arrays.
 *
 * Matching container families are preserved. Mixed container families
 * promote to Tensor.
 */
template<typename L, typename R, typename T>
struct PromoteArray;

/**
 * @brief Vector + Vector preserves Vector.
 */
template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Vector<LValue>,
	container::Vector<RValue>,
	T>
{
	using type = container::Vector<T>;
};

/**
 * @brief Matrix + Matrix preserves Matrix.
 */
template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Matrix<LValue>,
	container::Matrix<RValue>,
	T>
{
	using type = container::Matrix<T>;
};

/**
 * @brief Tensor + Tensor preserves Tensor.
 */
template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Tensor<LValue>,
	container::Tensor<RValue>,
	T>
{
	using type = container::Tensor<T>;
};

/**
 * @brief Mixed Vector/Matrix operations promote to Tensor.
 */
template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Vector<LValue>,
	container::Matrix<RValue>,
	T>
{
	using type = container::Tensor<T>;
};

template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Matrix<LValue>,
	container::Vector<RValue>,
	T>
{
	using type = container::Tensor<T>;
};

/**
 * @brief Mixed Vector/Tensor operations promote to Tensor.
 */
template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Vector<LValue>,
	container::Tensor<RValue>,
	T>
{
	using type = container::Tensor<T>;
};

template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Tensor<LValue>,
	container::Vector<RValue>,
	T>
{
	using type = container::Tensor<T>;
};

/**
 * @brief Mixed Matrix/Tensor operations promote to Tensor.
 */
template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Matrix<LValue>,
	container::Tensor<RValue>,
	T>
{
	using type = container::Tensor<T>;
};

template<typename LValue, typename RValue, typename T>
struct PromoteArray<
	container::Tensor<LValue>,
	container::Matrix<RValue>,
	T>
{
	using type = container::Tensor<T>;
};

/**
 * @brief Convenience alias for the promoted array container type.
 */
template<typename L, typename R, typename T>
using promote_array_t =
	typename PromoteArray<
		std::remove_cvref_t<L>,
		std::remove_cvref_t<R>,
		std::remove_cvref_t<T>
	>::type;

} // namespace stratax::core