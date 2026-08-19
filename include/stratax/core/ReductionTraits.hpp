#pragma once

#include <type_traits>

/**
 * @brief Selects accumulator and result types for numeric reductions.
 *
 * The primary template preserves the input dtype. Specializations may select
 * wider or otherwise more appropriate types for individual reductions.
 *
 * @tparam T Unqualified array value type.
 */
template<typename T>
struct ReductionTraits
{
	using sum_type = T;
	using prod_type = T;
};

/**
 * @brief Accumulator and result type used by `reduction::sum`.
 *
 * cv- and reference qualifiers are removed before consulting the trait.
 * @tparam T Array value type, possibly cv- or reference-qualified.
 */
template<typename T>
using reduction_sum_t =
	typename ReductionTraits<std::remove_cvref_t<T>>::sum_type;

/**
 * @brief Accumulator and result type used by `reduction::prod`.
 *
 * cv- and reference qualifiers are removed before consulting the trait.
 * @tparam T Array value type, possibly cv- or reference-qualified.
 */
template<typename T>
using reduction_prod_t =
	typename ReductionTraits<std::remove_cvref_t<T>>::prod_type;
