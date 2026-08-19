#include <concepts>

#include <stratax.h>

static_assert(std::same_as<
	stratax::core::rebind_array_t<
		stratax::container::Vector<int>,
		double>,
	stratax::container::Vector<double>>);

static_assert(std::same_as<
	stratax::core::rebind_array_t<
		const stratax::container::Matrix<int>&,
		double>,
	stratax::container::Matrix<double>>);

static_assert(std::same_as<
	stratax::core::rebind_array_t<
		volatile stratax::container::Tensor<int>&&,
		const double&>,
	stratax::container::Tensor<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Vector<int>,
		stratax::container::Vector<float>,
		double>,
	stratax::container::Vector<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Matrix<int>,
		stratax::container::Matrix<float>,
		double>,
	stratax::container::Matrix<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Tensor<int>,
		stratax::container::Tensor<float>,
		double>,
	stratax::container::Tensor<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Vector<int>,
		stratax::container::Matrix<float>,
		double>,
	stratax::container::Tensor<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Matrix<int>,
		stratax::container::Vector<float>,
		double>,
	stratax::container::Tensor<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Vector<int>,
		stratax::container::Tensor<float>,
		double>,
	stratax::container::Tensor<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Tensor<int>,
		stratax::container::Vector<float>,
		double>,
	stratax::container::Tensor<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		stratax::container::Matrix<int>,
		stratax::container::Tensor<float>,
		double>,
	stratax::container::Tensor<double>>);

static_assert(std::same_as<
	stratax::core::promote_array_t<
		const stratax::container::Tensor<int>&,
		volatile stratax::container::Matrix<float>&&,
		const double&>,
	stratax::container::Tensor<double>>);
