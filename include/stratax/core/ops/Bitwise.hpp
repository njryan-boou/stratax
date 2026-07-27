#pragma once

#include <stratax/core/Concepts.hpp>

// Unary

template<Array A>
requires Integer<typename A::value_type>
A operator~(const A& value);

// Array-array

template<Array A>
requires Integer<typename A::value_type>
A operator&(const A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A operator|(const A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A operator^(const A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A operator<<(const A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A operator>>(const A& lhs, const A& rhs);

// Array-scalar

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator&(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator|(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator^(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator<<(const A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A operator>>(const A& lhs, const Scalar& rhs);

// Scalar-array (reverse)

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator&(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator|(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator^(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator<<(const Scalar& lhs, const A& rhs);

template<Integer Scalar, Array A>
requires Integer<typename A::value_type>
A operator>>(const Scalar& lhs, const A& rhs);

// In-place array-array

template<Array A>
requires Integer<typename A::value_type>
A& operator&=(A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A& operator|=(A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A& operator^=(A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A& operator<<=(A& lhs, const A& rhs);

template<Array A>
requires Integer<typename A::value_type>
A& operator>>=(A& lhs, const A& rhs);

// In-place array-scalar

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator&=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator|=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator^=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator<<=(A& lhs, const Scalar& rhs);

template<Array A, Integer Scalar>
requires Integer<typename A::value_type>
A& operator>>=(A& lhs, const Scalar& rhs);
