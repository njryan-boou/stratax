
#pragma once

#include <iterator>
#include <sstream>
#include <ostream>

#include <stratax/containers/Vector.hpp>
#include <stratax/containers/Matrix.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Exceptions.hpp>

namespace stratax::container {

template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vector)
{
    for (auto it = vector.begin(); it != vector.end(); ++it)
    {
        os << *it;

        if (std::next(it) != vector.end())
        {
            os << " ";
        }
    }

    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
{
    for (std::size_t i = 0; i < matrix.rows(); ++i)
    {
        for (std::size_t j = 0; j < matrix.cols(); ++j)
        {
            os << matrix(i, j);

            if (j + 1 != matrix.cols())
                os << ' ';
        }

        if (i + 1 != matrix.rows())
            os << '\n';
    }

    return os;
}

} 
