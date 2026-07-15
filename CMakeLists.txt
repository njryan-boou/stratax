cmake_minimum_required(VERSION 3.20)

project(
    Stratax
    VERSION 0.1.0
    LANGUAGES CXX
)

include(CTest)

include(cmake/CompilerOptions.cmake)
include(cmake/Documentation.cmake)
include(cmake/PythonBindings.cmake)
include(cmake/Testing.cmake)