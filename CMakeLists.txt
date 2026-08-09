cmake_minimum_required(VERSION 3.20)

project(
    Stratax
    VERSION 0.3.1
    LANGUAGES CXX
)

include(CTest)

include(cmake/CompilerOptions.cmake)
include(cmake/Documentation.cmake)

option(
    STRATAX_BUILD_PYTHON_BINDINGS
    "Build the Stratax Python extension"
    ON
)

if(STRATAX_BUILD_PYTHON_BINDINGS)
    include(cmake/PythonBindings.cmake)
endif()

include(cmake/Testing.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
