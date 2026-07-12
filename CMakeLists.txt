cmake_minimum_required(VERSION 3.20)

project(Stratax
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CTest)

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating Stratax API documentation"
        VERBATIM
    )
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    bindings/module.cpp
    bindings/shape.cpp
    bindings/vector.cpp
    bindings/matrix.cpp
    bindings/tensor.cpp
)

target_include_directories(_core PRIVATE include)

add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_core>
        ${CMAKE_SOURCE_DIR}/python/stratax/$<TARGET_FILE_NAME:_core>
)

install(TARGETS _core
    LIBRARY DESTINATION stratax
    RUNTIME DESTINATION stratax
)

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

option(STRATAX_BUILD_TESTS "Build Stratax tests" ON)

if(BUILD_TESTING AND STRATAX_BUILD_TESTS)

    find_package(GTest CONFIG QUIET)

    if(NOT GTest_FOUND)
        include(FetchContent)

        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )

        FetchContent_MakeAvailable(googletest)
    endif()

    file(GLOB_RECURSE STRATAX_CPP_TEST_SOURCES CONFIGURE_DEPENDS
        tests/cpp/*.cpp
    )

    add_executable(stratax_cpp_tests
        ${STRATAX_CPP_TEST_SOURCES}
    )

    target_include_directories(stratax_cpp_tests PRIVATE include)

    target_link_libraries(stratax_cpp_tests
        PRIVATE GTest::gtest_main
    )

    include(GoogleTest)

    gtest_discover_tests(stratax_cpp_tests)

endif()