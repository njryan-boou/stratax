# =============================================================================
# Testing
# =============================================================================

option(
    STRATAX_BUILD_TESTS
    "Build Stratax tests"
    ON
)

if(BUILD_TESTING AND STRATAX_BUILD_TESTS)

    find_package(GTest CONFIG QUIET)

    if(NOT GTest_FOUND)

        include(FetchContent)

        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        FetchContent_Declare(

            googletest

            URL
                https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip

            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )

        FetchContent_MakeAvailable(googletest)

    endif()

    file(

        GLOB_RECURSE

        STRATAX_CPP_TEST_SOURCES

        CONFIGURE_DEPENDS

        tests/cpp/*.cpp
    )

    add_executable(

        stratax_cpp_tests

        ${STRATAX_CPP_TEST_SOURCES}
    )

    target_include_directories(

        stratax_cpp_tests

        PRIVATE
            include
    )

    target_link_libraries(

        stratax_cpp_tests

        PRIVATE
            GTest::gtest_main
    )

    include(GoogleTest)

    gtest_discover_tests(
        stratax_cpp_tests
    )

endif()