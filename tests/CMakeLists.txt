# Fetch GoogleTest (set GTEST_TAG to a release tag for a stable version)
FetchContent_Declare(
	gtest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(gtest)

function(amazonia_add_test name)
    add_executable(test_${name} "${name}.cu")
    target_link_libraries(test_${name} PRIVATE GTest::gtest_main amazonia::core)
    add_test(NAME test_${name} COMMAND test_${name})
endfunction()

amazonia_add_test(image2d_view)