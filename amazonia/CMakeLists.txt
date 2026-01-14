add_library(amazonia-core INTERFACE)
add_library(amazonia::core ALIAS amazonia-core)
target_include_directories(amazonia-core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(amazonia-io-stb src/io/stb_image_implementation.cu)
target_compile_options(amazonia-io-stb PRIVATE "--diag-suppress=550")
add_library(amazonia-io INTERFACE)
add_library(amazonia::io ALIAS amazonia-io)
target_link_libraries(amazonia-io INTERFACE amazonia-io-stb amazonia-core)