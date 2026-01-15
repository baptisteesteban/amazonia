add_library(amazonia-core)
add_library(amazonia::core ALIAS amazonia-core)
target_sources(amazonia-core PRIVATE
    src/dt/generalised_geodesic.cu
)
target_include_directories(amazonia-core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(amazonia-io-stb OBJECT src/io/stb_image_implementation.cu)
target_compile_options(amazonia-io-stb PRIVATE "--diag-suppress=550")
add_library(amazonia-io)
add_library(amazonia::io ALIAS amazonia-io)
target_sources(amazonia-io PRIVATE
    src/io/imread.cu
    src/io/imwrite.cu
)
target_link_libraries(amazonia-io PRIVATE amazonia-io-stb)
target_link_libraries(amazonia-io PUBLIC amazonia-core)