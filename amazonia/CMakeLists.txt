add_library(amazonia-core INTERFACE)
add_library(amazonia::core ALIAS amazonia-core)
target_include_directories(amazonia-core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)