cmake_minimum_required(VERSION 3.10)
project(prism_benchmark)
set(CMAKE_CXX_STANDARD 17)

# Find ncnn
set(ncnn_DIR "${NCNN_DIR}")
find_package(ncnn REQUIRED)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark ncnn)
