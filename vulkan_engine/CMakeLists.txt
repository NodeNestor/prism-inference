cmake_minimum_required(VERSION 3.16)
project(PrismVulkanEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# Benchmark executable
add_executable(benchmark
    benchmark.cpp
    prism_vulkan.cpp
    deps/volk.c
)

target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/deps
)

# volk needs VK_USE_PLATFORM_WIN32_KHR
target_compile_definitions(benchmark PRIVATE
    VK_USE_PLATFORM_WIN32_KHR
    VOLK_STATIC_DEFINES
)

# No need to link vulkan-1.lib — volk loads it dynamically
# But we do need the Windows API
target_link_libraries(benchmark PRIVATE)

if(MSVC)
    target_compile_options(benchmark PRIVATE /O2 /W3)
else()
    target_compile_options(benchmark PRIVATE -O2 -Wall)
endif()

# Static link MinGW runtime
if(MINGW)
    target_link_options(benchmark PRIVATE -static -static-libgcc -static-libstdc++)
endif()
