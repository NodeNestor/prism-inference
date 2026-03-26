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

# Scaling benchmark (multi-config comparison)
add_executable(scaling_bench
    prism_scaling_bench.cpp
    deps/volk.c
)

target_include_directories(scaling_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/deps
)

target_compile_definitions(scaling_bench PRIVATE
    VK_USE_PLATFORM_WIN32_KHR
    VOLK_STATIC_DEFINES
)

target_link_libraries(scaling_bench PRIVATE)

if(MSVC)
    target_compile_options(scaling_bench PRIVATE /O2 /W3)
else()
    target_compile_options(scaling_bench PRIVATE -O2 -Wall)
endif()

if(MINGW)
    target_link_options(scaling_bench PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Linear attention benchmark
add_executable(linear_bench
    prism_linear_bench.cpp
    deps/volk.c
)
target_include_directories(linear_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/deps)
target_compile_definitions(linear_bench PRIVATE VK_USE_PLATFORM_WIN32_KHR VOLK_STATIC_DEFINES)
target_link_libraries(linear_bench PRIVATE)
if(MSVC)
    target_compile_options(linear_bench PRIVATE /O2 /W3)
else()
    target_compile_options(linear_bench PRIVATE -O2 -Wall)
endif()
if(MINGW)
    target_link_options(linear_bench PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# MoE benchmark (clean, isolated)
add_executable(moe_bench prism_moe_bench.cpp deps/volk.c)
target_include_directories(moe_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/deps)
target_compile_definitions(moe_bench PRIVATE VK_USE_PLATFORM_WIN32_KHR VOLK_STATIC_DEFINES)
target_link_libraries(moe_bench PRIVATE)
if(MSVC)
    target_compile_options(moe_bench PRIVATE /O2 /W3)
else()
    target_compile_options(moe_bench PRIVATE -O2 -Wall)
endif()
if(MINGW)
    target_link_options(moe_bench PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Overhead analysis benchmark
add_executable(overhead_bench
    prism_overhead_bench.cpp
    deps/volk.c
)
target_include_directories(overhead_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/deps)
target_compile_definitions(overhead_bench PRIVATE VK_USE_PLATFORM_WIN32_KHR VOLK_STATIC_DEFINES)
target_link_libraries(overhead_bench PRIVATE)
if(MSVC)
    target_compile_options(overhead_bench PRIVATE /O2 /W3)
else()
    target_compile_options(overhead_bench PRIVATE -O2 -Wall)
endif()
if(MINGW)
    target_link_options(overhead_bench PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Shader compilation helper
find_program(GLSLANG glslangValidator)
if(GLSLANG)
    set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shaders")
    set(NEW_SHADERS
        attention_windowed_h1024.comp.glsl
        attention_windowed_h2048.comp.glsl
    )
    foreach(shader ${NEW_SHADERS})
        string(REPLACE ".comp.glsl" ".spv" spv_name ${shader})
        add_custom_command(
            OUTPUT ${SHADER_DIR}/${spv_name}
            COMMAND ${GLSLANG} --target-env vulkan1.3 -o ${SHADER_DIR}/${spv_name} ${SHADER_DIR}/${shader}
            DEPENDS ${SHADER_DIR}/${shader}
            COMMENT "Compiling ${shader}"
        )
        list(APPEND SPV_OUTPUTS ${SHADER_DIR}/${spv_name})
    endforeach()
    add_custom_target(compile_shaders DEPENDS ${SPV_OUTPUTS})
    add_dependencies(scaling_bench compile_shaders)
endif()
