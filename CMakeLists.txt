cmake_minimum_required(VERSION 3.24)

# Toolchain. The CUDA toolkit must match the torch build you run against
# (torch 2.7 cu128 -> CUDA 12.8). Point CMake at it with either
#   export CUDACXX=/path/to/cuda-12.8/bin/nvcc      or
#   cmake -DCMAKE_CUDA_COMPILER=... -DCUDAToolkit_ROOT=...
# A plain `pip install -e .` picks them up from the environment.
# GPU architecture: defaults to the GPU of the build machine (`native`), which
# needs a visible GPU at configure time. For a binary that runs on several GPUs
# pass an explicit list, e.g. -DCMAKE_CUDA_ARCHITECTURES="89;120". A wrong
# architecture fails at RUNTIME ("no kernel image is available"), not at build time.
# Set the default before project() initializes it from the compiler.
if((NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
   AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(ropesim LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(pybind11   CONFIG REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenMP  REQUIRED)  # lbvh.cu uses omp pragmas in host code
find_package(Threads REQUIRED)
# glm is header-only; apt: libglm-dev, conda: glm. Override with -DGLM_INCLUDE_DIR=...
find_path(GLM_INCLUDE_DIR glm/glm.hpp
          HINTS "$ENV{GLM_INCLUDE_DIR}" "$ENV{CONDA_PREFIX}/include" "${CUDAToolkit_ROOT}/include"
          REQUIRED)

# ---------------------------------------------------------------------------
# Vendored YarnBall solver (third_party/yarnball, GPL-3.0 — see VENDOR.md).
# Headless subset: 6 solver TUs + the LBVH TU, compiled with YARNBALL_HEADLESS
# so no GL/GLFW/imgui/assimp/freetype headers or libs are needed at all.
# ---------------------------------------------------------------------------
set(YARNBALL_ROOT "${CMAKE_SOURCE_DIR}/third_party/yarnball")

add_library(yarnball STATIC
    ${YARNBALL_ROOT}/KittenEngine/YarnBall/YarnBall.cu
    ${YARNBALL_ROOT}/KittenEngine/YarnBall/sim/step.cpp
    ${YARNBALL_ROOT}/KittenEngine/YarnBall/sim/iteration.cu
    ${YARNBALL_ROOT}/KittenEngine/YarnBall/sim/cosserat.cu
    ${YARNBALL_ROOT}/KittenEngine/YarnBall/sim/collision.cu
    ${YARNBALL_ROOT}/KittenEngine/YarnBall/sim/statistics.cpp
    ${YARNBALL_ROOT}/KittenEngine/KittenEngine/KittenGpuLBVH/lbvh.cu)

set_target_properties(yarnball PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(yarnball PUBLIC YARNBALL_HEADLESS)
target_include_directories(yarnball PUBLIC
    ${YARNBALL_ROOT}/KittenEngine
    ${YARNBALL_ROOT}/KittenEngine/KittenEngine
    ${YARNBALL_ROOT}/KittenEngine/KittenEngine/includes
    ${GLM_INCLUDE_DIR})
target_link_libraries(yarnball PUBLIC
    CUDA::cudart OpenMP::OpenMP_CXX Threads::Threads)
# OpenMP must reach both compile and link of CUDA TUs: lbvh.cu uses
# `#pragma omp` inside CUDA-compiled host code.
target_compile_options(yarnball PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

# ---------------------------------------------------------------------------
# Python module: bindings.cpp (pybind11 + CUDA-array-interface parsing) plus the
# ropesim CUDA kernels. Nothing here includes torch.
# ---------------------------------------------------------------------------
pybind11_add_module(_ropesim csrc/bindings.cpp csrc/batch.cu csrc/sdf.cu)
target_link_libraries(_ropesim PRIVATE yarnball)

install(TARGETS _ropesim DESTINATION ropesim)
