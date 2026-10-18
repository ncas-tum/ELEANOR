cmake_minimum_required(VERSION 3.28.3)
project(eleanor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    set(_eleanor_cuda_default ON)
else()
    set(_eleanor_cuda_default OFF)
endif()
option(ELEANOR_WITH_CUDA "Build the CUDA kernels" ${_eleanor_cuda_default})


if(ELEANOR_WITH_CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "ELEANOR_WITH_CUDA=ON but no CUDA compiler was found")
    endif()

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    enable_language(CUDA)

    # cu126 / cu130 from the nvcc version (major.minor only, e.g. 12.6.3 -> cu126)
    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${CMAKE_CUDA_COMPILER_VERSION}")
    set(CUDA_SUFFIX "cu${CMAKE_MATCH_1}${CMAKE_MATCH_2}")

    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER} (${CMAKE_CUDA_COMPILER_VERSION})")
    message(STATUS "CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
    message(STATUS "CUDA enabled. Suffix: ${CUDA_SUFFIX}")
else()
    set(CUDA_SUFFIX "cpu")
    message(STATUS "Building for CPU. Suffix: cpu")
endif()

find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(FATAL_ERROR
        "PyTorch was not found, but ELEANOR's extension is built against it.\n"
        "Pick a hardware variant when syncing: uv sync --extra cpu "
        "(or --extra cuda12 / cuda13 / mps).")
endif()
# Development.Module only: an extension module must not link libpython
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Apple Clang has no OpenMP of its own. PyTorch's macOS wheels ship omp.h and
# libomp.dylib, so use those: linking a second runtime (e.g. Homebrew's) next
# to torch's makes libomp abort with "OMP: Error #15" at import time.
if(APPLE AND NOT OpenMP_CXX_FLAGS AND EXISTS "${TORCH_INSTALL_PREFIX}/lib/libomp.dylib")
    set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${TORCH_INSTALL_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${TORCH_INSTALL_PREFIX}/lib/libomp.dylib")
endif()
find_package(OpenMP REQUIRED)

set(MODULE_NAME "_C_${CUDA_SUFFIX}")

# -----------------------------------------------------------------------------
# 3. Source File Gathering
# -----------------------------------------------------------------------------
set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eleanor/torch/models/csrc")

# Gather all .cpp files
file(GLOB_RECURSE CPP_SOURCES "${CSRC_DIR}/*.cpp")

# Gather .cu files only if CUDA is enabled
set(CUDA_SOURCES "")
if(NOT CUDA_SUFFIX STREQUAL "cpu")
    file(GLOB_RECURSE CUDA_SOURCES "${CSRC_DIR}/*.cu")
endif()

set(ALL_SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})

# -----------------------------------------------------------------------------
# 4. Target Definition
# -----------------------------------------------------------------------------
# We create a MODULE library (this is what PyTorch extensions are)
add_library(${MODULE_NAME} MODULE ${ALL_SOURCES})

# Add Python and Torch include directories
target_include_directories(${MODULE_NAME} PRIVATE 
    ${Python_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
# Ensure the output is a shared library with the correct name
set_target_properties(${MODULE_NAME} PROPERTIES 
    PREFIX "" 
    OUTPUT_NAME ${MODULE_NAME}
)

if(ELEANOR_WITH_CUDA)
    # Architectures come from TORCH_CUDA_ARCH_LIST via find_package(Torch);
    # turn off CMake's own arch flags so there is a single source of truth
    set_target_properties(${MODULE_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
endif()

# Handle Debug vs Release
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CXX_OPT "-Og")
    # set(NVCC_OPT "-Og")
    set(LINK_OPT "-Og")
else()
    set(CXX_OPT "-O3")
    set(NVCC_OPT "-O3")
    set(LINK_OPT "")
endif()

# C++ Flags
target_compile_options(${MODULE_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPT} -fdiagnostics-color=always>
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_OPT} -U Py_LIMITED_API>
)

# Use only the stable Python API (3.11+), so a single cp311-abi3 wheel works on
# every newer CPython. models.cpp is the only file that includes Python.h.
target_compile_definitions(${MODULE_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:Py_LIMITED_API=0x030B0000>
)

# Define the macro used by models.cpp for the PyInit function
# Note: No quotes here, because the C++ code uses it for token concatenation
target_compile_definitions(${MODULE_NAME} PRIVATE TORCH_EXTENSION_NAME=${MODULE_NAME})

target_link_options(${MODULE_NAME} PRIVATE ${LINK_OPT})
target_link_libraries(${MODULE_NAME} PRIVATE
    Python::Module
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
)

install(TARGETS ${MODULE_NAME} DESTINATION eleanor/torch/models)