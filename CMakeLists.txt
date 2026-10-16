cmake_minimum_required(VERSION 3.18)
project(eleanor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(CUDA_SUFFIX "cpu")

# Priority 1: Environment variable CUDA_VERSION
if(DEFINED ENV{CUDA_VERSION})
    set(ENV_CUDA_VER $ENV{CUDA_VERSION})
    if(NOT ENV_CUDA_VER STREQUAL "cpu")
        string(REPLACE "." "" CLEAN_VER ${ENV_CUDA_VER})
        set(CUDA_SUFFIX "cu${CLEAN_VER}")
    endif()
# Priority 2: Check if CUDA is available via Torch (simplified check)
# Note: We check for the existence of the CUDA compiler if not explicitly 'cpu'
else()
    # We attempt to find nvcc only if we aren't forced to CPU
    find_program(NVCC_PATH nvcc)
    if(NVCC_PATH)
        # In a real scenario, we'd extract the version from nvcc --version
        # For now, we'll use a placeholder or let the user set CUDA_VERSION
        # To mirror setup.py, we can use the environment or a default
        set(CUDA_SUFFIX "cu126") # Or implement a version check here
    endif()
endif()

# IMPORTANT: Only enable the CUDA language if we aren't building for CPU
if(NOT CUDA_SUFFIX STREQUAL "cpu")
    enable_language(CUDA)
    message(STATUS "CUDA enabled. Suffix: ${CUDA_SUFFIX}")
else()
    message(STATUS "Building for CPU. Suffix: cpu")
endif()

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

# -----------------------------------------------------------------------------
# 5. Compiler Flags & Options
# -----------------------------------------------------------------------------
# Handle Debug vs Release
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CXX_OPT "-Og")
    set(NVCC_OPT "-Og")
    set(LINK_OPT "-Og")
else()
    set(CXX_OPT "-O3")
    set(NVCC_OPT "-O3")
    set(LINK_OPT "")
endif()

# C++ Flags
target_compile_options(${MODULE_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPT} -fopenmp -fdiagnostics-color=always>
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_OPT} -U Py_LIMITED_API>
)

# Define the macro used by models.cpp for the PyInit function
# Note: No quotes here, because the C++ code uses it for token concatenation
target_compile_definitions(${MODULE_NAME} PRIVATE TORCH_EXTENSION_NAME=${MODULE_NAME})

# CUDA Architecture (Optional but recommended: adjust to your target GPUs)
set_target_properties(${MODULE_NAME} PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")

# Linker Flags
target_link_options(${MODULE_NAME} PRIVATE ${LINK_OPT})

# Link PyTorch and OpenMP
target_link_libraries(${MODULE_NAME} PRIVATE ${TORCH_LIBRARIES})
find_package(OpenMP REQUIRED)
target_link_libraries(${MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)

# -----------------------------------------------------------------------------
# 6. Installation (for scikit-build-core)
# -----------------------------------------------------------------------------
# This tells scikit-build-core where to put the resulting .so file 
# so that it ends up inside the python package.
install(TARGETS ${MODULE_NAME} DESTINATION eleanor/torch/models)