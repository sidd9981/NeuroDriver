cmake_minimum_required(VERSION 3.18)
project(neurodriver_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find LibTorch ---
# On Apple Silicon (M3 Pro):
#   cmake -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") ..
#
# Or via Homebrew:
#   brew install pytorch
#   cmake ..

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --- Main inference binary ---
add_executable(neurodriver_inference src/main.cpp)
target_link_libraries(neurodriver_inference "${TORCH_LIBRARIES}")
target_include_directories(neurodriver_inference PRIVATE ${TORCH_INCLUDE_DIRS})

# --- Copy model files to build dir for convenience ---
if(EXISTS "${CMAKE_SOURCE_DIR}/models/driving_model.pt")
    file(COPY "${CMAKE_SOURCE_DIR}/models/driving_model.pt"
         DESTINATION "${CMAKE_BINARY_DIR}/models")
    message(STATUS "Copied driving_model.pt to build/models/")
endif()

# --- macOS: embed the actual libtorch lib path into the binary's rpath ---
# This eliminates the need for DYLD_LIBRARY_PATH at runtime.
if(APPLE)
    # Extract the directory containing libc10.dylib, libtorch.dylib, etc.
    # TORCH_LIBRARIES contains targets like "torch;torch_cpu;c10" —
    # we resolve one of them to find the lib directory.
    foreach(_lib ${TORCH_LIBRARIES})
        if(TARGET ${_lib})
            get_target_property(_loc ${_lib} IMPORTED_LOCATION)
            if(_loc)
                get_filename_component(_torch_lib_dir "${_loc}" DIRECTORY)
                break()
            endif()
        endif()
    endforeach()

    if(_torch_lib_dir)
        message(STATUS "LibTorch lib dir: ${_torch_lib_dir}")
        set_target_properties(neurodriver_inference PROPERTIES
            BUILD_RPATH "${_torch_lib_dir}"
            INSTALL_RPATH "${_torch_lib_dir}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        # Fallback: derive from Torch_DIR
        get_filename_component(_torch_lib_dir "${Torch_DIR}/../../../lib" ABSOLUTE)
        message(STATUS "LibTorch lib dir (fallback): ${_torch_lib_dir}")
        set_target_properties(neurodriver_inference PROPERTIES
            BUILD_RPATH "${_torch_lib_dir}"
            INSTALL_RPATH "${_torch_lib_dir}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endif()

message(STATUS "LibTorch version: ${Torch_VERSION}")