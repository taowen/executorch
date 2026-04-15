# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT EXECUTORCH_BUILD_PYBIND)
  return()
endif()

pybind11_add_module(_exshader_runtime SHARED pybindings.cpp)

target_include_directories(
  _exshader_runtime PRIVATE ${_common_include_directories}
)

target_link_libraries(
  _exshader_runtime
  PRIVATE executorch extension_module_static extension_tensor extension_data_loader
)

if(TARGET optimized_native_cpu_ops_lib)
  target_link_libraries(_exshader_runtime PRIVATE optimized_native_cpu_ops_lib)
elseif(TARGET portable_ops_lib)
  target_link_libraries(_exshader_runtime PRIVATE portable_ops_lib)
endif()

if(TARGET vulkan_backend)
  target_link_libraries(_exshader_runtime PRIVATE vulkan_backend)
endif()

set_target_properties(
  _exshader_runtime
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             CXX_VISIBILITY_PRESET "hidden"
             INTERPROCEDURAL_OPTIMIZATION TRUE
             CXX_STANDARD 20
             LIBRARY_OUTPUT_DIRECTORY "${EXECUTORCH_ROOT}/exshader/runtime"
)

if(APPLE)
  set(RPATH "@loader_path")
else()
  set(RPATH "$ORIGIN")
endif()
set_target_properties(
  _exshader_runtime PROPERTIES BUILD_RPATH "${RPATH}" INSTALL_RPATH "${RPATH}"
)

install(TARGETS _exshader_runtime
        LIBRARY DESTINATION exshader/runtime
)
