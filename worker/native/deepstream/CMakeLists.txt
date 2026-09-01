cmake_minimum_required(VERSION 3.22)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "120-real;120-virtual" CACHE STRING
      "CUDA architectures for the deployment GPU")
endif()
project(seeon_deepstream_preflight VERSION 1.0.0 LANGUAGES CXX CUDA)

find_package(PkgConfig REQUIRED)
pkg_check_modules(
  GSTREAMER REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-base-1.0
  gstreamer-video-1.0
)

set(SEEON_WARNINGS -Wall -Wextra -Wpedantic -Werror)
option(SEEON_ENABLE_SANITIZERS "Run pure native tests with ASan and UBSan" OFF)
# Bit-stable float behavior for the parity-pinned perception math.
set(SEEON_FP_FLAGS -ffp-contract=off)

# TensorRT + CUDA runtime (present in the pinned DeepStream image).
find_library(SEEON_NVINFER nvinfer REQUIRED)
find_library(
  SEEON_CUDART cudart
  PATHS /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib
  REQUIRED
)
find_path(
  SEEON_TRT_INCLUDE NvInfer.h
  PATHS /usr/include/x86_64-linux-gnu /usr/include
  REQUIRED
)
find_path(
  SEEON_CUDA_INCLUDE cuda_runtime_api.h
  PATHS /usr/local/cuda/include /usr/local/cuda/targets/x86_64-linux/include
  REQUIRED
)
find_path(
  SEEON_NVBUFSURFACE_INCLUDE nvbufsurface.h
  PATHS /opt/nvidia/deepstream/deepstream/sources/includes
  REQUIRED
)

add_executable(
  seeon-deepstream-nvmm-cuda-interop-probe
  src/nvmm_cuda_interop_probe.cu
  src/native_perception.cpp
  src/preprocess_gpu.cu
)
set_target_properties(
  seeon-deepstream-nvmm-cuda-interop-probe
  PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
)
target_compile_features(seeon-deepstream-nvmm-cuda-interop-probe PRIVATE cxx_std_20)
target_compile_options(
  seeon-deepstream-nvmm-cuda-interop-probe
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SEEON_WARNINGS};${SEEON_FP_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)
target_include_directories(
  seeon-deepstream-nvmm-cuda-interop-probe
  PRIVATE ${SEEON_CUDA_INCLUDE} ${SEEON_NVBUFSURFACE_INCLUDE}
)
target_link_libraries(
  seeon-deepstream-nvmm-cuda-interop-probe PRIVATE PkgConfig::GSTREAMER ${SEEON_CUDART}
)

add_executable(
  seeon-deepstream-preflight
  src/main.cpp
  src/native_perception.cpp
  src/postprocess_gpu.cu
  src/preprocess_cpu.cpp
  src/preprocess_gpu.cu
  src/trt_perception.cpp
)
target_compile_features(seeon-deepstream-preflight PRIVATE cxx_std_20)
target_compile_options(
  seeon-deepstream-preflight
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SEEON_WARNINGS};${SEEON_FP_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)
target_include_directories(
  seeon-deepstream-preflight PRIVATE ${SEEON_TRT_INCLUDE} ${SEEON_CUDA_INCLUDE}
)
target_link_libraries(
  seeon-deepstream-preflight PRIVATE PkgConfig::GSTREAMER ${SEEON_NVINFER} ${SEEON_CUDART}
)

add_executable(
  seeon-deepstream-child
  src/au_parser.cpp
  src/au_transport.cpp
  src/child.cpp
  src/child_commands.cpp
  src/child_command_support.cpp
  src/child_failures.cpp
  src/child_server.cpp
  src/encoded_source_bus.cpp
  src/encoded_source_branch.cpp
  src/encoded_source_sample.cpp
  src/native_perception.cpp
  src/postprocess_gpu.cu
  src/preprocess_cpu.cpp
  src/preprocess_gpu.cu
  src/perception_wire.cpp
  src/preview_transport.cpp
  src/source_bus.cpp
  src/source_runtime.cpp
  src/source_uri.cpp
  src/trt_perception.cpp
)
target_compile_features(seeon-deepstream-child PRIVATE cxx_std_20)
target_compile_definitions(seeon-deepstream-child PRIVATE SEEON_HAS_GSTREAMER=1)
target_compile_options(
  seeon-deepstream-child
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SEEON_WARNINGS};${SEEON_FP_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)
target_include_directories(
  seeon-deepstream-child
  PRIVATE ${SEEON_TRT_INCLUDE} ${SEEON_CUDA_INCLUDE} ${SEEON_NVBUFSURFACE_INCLUDE}
)
target_link_libraries(
  seeon-deepstream-child PRIVATE PkgConfig::GSTREAMER ${SEEON_NVINFER} ${SEEON_CUDART}
)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_executable(
    seeon-deepstream-ipc-test
    src/ipc_protocol_test.cpp
    src/perception_wire.cpp
  )
  target_compile_features(seeon-deepstream-ipc-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-ipc-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-ipc-test COMMAND seeon-deepstream-ipc-test)
  add_test(
    NAME seeon-deepstream-wire-cross-language-test
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/perception_wire_cross_language_test.py
      $<TARGET_FILE:seeon-deepstream-ipc-test>
  )
  add_executable(
    seeon-deepstream-source-runtime-test
    src/source_runtime_test.cpp
    src/au_transport.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
  )
  target_compile_features(seeon-deepstream-source-runtime-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-source-runtime-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-source-runtime-test COMMAND seeon-deepstream-source-runtime-test)
  add_executable(
    seeon-deepstream-source-runtime-gstreamer-test
    src/source_runtime_gstreamer_test.cpp
    src/au_parser.cpp
    src/au_transport.cpp
    src/encoded_source_bus.cpp
    src/encoded_source_branch.cpp
    src/encoded_source_sample.cpp
    src/source_bus.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
  )
  add_executable(
    seeon-deepstream-native-perception-driver
    src/native_perception_driver.cpp
    src/native_perception.cpp
    src/preprocess_cpu.cpp
  )
  target_compile_features(seeon-deepstream-native-perception-driver PRIVATE cxx_std_20)
  target_compile_options(
    seeon-deepstream-native-perception-driver PRIVATE ${SEEON_WARNINGS} ${SEEON_FP_FLAGS}
  )
  add_executable(
    seeon-deepstream-native-perception-float-test
    src/native_perception_float_test.cpp
    src/native_perception.cpp
  )
  target_compile_features(seeon-deepstream-native-perception-float-test PRIVATE cxx_std_20)
  target_compile_options(
    seeon-deepstream-native-perception-float-test PRIVATE ${SEEON_WARNINGS} ${SEEON_FP_FLAGS}
  )
  add_test(
    NAME seeon-deepstream-native-perception-float-test
    COMMAND seeon-deepstream-native-perception-float-test
  )
  target_compile_features(seeon-deepstream-source-runtime-gstreamer-test PRIVATE cxx_std_20)
  target_compile_definitions(
    seeon-deepstream-source-runtime-gstreamer-test PRIVATE SEEON_HAS_GSTREAMER=1
  )
  target_compile_options(
    seeon-deepstream-source-runtime-gstreamer-test PRIVATE ${SEEON_WARNINGS}
  )
  target_include_directories(
    seeon-deepstream-source-runtime-gstreamer-test
    PRIVATE ${SEEON_CUDA_INCLUDE} ${SEEON_NVBUFSURFACE_INCLUDE}
  )
  target_link_libraries(
    seeon-deepstream-source-runtime-gstreamer-test
    PRIVATE PkgConfig::GSTREAMER ${SEEON_CUDART}
  )
  add_test(
    NAME seeon-deepstream-source-runtime-gstreamer-test
    COMMAND seeon-deepstream-source-runtime-gstreamer-test
  )
  add_executable(
    seeon-deepstream-child-commands-test
    src/child_commands_test.cpp
    src/au_transport.cpp
    src/child_commands.cpp
    src/child_command_support.cpp
    src/child_failures.cpp
    src/native_perception.cpp
    src/perception_wire.cpp
    src/preview_transport.cpp
    src/source_runtime.cpp
    src/source_uri.cpp
    src/trt_perception_stub.cpp
  )
  target_compile_features(seeon-deepstream-child-commands-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-child-commands-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-child-commands-test COMMAND seeon-deepstream-child-commands-test)
  add_executable(seeon-deepstream-workspace-pool-test src/workspace_pool_test.cpp)
  target_compile_features(seeon-deepstream-workspace-pool-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-workspace-pool-test PRIVATE ${SEEON_WARNINGS})
  find_package(Threads REQUIRED)
  target_link_libraries(seeon-deepstream-workspace-pool-test PRIVATE Threads::Threads)
  add_test(NAME seeon-deepstream-workspace-pool-test
           COMMAND seeon-deepstream-workspace-pool-test)
  add_executable(seeon-deepstream-au-parser-test src/au_parser_test.cpp src/au_parser.cpp)
  target_compile_features(seeon-deepstream-au-parser-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-au-parser-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-au-parser-test COMMAND seeon-deepstream-au-parser-test)
  add_executable(seeon-deepstream-au-transport-test src/au_transport_test.cpp src/au_transport.cpp)
  target_compile_features(seeon-deepstream-au-transport-test PRIVATE cxx_std_20)
  target_compile_options(seeon-deepstream-au-transport-test PRIVATE ${SEEON_WARNINGS})
  add_test(NAME seeon-deepstream-au-transport-test COMMAND seeon-deepstream-au-transport-test)
  add_test(
    NAME seeon-deepstream-nvmm-cuda-interop-probe
    COMMAND seeon-deepstream-nvmm-cuda-interop-probe
  )
  set_tests_properties(seeon-deepstream-nvmm-cuda-interop-probe PROPERTIES LABELS gpu)
  add_executable(
    seeon-deepstream-preprocess-gpu-test
    src/preprocess_gpu_test.cu
    src/native_perception.cpp
    src/preprocess_cpu.cpp
    src/preprocess_gpu.cu
  )
  set_target_properties(
    seeon-deepstream-preprocess-gpu-test
    PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  )
  target_compile_features(seeon-deepstream-preprocess-gpu-test PRIVATE cxx_std_20)
  target_compile_options(
    seeon-deepstream-preprocess-gpu-test
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${SEEON_WARNINGS};${SEEON_FP_FLAGS}>
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
  )
  target_include_directories(
    seeon-deepstream-preprocess-gpu-test PRIVATE ${SEEON_CUDA_INCLUDE}
  )
  target_link_libraries(seeon-deepstream-preprocess-gpu-test PRIVATE ${SEEON_CUDART})
  add_test(
    NAME seeon-deepstream-preprocess-gpu-test
    COMMAND seeon-deepstream-preprocess-gpu-test
  )
  set_tests_properties(seeon-deepstream-preprocess-gpu-test PROPERTIES LABELS gpu)
  add_executable(
    seeon-deepstream-postprocess-gpu-test
    src/postprocess_gpu_test.cu
    src/postprocess_gpu.cu
  )
  set_target_properties(
    seeon-deepstream-postprocess-gpu-test
    PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
  )
  target_compile_features(seeon-deepstream-postprocess-gpu-test PRIVATE cxx_std_20)
  target_compile_options(
    seeon-deepstream-postprocess-gpu-test
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${SEEON_WARNINGS};${SEEON_FP_FLAGS}>
      $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
  )
  target_include_directories(
    seeon-deepstream-postprocess-gpu-test PRIVATE ${SEEON_CUDA_INCLUDE}
  )
  target_link_libraries(seeon-deepstream-postprocess-gpu-test PRIVATE ${SEEON_CUDART})
  add_test(
    NAME seeon-deepstream-postprocess-gpu-test
    COMMAND seeon-deepstream-postprocess-gpu-test
  )
  set_tests_properties(seeon-deepstream-postprocess-gpu-test PROPERTIES LABELS gpu)

  set(
    SEEON_NATIVE_TEST_TARGETS
    seeon-deepstream-ipc-test
    seeon-deepstream-source-runtime-test
    seeon-deepstream-source-runtime-gstreamer-test
    seeon-deepstream-native-perception-driver
    seeon-deepstream-native-perception-float-test
    seeon-deepstream-child-commands-test
    seeon-deepstream-au-parser-test
    seeon-deepstream-au-transport-test
    seeon-deepstream-workspace-pool-test
  )
  if(SEEON_ENABLE_SANITIZERS)
    foreach(target IN LISTS SEEON_NATIVE_TEST_TARGETS)
      target_compile_options(
        ${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer
      )
      target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endforeach()
    set_tests_properties(
      seeon-deepstream-ipc-test
      seeon-deepstream-wire-cross-language-test
      seeon-deepstream-source-runtime-test
      seeon-deepstream-source-runtime-gstreamer-test
      seeon-deepstream-native-perception-float-test
      seeon-deepstream-child-commands-test
      seeon-deepstream-au-parser-test
      seeon-deepstream-au-transport-test
      seeon-deepstream-workspace-pool-test
      PROPERTIES
        ENVIRONMENT
          "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
    )
  endif()

  configure_file(
    runtime-ctest.cmake.in runtime-ctest/CTestTestfile.cmake COPYONLY
  )
  install(
    TARGETS
      seeon-deepstream-ipc-test
      seeon-deepstream-source-runtime-test
      seeon-deepstream-source-runtime-gstreamer-test
      seeon-deepstream-native-perception-float-test
      seeon-deepstream-child-commands-test
      seeon-deepstream-au-parser-test
      seeon-deepstream-au-transport-test
      seeon-deepstream-workspace-pool-test
      seeon-deepstream-nvmm-cuda-interop-probe
      seeon-deepstream-preprocess-gpu-test
      seeon-deepstream-postprocess-gpu-test
    RUNTIME DESTINATION native-build/bin
  )
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/runtime-ctest/CTestTestfile.cmake
    DESTINATION native-build
  )
  install(
    FILES src/perception_wire_cross_language_test.py
    DESTINATION native-build/data
  )
endif()

install(TARGETS seeon-deepstream-preflight seeon-deepstream-child RUNTIME DESTINATION bin)
