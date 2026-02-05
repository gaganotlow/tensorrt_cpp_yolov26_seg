cmake_minimum_required(VERSION 3.11)
project(TensorRT_YoloV26_E2E_Segmentation LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

# Find CUDA runtime library
find_library(CUDA_RUNTIME_LIBRARY cudart PATHS /usr/local/cuda/lib64 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)

# YOLOv26 End2End Segmentation utilities (GPU-accelerated)
add_library(yolo_utils SHARED
    utils/postprocess_cuda.cu
    utils/preprocess.cu
)
target_link_libraries(yolo_utils PUBLIC ${OpenCV_LIBS} ${CUDA_RUNTIME_LIBRARY})
set_target_properties(yolo_utils PROPERTIES CUDA_ARCHITECTURES "61;70;75;80;86")


# Build engine from ONNX
add_executable(build
    build.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES})
target_link_libraries(build PRIVATE yolo_utils)
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61;70;75;80;86")

# Runtime inference
add_executable(runtime
    runtime.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
)
target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(runtime PRIVATE ${TensorRT_LIBRARIES})
target_link_libraries(runtime PRIVATE yolo_utils)
set_target_properties(runtime PROPERTIES CUDA_ARCHITECTURES "61;70;75;80;86")
