cmake_minimum_required(VERSION 3.10)
project(AeroTrack)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-1.21.0")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")

# Find Dependencies
find_package(OpenCV REQUIRED)

# Include the headers
include_directories(${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})

# Create the Executable
add_executable(AeroTrack main.cpp detector.cpp tracker.cpp)

# Link the Libraries
target_link_libraries(AeroTrack PRIVATE 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)