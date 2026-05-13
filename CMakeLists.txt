cmake_minimum_required(VERSION 3.10)
project(Multi_stereo)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)

set(DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data" CACHE PATH "Path to SLAM datasets")


find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)
file(GLOB MODULE_SOURCES "modules/*.cpp")
add_executable(multi_stereo_test multi_stereo.cpp ${MODULE_SOURCES})

target_compile_definitions(multi_stereo_test PRIVATE DATA_PATH="${DATA_DIR}")
target_include_directories(multi_stereo_test PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

target_link_libraries(multi_stereo_test PRIVATE ${OpenCV_LIBS})