cmake_minimum_required(VERSION 3.15)
project(TitanAGI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # For shared libraries

set("OpenCV_DIR" "/usr/local/lib/cmake/opencv4/")
set("OpenCV_LIB" "/usr/local/lib")
# --- 运行时性能优化 ---
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -pthread)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # LTO
    endif()
endif()

# --- 依赖查找 ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED PATHS "/usr/local/include/opencv4/")

# --- 定义公共包含路径 ---
include_directories(include, "/usr/local/include/opencv4/", "3rd")

# --- 核心模块 (Header-only) ---
add_library(titan_core INTERFACE)
target_include_directories(titan_core INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(titan_core INTERFACE Eigen3::Eigen)

# --- 实现模块 ---
# Memory
add_library(titan_memory STATIC src/memory/sparse_gp_memory.cpp)
target_link_libraries(titan_memory PUBLIC titan_core)

# Perception
add_library(titan_perception STATIC src/perception/perception_system.cpp)
target_link_libraries(titan_perception PUBLIC titan_core)

# Control
add_library(titan_control STATIC src/control/fep_controller.cpp)
target_link_libraries(titan_control PUBLIC titan_memory)

# Agent
add_library(titan_agent STATIC src/agent/titan_agent.cpp)
target_link_libraries(titan_agent PUBLIC titan_control titan_perception)

# --- 可执行文件 ---
add_executable(titan_main src/main.cpp)
target_link_libraries(titan_main PRIVATE titan_agent -L${OpenCV_LIB} -lpthread -lopencv_core -lopencv_videoio -lopencv_highgui)