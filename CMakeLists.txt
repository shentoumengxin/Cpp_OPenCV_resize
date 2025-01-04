cmake_minimum_required(VERSION 3.10)
project(CustomResize)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 启用 AVX2 和 FMA 指令集
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma")
endif()

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS} src)

# 添加可执行文件，并指定源文件路径
add_executable(custom_resize src/main.cpp src/resize.cpp)

# 链接OpenCV库
target_link_libraries(custom_resize ${OpenCV_LIBS})
