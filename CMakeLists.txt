cmake_minimum_required(VERSION 3.10)
project(CustomResize)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS} src)

# 添加可执行文件，并指定源文件路径
add_executable(custom_resize src/main.cpp src/resize.cpp)

# 链接OpenCV库
target_link_libraries(custom_resize ${OpenCV_LIBS})
