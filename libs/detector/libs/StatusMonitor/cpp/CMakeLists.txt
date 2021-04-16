cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -std=gnu++11  -pthread -Wmissing-declarations ")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

add_definitions(-D_GLIBCXX_USE_C99=1)

project(demo)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/../opencv/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} .)

add_executable(Demo demo.cpp ActionStatus.cpp)

target_link_libraries(Demo ${OpenCV_LIBS}  -ldl)