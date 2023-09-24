cmake_minimum_required(VERSION 3.26)
project(GCN)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic -Werror")
# Enable Address Sanitizer (ASan)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp -lstdc++ -lz")

add_executable(GCN main.cpp
        Graph.cpp
        Graph.h
        util.h
        Matrix.cpp
        Matrix.h
        Layer.cpp
        Layer.h
        Model.cpp
        Model.h
        GraphMatrix.h
)

target_link_libraries(GCN PUBLIC "${CMAKE_SOURCE_DIR}/libs/libblis.a")
target_link_libraries(GCN PUBLIC "${CMAKE_SOURCE_DIR}/libs/librsb.a")
target_include_directories(GCN PUBLIC "${CMAKE_SOURCE_DIR}/libs/include")


