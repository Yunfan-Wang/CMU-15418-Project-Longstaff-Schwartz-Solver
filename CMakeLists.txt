cmake_minimum_required(VERSION 3.12)
project(longstaff_schwartz_baseline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable extra warnings for debugging
if (NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Find MPI ----
find_package(MPI REQUIRED)

add_executable(lsm_price
    src/main.cpp
)

target_include_directories(lsm_price PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ---- Link MPI ----
target_include_directories(lsm_price SYSTEM PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(lsm_price PRIVATE MPI::MPI_CXX)