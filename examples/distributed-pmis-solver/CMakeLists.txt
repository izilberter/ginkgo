cmake_minimum_required(VERSION 3.16)
project(distributed-pmis-solver)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(
    distributed-pmis-solver
    distributed-pmis-solver.cpp
)
target_link_libraries(
    distributed-pmis-solver
    Ginkgo::ginkgo
)
