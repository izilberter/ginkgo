cmake_minimum_required(VERSION 3.16)
project(multigrid-rs-solver)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(
    multigrid-rs-solver
    multigrid-rs-solver.cpp
)
target_link_libraries(multigrid-rs-solver Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
#configure_file(data/b.mtx data/b.mtx COPYONLY)
#configure_file(data/x.mtx data/x.mtx COPYONLY)
