// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// This example demonstrates using PMIS algebraic multigrid as a
// preconditioner for CG on a distributed sparse linear system.
// The system is a 1D Laplacian on N grid points distributed across MPI ranks.
//
// Usage:
//   mpirun -n <nranks> ./distributed-pmis-solver [executor] [grid_points]
//
// Executor: reference (default), omp, cuda, hip
// Grid points: 100 (default)

#include <chrono>
#include <iostream>
#include <map>
#include <string>

#include <mpi.h>

#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])
{
    const gko::experimental::mpi::environment env(argc, argv);

    using ValueType = double;
    using LocalIndexType = gko::int32;
    using GlobalIndexType = gko::int64;

    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    using part_type =
        gko::experimental::distributed::Partition<LocalIndexType,
                                                  GlobalIndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using pmis = gko::multigrid::Pmis<ValueType, LocalIndexType>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        ValueType, LocalIndexType, GlobalIndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, LocalIndexType>;

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 100);

    const std::map<std::string,
                   std::function<std::shared_ptr<gko::Executor>(MPI_Comm)>>
        executor_factory{
            {"reference",
             [](MPI_Comm) { return gko::ReferenceExecutor::create(); }},
            {"omp", [](MPI_Comm) { return gko::OmpExecutor::create(); }},
            {"cuda",
             [](MPI_Comm comm) {
                 int device_id = gko::experimental::mpi::map_rank_to_device_id(
                     comm, gko::CudaExecutor::get_num_devices());
                 return gko::CudaExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }},
            {"hip",
             [](MPI_Comm comm) {
                 int device_id = gko::experimental::mpi::map_rank_to_device_id(
                     comm, gko::HipExecutor::get_num_devices());
                 return gko::HipExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }}};

    auto exec = executor_factory.at(executor_string)(MPI_COMM_WORLD);

    // Build a uniform row partition: each rank owns a contiguous block of rows
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(),
        static_cast<GlobalIndexType>(grid_dim)));

    // Assemble the 1D Laplacian [-1, 2, -1] on this rank's rows
    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];

    gko::matrix_data<ValueType, GlobalIndexType> A_data, b_data, x_data;
    A_data.size = {grid_dim, grid_dim};
    b_data.size = x_data.size = {grid_dim, 1};
    for (auto i = range_start; i < range_end; ++i) {
        if (i > 0) A_data.nonzeros.emplace_back(i, i - 1, -1.0);
        A_data.nonzeros.emplace_back(i, i, 2.0);
        if (i < static_cast<GlobalIndexType>(grid_dim) - 1)
            A_data.nonzeros.emplace_back(i, i + 1, -1.0);
        b_data.nonzeros.emplace_back(i, 0, std::sin(i * 0.01));
        x_data.nonzeros.emplace_back(i, 0, 0.0);
    }

    // Read the distributed matrix and vectors on the host, then move to exec
    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto b_host = dist_vec::create(exec->get_master(), comm);
    auto x_host = dist_vec::create(exec->get_master(), comm);
    A_host->read_distributed(A_data, partition);
    b_host->read_distributed(b_data, partition);
    x_host->read_distributed(x_data, partition);

    auto A = gko::share(dist_mtx::create(exec, comm));
    auto b = gko::share(dist_vec::create(exec, comm));
    auto x = dist_vec::create(exec, comm);
    A->copy_from(A_host);
    b->copy_from(b_host);
    x->copy_from(x_host);

    // Smoother: IR with Schwarz-wrapped block-Jacobi (distributed-compatible)
    auto schwarz_bj = gko::share(
        schwarz::build().with_local_solver(bj::build()).on(exec));
    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(schwarz_bj)
            .with_relaxation_factor(ValueType{0.67})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));

    // Coarsest-level solver: CG (supports distributed matrix)
    auto coarsest_gen = gko::share(
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(exec));

    // PMIS multigrid level factory
    auto mg_level_gen = gko::share(
        pmis::build()
            .with_strength_threshold(0.20)
            .on(exec));

    // Multigrid preconditioner
    auto mg_gen = gko::share(
        mg::build()
            .with_max_levels(3u)
            .with_min_coarse_rows(2u)
            .with_pre_smoother(smoother_gen)
            .with_post_uses_pre(true)
            .with_mg_level(mg_level_gen)
            .with_coarsest_solver(coarsest_gen)
            .with_cycle(gko::solver::multigrid::cycle::v)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));

    // Outer CG solver
    auto logger = gko::share(gko::log::Convergence<ValueType>::create());
    auto solver_gen =
        cg::build()
            .with_preconditioner(mg_gen)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(200u),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(1e-8))
            .on(exec);

    // Generate — retry up to 5x: PMIS coarsening is non-deterministic and can
    // intermittently fail with gko::InvalidStateError on some random seeds.
    comm.synchronize();
    ValueType t_gen_start = gko::experimental::mpi::get_walltime();

    std::unique_ptr<gko::LinOp> solver;
    for (int attempt = 1; attempt <= 5; ++attempt) {
        try {
            solver = solver_gen->generate(A);
            break;
        } catch (const gko::InvalidStateError& e) {
            if (rank == 0) {
                std::cerr << "PMIS attempt " << attempt
                          << " failed: " << e.what() << "\n";
            }
            if (attempt == 5) throw;
        }
    }
    solver->add_logger(logger);

    comm.synchronize();
    ValueType t_gen_end = gko::experimental::mpi::get_walltime();

    // Solve
    solver->apply(b, x);

    comm.synchronize();
    ValueType t_solve_end = gko::experimental::mpi::get_walltime();

    // Print results from rank 0
    if (rank == 0) {
        auto res_norm = gko::clone(
            exec->get_master(),
            gko::as<vec>(logger->get_residual_norm()));
        std::cout << "Num rows in matrix: " << grid_dim << "\n"
                  << "Num ranks:          " << comm.size() << "\n"
                  << "CG iterations:      " << logger->get_num_iterations()
                  << "\n"
                  << "Final residual:     " << res_norm->at(0, 0) << "\n"
                  << "Generate time [ms]: "
                  << (t_gen_end - t_gen_start) * 1e3 << "\n"
                  << "Solve time [ms]:    "
                  << (t_solve_end - t_gen_end) * 1e3 << "\n";
    }
}
