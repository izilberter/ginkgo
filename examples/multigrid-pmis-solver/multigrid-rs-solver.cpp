// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using pmis = gko::multigrid::Pmis<ValueType, IndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Usage: <exe> [executor] [strength_threshold] [max_levels] [cycle: v|f|w]
    //              [smoother_iters] [omega] [coarsest_iters] [truncation_factor]
    // Tuned defaults: threshold=0.20, max_levels=3, cycle=v,
    //                 smoother_iters=1, omega=0.67, coarsest_iters=1,
    //                 truncation_factor=0.0 (disabled)
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const double strength_threshold = argc >= 3 ? std::stod(argv[2]) : 0.20;
    const unsigned int max_levels   = argc >= 4 ? std::stoul(argv[3]) : 3u;
    const std::string cycle_str     = argc >= 5 ? argv[4] : "v";
    const unsigned int smoother_iters = argc >= 6 ? std::stoul(argv[5]) : 1u;
    const double omega              = argc >= 7 ? std::stod(argv[6]) : 0.67;
    const unsigned int coarsest_iters = argc >= 8 ? std::stoul(argv[7]) : 1u;
    const double truncation_factor  = argc >= 9 ? std::stod(argv[8]) : 0.0;

    std::map<std::string, gko::solver::multigrid::cycle> cycle_map{
        {"v", gko::solver::multigrid::cycle::v},
        {"f", gko::solver::multigrid::cycle::f},
        {"w", gko::solver::multigrid::cycle::w},
    };
    if (cycle_map.find(cycle_str) == cycle_map.end()) {
        std::cerr << "Unknown cycle type '" << cycle_str
                  << "'. Use v, f, or w.\n";
        return 1;
    }
    const auto cycle_type = cycle_map.at(cycle_str);

    std::cout << "Parameters:"
              << "  executor=" << executor_string
              << "  threshold=" << strength_threshold
              << "  max_levels=" << max_levels
              << "  cycle=" << cycle_str
              << "  smoother_iters=" << smoother_iters
              << "  omega=" << omega
              << "  coarsest_iters=" << coarsest_iters
              << "  truncation_factor=" << truncation_factor
              << "\n";

    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    const auto exec = exec_map.at(executor_string)();

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    gko::size_type size = A->get_size()[0];
//    auto host_x = share(gko::read<vec>(std::ifstream("data/x.mtx"), exec->get_master()));
//    auto host_b = share(gko::read<vec>(std::ifstream("data/b.mtx"), exec->get_master()));
    // Create RHS as 1 and initial guess as 0
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    auto host_b = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
        host_b->at(i, 0) = 1.;
    }
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    x->copy_from(host_x);
    b->copy_from(host_b);

    // Calculate initial residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);
    b->copy_from(host_b);

    // Scale b (and x) to O(1) — avoids rhs_norm NaN when ||b|| << 1
    auto b_norm_dev = gko::initialize<vec>({0.0}, exec);
    b->compute_norm2(b_norm_dev);
    auto b_norm_host = gko::initialize<vec>({0.0}, exec->get_master());
    b_norm_host->copy_from(b_norm_dev);
    const ValueType b_scale = b_norm_host->at(0, 0);
    std::cout << "||b|| = " << b_scale << "  (scaling system by 1/||b||)\n";
    auto inv_b_scale = gko::initialize<vec>({ValueType{1} / b_scale}, exec);
    auto b_scale_vec = gko::initialize<vec>({b_scale}, exec);
    b->scale(inv_b_scale);
    x->scale(inv_b_scale);

    // Stopping criteria
    const gko::remove_complex<ValueType> tolerance = 1e-14;
    auto iter_stop =
        gko::share(gko::stop::Iteration::build().with_max_iters(200u).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_baseline(gko::stop::mode::rhs_norm)
                                   .with_reduction_factor(tolerance)
                                   .on(exec));

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // Jacobi (block size 1) as inner preconditioner
    auto jacobi_gen =
        gko::share(bj::build().with_max_block_size(1u).on(exec));

    // Pre/post smoother: IR(Jacobi, smoother_iters steps, omega)
    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(jacobi_gen)
            .with_relaxation_factor(static_cast<ValueType>(omega))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(smoother_iters))
            .on(exec));

    // Coarsest-level solver: IR(Jacobi, coarsest_iters steps, omega)
    auto coarsest_gen = gko::share(
        ir::build()
            .with_solver(jacobi_gen)
            .with_relaxation_factor(static_cast<ValueType>(omega))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(coarsest_iters))
            .on(exec));

    // PMIS multigrid level factory
    auto mg_level_gen =
        gko::share(pmis::build()
                .with_skip_sorting(true)
                .with_strength_threshold(strength_threshold)
                .with_truncation_factor(truncation_factor)
                .on(exec));

    // Multigrid preconditioner
    auto multigrid_gen =
        mg::build()
            .with_max_levels(max_levels)
            .with_min_coarse_rows(2u)
            .with_pre_smoother(smoother_gen)
            .with_post_uses_pre(true)
            .with_mg_level(mg_level_gen)
            .with_coarsest_solver(coarsest_gen)
            .with_cycle(cycle_type)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);

    // Outer CG solver
    auto solver_gen = cg::build()
                          .with_criteria(iter_stop, tol_stop)
                          .with_preconditioner(gko::share(std::move(multigrid_gen)))
                          .on(exec);

    // Generate — retry up to 5x because PMIS coarsening is non-deterministic
    // and can intermittently throw gko::InvalidStateError ("no progress").
    std::chrono::nanoseconds gen_time(0);
    std::unique_ptr<gko::LinOp> solver;
    for (int attempt = 1; attempt <= 5; ++attempt) {
        try {
            auto gen_tic = std::chrono::steady_clock::now();
            solver = solver_gen->generate(A);
            exec->synchronize();
            auto gen_toc = std::chrono::steady_clock::now();
            gen_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                gen_toc - gen_tic);
            break;
        } catch (const gko::InvalidStateError& e) {
            std::cerr << "PMIS attempt " << attempt << " failed: " << e.what()
                      << "\n";
            if (attempt == 5) throw;
        }
    }

    // Print multigrid hierarchy
    {
        auto* mg_solver = gko::as<mg>(
            gko::as<cg>(solver.get())->get_preconditioner().get());
        const auto& levels = mg_solver->get_mg_level_list();
        std::cout << "\nMultigrid hierarchy (" << levels.size()
                  << " levels):\n";
        std::cout << std::setw(9)  << "Level"
                  << std::setw(12) << "Fine rows"
                  << std::setw(12) << "Coarse rows"
                  << std::setw(8)  << "Ratio"
                  << std::setw(12) << "P nnz"
                  << std::setw(12) << "R nnz"
                  << std::setw(12) << "Ac nnz" << "\n";
        for (gko::size_type i = 0; i < levels.size(); ++i) {
            const auto& lvl = levels[i];
            auto fine_rows   = lvl->get_fine_op()->get_size()[0];
            auto coarse_rows = lvl->get_coarse_op()->get_size()[0];
            auto p_nnz  = gko::as<const mtx>(lvl->get_prolong_op())
                              ->get_num_stored_elements();
            auto r_nnz  = gko::as<const mtx>(lvl->get_restrict_op())
                              ->get_num_stored_elements();
            auto ac_nnz = gko::as<const mtx>(lvl->get_coarse_op())
                              ->get_num_stored_elements();
            std::cout << std::setw(9)  << i
                      << std::setw(12) << fine_rows
                      << std::setw(12) << coarse_rows
                      << std::fixed << std::setprecision(2)
                      << std::setw(8)  << static_cast<double>(fine_rows) /
                                          static_cast<double>(coarse_rows)
                      << std::defaultfloat
                      << std::setw(12) << p_nnz
                      << std::setw(12) << r_nnz
                      << std::setw(12) << ac_nnz << "\n";
        }
        // Coarsest system (solved directly, no further coarsening)
        auto coarsest_rows =
            levels.back()->get_coarse_op()->get_size()[0];
        auto coarsest_nnz = gko::as<const mtx>(levels.back()->get_coarse_op())
                                ->get_num_stored_elements();
        std::cout << std::setw(9)  << "coarsest"
                  << std::setw(12) << coarsest_rows
                  << std::setw(12) << "(solver)"
                  << std::setw(8)  << "-"
                  << std::setw(12) << "-"
                  << std::setw(12) << "-"
                  << std::setw(12) << coarsest_nnz << "\n";
        std::cout << "\n";
    }

    // Solve
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Recover true x and b, then calculate residual
    x->scale(b_scale_vec);
    b->copy_from(host_b);
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    const int num_iters = static_cast<int>(logger->get_num_iterations());
    const double solve_ms = static_cast<double>(time.count()) / 1e6;
    std::cout << "CG iteration count:              " << num_iters << "\n";
    std::cout << "CG generation time [ms]:         "
              << static_cast<double>(gen_time.count()) / 1e6 << "\n";
    std::cout << "CG execution time [ms]:          " << solve_ms << "\n";
    std::cout << "CG execution time per iter [ms]: "
              << solve_ms / num_iters << "\n";

    // Machine-readable summary line for sweep scripts
    std::cout << "RESULT threshold=" << strength_threshold
              << " max_levels=" << max_levels
              << " cycle=" << cycle_str
              << " smoother_iters=" << smoother_iters
              << " omega=" << omega
              << " coarsest_iters=" << coarsest_iters
              << " truncation_factor=" << truncation_factor
              << " cg_iters=" << num_iters
              << " solve_ms=" << solve_ms
              << " ms_per_iter=" << solve_ms / num_iters
              << "\n";
}
