// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/multigrid/pmis.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/collective_communicator.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/row_gatherer.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/array_access.hpp"
#include "core/base/dispatch_helper.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/precision_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/distributed/index_map_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/multigrid/pmis_kernels.hpp"


namespace gko {
namespace multigrid {
namespace pmis {
namespace {


GKO_REGISTER_OPERATION(compute_row_maxabs, pmis::compute_row_maxabs);
GKO_REGISTER_OPERATION(compute_strong_dep_row, pmis::compute_strong_dep_row);
GKO_REGISTER_OPERATION(compute_strong_dep, pmis::compute_strong_dep);
GKO_REGISTER_OPERATION(initialize_weight_and_status,
                       pmis::initialize_weight_and_status);
GKO_REGISTER_OPERATION(classify, pmis::classify);
GKO_REGISTER_OPERATION(count, pmis::count);
GKO_REGISTER_OPERATION(direct_interpolation_row_count,
                       pmis::direct_interpolation_row_count);
GKO_REGISTER_OPERATION(direct_interpolation_fill,
                       pmis::direct_interpolation_fill);
GKO_REGISTER_OPERATION(truncate_prolongation_count,
                       pmis::truncate_prolongation_count);
GKO_REGISTER_OPERATION(truncate_prolongation_fill,
                       pmis::truncate_prolongation_fill);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(convert_precision, components::convert_precision);
// Distributed kernels
GKO_REGISTER_OPERATION(update_row_maxabs_off_diag,
                       pmis::update_row_maxabs_off_diag);
GKO_REGISTER_OPERATION(compute_strong_ghost_dep_row,
                       pmis::compute_strong_ghost_dep_row);
GKO_REGISTER_OPERATION(compute_strong_ghost_dep,
                       pmis::compute_strong_ghost_dep);
GKO_REGISTER_OPERATION(classify_with_ghosts, pmis::classify_with_ghosts);
GKO_REGISTER_OPERATION(direct_interpolation_row_count_dist,
                       pmis::direct_interpolation_row_count_dist);
GKO_REGISTER_OPERATION(direct_interpolation_fill_diag,
                       pmis::direct_interpolation_fill_diag);
GKO_REGISTER_OPERATION(direct_interpolation_fill_off_diag,
                       pmis::direct_interpolation_fill_off_diag);
}  // anonymous namespace
}  // namespace pmis


#if GINKGO_BUILD_MPI

namespace index_map_ops {
namespace {

GKO_REGISTER_OPERATION(map_to_global, index_map::map_to_global);

}
}  // namespace index_map_ops

#endif  // GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
typename Pmis<ValueType, IndexType>::parameters_type
Pmis<ValueType, IndexType>::parse(const config::pnode& config,
                                  const config::registry& context,
                                  const config::type_descriptor& td_for_child)
{
    auto params = Pmis<ValueType, IndexType>::build();
    config::config_check_decorator config_check(config);
    if (auto& obj = config_check.get("strength_threshold")) {
        params.with_strength_threshold(
            config::get_value<remove_complex<ValueType>>(obj));
    }
    if (auto& obj = config_check.get("skip_sorting")) {
        params.with_skip_sorting(config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("truncation_factor")) {
        params.with_truncation_factor(
            config::get_value<remove_complex<ValueType>>(obj));
    }

    return params;
}


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    this->get_composition()->apply(b, x);
}


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                            const LinOp* beta, LinOp* x) const
{
    this->get_composition()->apply(alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
Pmis<ValueType, IndexType>::Pmis(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec))
{}


template <typename ValueType, typename IndexType>
Pmis<ValueType, IndexType>::Pmis(const Factory* factory,
                                 std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), system_matrix->get_size()),
      EnableMultigridLevel<ValueType>(system_matrix),
      parameters_{factory->get_parameters()},
      system_matrix_{system_matrix}
{
    GKO_ASSERT(parameters_.strength_threshold <= 1.0);
    GKO_ASSERT(parameters_.strength_threshold >= 0.0);
    if (system_matrix_->get_size()[0] != 0) {
        // generate on the existed matrix
        this->generate();
    }
}


#if GINKGO_BUILD_MPI

// Exchange an array of T values for all ghost (off-diagonal) nodes.
// send_idxs[0..send_size) are LOCAL indices whose values we send to other ranks.
// nlocal is the total length of local_values (needed to copy to host).
// Always works through host memory to avoid device-kernel type constraints.
// Returns an array of size recv_size (on exec) containing values for ghost nodes.
template <typename T, typename IndexType>
static array<T> halo_exchange(
    std::shared_ptr<const Executor> exec,
    const experimental::mpi::communicator& comm,
    const experimental::mpi::CollectiveCommunicator* coll_comm,
    size_type nlocal,
    const IndexType* send_idxs,
    const T* local_values)
{
    const auto send_size = coll_comm->get_send_size();
    const auto recv_size = coll_comm->get_recv_size();
    auto host = exec->get_master();

    // Copy send_idxs and local_values to host, gather send buffer
    array<IndexType> h_idxs(host, send_size);
    host->copy_from(exec, send_size, send_idxs, h_idxs.get_data());
    array<T> h_local(host, nlocal);
    host->copy_from(exec, nlocal, local_values, h_local.get_data());
    array<T> h_send(host, send_size);
    for (size_type i = 0; i < send_size; i++) {
        h_send.get_data()[i] = h_local.get_const_data()[h_idxs.get_const_data()[i]];
    }

    // MPI exchange (always through host)
    array<T> h_recv(host, recv_size);
    exec->synchronize();
    coll_comm->i_all_to_all_v(host, h_send.get_const_data(), h_recv.get_data())
        .wait();

    // Copy result to exec
    array<T> recv_buf(exec, recv_size);
    exec->copy_from(host, recv_size, h_recv.get_const_data(), recv_buf.get_data());
    return recv_buf;
}


template <typename ValueType, typename IndexType>
template <typename GlobalIndexType>
void Pmis<ValueType, IndexType>::generate_distributed(
    std::shared_ptr<const experimental::distributed::Matrix<
        ValueType, IndexType, GlobalIndexType>>
        matrix)
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
    using dist_matrix_type =
        experimental::distributed::Matrix<ValueType, IndexType, GlobalIndexType>;

    // 64-bit local indices are not supported: the coarse matrix construction
    // uses SpGeMM, which CUDA's cuSPARSE only supports with 32-bit indices.
    if (sizeof(IndexType) > sizeof(int32)) GKO_NOT_IMPLEMENTED;

    auto exec = this->get_executor();
    auto comm = matrix->get_communicator();

    // Extract row-gatherer info here (Pmis is a friend of Matrix)
    auto coll_comm = matrix->row_gatherer_->get_collective_communicator();
    const auto* send_idxs = matrix->row_gatherer_->get_const_send_idxs();

    auto diag_csr = gko::as<const csr_type>(matrix->get_diag_matrix());
    auto off_diag_csr = gko::as<const csr_type>(matrix->get_off_diag_matrix());
    const auto nrows = diag_csr->get_size()[0];
    const auto nghost = off_diag_csr->get_size()[1];
    const auto threshold = this->get_parameters().strength_threshold;

    // 1. Compute row_maxabs over both diag and off-diag blocks
    array<real_type> row_maxabs(exec, nrows);
    exec->run(pmis::make_compute_row_maxabs(diag_csr.get(), row_maxabs.get_data()));
    exec->run(pmis::make_update_row_maxabs_off_diag(off_diag_csr.get(),
                                                     row_maxabs.get_data()));

    // 2. Build local strong dep (diag block)
    array<IndexType> sparsity_rows(exec, nrows + 1);
    exec->run(pmis::make_compute_strong_dep_row(diag_csr.get(),
        row_maxabs.get_const_data(), threshold, sparsity_rows.get_data()));
    exec->run(pmis::make_prefix_sum_nonnegative(sparsity_rows.get_data(),
                                                nrows + 1));
    auto local_nnz = get_element(sparsity_rows, nrows);
    array<IndexType> local_cols(exec, local_nnz);
    auto strong_dep = matrix::SparsityCsr<ValueType, IndexType>::create(
        exec, diag_csr->get_size(),
        std::move(local_cols), std::move(sparsity_rows));
    exec->run(pmis::make_compute_strong_dep(diag_csr.get(),
        row_maxabs.get_const_data(), threshold, strong_dep.get()));

    // 3. Build ghost strong dep (off-diag block, col indices = local ghost indices)
    array<IndexType> ghost_sparsity_rows(exec, nrows + 1);
    exec->run(pmis::make_compute_strong_ghost_dep_row(off_diag_csr.get(),
        row_maxabs.get_const_data(), threshold, ghost_sparsity_rows.get_data()));
    exec->run(pmis::make_prefix_sum_nonnegative(ghost_sparsity_rows.get_data(),
                                                 nrows + 1));
    auto ghost_nnz = get_element(ghost_sparsity_rows, nrows);
    array<IndexType> ghost_cols(exec, ghost_nnz);
    dim<2> ghost_dep_size{nrows, nghost};
    auto strong_ghost_dep = matrix::SparsityCsr<ValueType, IndexType>::create(
        exec, ghost_dep_size,
        std::move(ghost_cols), std::move(ghost_sparsity_rows));
    exec->run(pmis::make_compute_strong_ghost_dep(off_diag_csr.get(),
        row_maxabs.get_const_data(), threshold, strong_ghost_dep.get()));

    // 4. Initialize weight and status from local transpose strong dep
    auto trans_strong_dep =
        as<matrix::SparsityCsr<ValueType, IndexType>>(strong_dep->transpose());
    array<real_type> weight(exec, nrows);
    array<int> status(exec, nrows);
    array<int> new_status(exec, nrows);
    exec->run(pmis::make_initialize_weight_and_status(
        trans_strong_dep.get(), weight.get_data(), status.get_data()));

    // 5. Communicate ghost weights once (they don't change)
    array<real_type> ghost_weight =
        halo_exchange<real_type>(exec, comm, coll_comm.get(), nrows, send_idxs,
                                 weight.get_const_data());
    array<int> ghost_status(exec, nghost);

    // 6. Classify loop with ghost communication (use raw pointer swap like local case)
    auto* status_ptr = status.get_data();
    auto* new_status_ptr = new_status.get_data();
    size_type local_num = 0;
    exec->run(pmis::make_count(nrows, status_ptr, &local_num));
    // Use global count so all ranks loop the same number of times (avoid MPI deadlock)
    size_type global_num = local_num;
    comm.all_reduce(exec->get_master(), &global_num, 1, MPI_SUM);
    while (global_num != 0) {
        ghost_status = halo_exchange<int>(exec, comm, coll_comm.get(), nrows,
                                          send_idxs, status_ptr);
        exec->run(pmis::make_classify_with_ghosts(
            weight.get_const_data(), strong_dep.get(), strong_ghost_dep.get(),
            ghost_weight.get_const_data(), ghost_status.get_const_data(),
            status_ptr, new_status_ptr));
        size_type new_local = 0;
        exec->run(pmis::make_count(nrows, new_status_ptr, &new_local));
        size_type new_global = new_local;
        comm.all_reduce(exec->get_master(), &new_global, 1, MPI_SUM);
        GKO_THROW_IF_INVALID(new_global != global_num, "no progress in Pmis");
        global_num = new_global;
        std::swap(status_ptr, new_status_ptr);
    }

    // 7. Build local coarse_map (prefix sum of coarse indicators)
    array<IndexType> coarse_map(exec, nrows + 1);
    static_assert(kernels::pmis::coarse == 1 && kernels::pmis::fine == 0,
                  "prefix sum directly gives coarse count");
    exec->run(pmis::make_convert_precision(nrows, status_ptr,
                                            coarse_map.get_data()));
    exec->run(pmis::make_prefix_sum_nonnegative(coarse_map.get_data(), nrows + 1));
    const auto num_local_coarse =
        static_cast<int64>(get_element(coarse_map, nrows));

    // 8. Build coarse partition (one contiguous range per rank)
    auto coarse_partition = gko::share(
        experimental::distributed::build_partition_from_local_size<
            IndexType, GlobalIndexType>(exec, comm, num_local_coarse));

    // 9. Communicate ghost coarse global indices
    // For each local node i: send its coarse global index (or sentinel=-1 if fine).
    const GlobalIndexType sentinel = -1;
    // coarse_partition lives on exec (may be a GPU), so get_range_bounds() returns
    // a device pointer. Copy the needed element to host before indexing.
    GlobalIndexType part_offset;
    exec->get_master()->copy_from(exec, 1,
        coarse_partition->get_range_bounds() + comm.rank(), &part_offset);

    const auto send_size = coll_comm->get_send_size();
    const auto recv_size = coll_comm->get_recv_size();

    // Build "coarse global index or sentinel" for each local node (on host)
    array<GlobalIndexType> local_coarse_global(exec->get_master(), nrows);
    {
        array<IndexType> h_coarse_map(exec->get_master(), nrows + 1);
        array<int> h_status(exec->get_master(), nrows);
        exec->get_master()->copy_from(exec, nrows + 1,
            coarse_map.get_const_data(), h_coarse_map.get_data());
        exec->get_master()->copy_from(exec, nrows, status_ptr, h_status.get_data());
        for (size_type i = 0; i < nrows; i++) {
            if (h_status.get_const_data()[i] == kernels::pmis::coarse) {
                local_coarse_global.get_data()[i] =
                    part_offset + static_cast<GlobalIndexType>(
                        h_coarse_map.get_const_data()[i]);
            } else {
                local_coarse_global.get_data()[i] = sentinel;
            }
        }
    }
    array<GlobalIndexType> local_coarse_global_dev(exec, nrows);
    exec->copy_from(exec->get_master(), nrows,
                    local_coarse_global.get_const_data(),
                    local_coarse_global_dev.get_data());

    // Gather + halo-exchange to get ghost coarse global indices
    auto off_diag_coarse_global = halo_exchange<GlobalIndexType>(
        exec, comm, coll_comm.get(), nrows, send_idxs,
        local_coarse_global_dev.get_const_data());

    // 10. Build coarse index map for the off-diagonal block.
    // Copy off_diag_coarse_global to host for processing.
    array<GlobalIndexType> h_off_diag(exec->get_master(), recv_size);
    exec->get_master()->copy_from(exec, recv_size,
        off_diag_coarse_global.get_const_data(), h_off_diag.get_data());

    // Filter out sentinel entries before constructing the index_map (find_range
    // would return garbage for negative global indices).
    array<GlobalIndexType> h_off_diag_valid(exec->get_master(), recv_size);
    size_type valid_count = 0;
    for (size_type k = 0; k < recv_size; k++) {
        if (h_off_diag.get_const_data()[k] != sentinel) {
            h_off_diag_valid.get_data()[valid_count++] =
                h_off_diag.get_const_data()[k];
        }
    }
    array<GlobalIndexType> off_diag_coarse_valid(exec, valid_count);
    exec->copy_from(exec->get_master(), valid_count,
                    h_off_diag_valid.get_const_data(),
                    off_diag_coarse_valid.get_data());

    auto coarse_imap =
        experimental::distributed::index_map<IndexType, GlobalIndexType>(
            exec, coarse_partition, comm.rank(), off_diag_coarse_valid);

    // Map ghost coarse global indices to local off-diagonal coarse indices.
    // Sentinel entries (-1) are not in the imap and will map to invalid_index.
    auto off_diag_coarse_local = coarse_imap.map_to_local(
        off_diag_coarse_global,
        experimental::distributed::index_space::non_local);

    // Build ghost_coarse_status and ghost_coarse_local_idx arrays (host → device)
    array<int> ghost_coarse_status_arr(exec, nghost);
    array<IndexType> ghost_coarse_local_idx_arr(exec, nghost);
    {
        array<IndexType> h_local(exec->get_master(), recv_size);
        exec->get_master()->copy_from(exec, recv_size,
            off_diag_coarse_local.get_const_data(), h_local.get_data());
        array<int> h_gcs(exec->get_master(), nghost);
        array<IndexType> h_gcl(exec->get_master(), nghost);
        for (size_type k = 0; k < nghost; k++) {
            if (h_off_diag.get_const_data()[k] != sentinel) {
                h_gcs.get_data()[k] = kernels::pmis::coarse;
                h_gcl.get_data()[k] = h_local.get_const_data()[k];
            } else {
                h_gcs.get_data()[k] = kernels::pmis::fine;
                h_gcl.get_data()[k] = -1;
            }
        }
        exec->copy_from(exec->get_master(), nghost,
                        h_gcs.get_const_data(), ghost_coarse_status_arr.get_data());
        exec->copy_from(exec->get_master(), nghost,
                        h_gcl.get_const_data(), ghost_coarse_local_idx_arr.get_data());
    }

    // 11. Build prolongation row ptrs (diag + off-diag blocks)
    array<IndexType> prolong_diag_rows(exec, nrows + 1);
    array<IndexType> prolong_off_diag_rows(exec, nrows + 1);
    exec->run(pmis::make_direct_interpolation_row_count_dist(
        strong_dep.get(), strong_ghost_dep.get(),
        status_ptr, ghost_coarse_status_arr.get_const_data(),
        prolong_diag_rows.get_data(), prolong_off_diag_rows.get_data()));
    exec->run(pmis::make_prefix_sum_nonnegative(prolong_diag_rows.get_data(),
                                                 nrows + 1));
    exec->run(pmis::make_prefix_sum_nonnegative(prolong_off_diag_rows.get_data(),
                                                 nrows + 1));
    IndexType prolong_diag_nnz = get_element(prolong_diag_rows, nrows);
    IndexType prolong_off_diag_nnz = get_element(prolong_off_diag_rows, nrows);

    // 12. Fill prolongation diag and off-diag blocks
    array<IndexType> prolong_diag_cols(exec, prolong_diag_nnz);
    array<ValueType> prolong_diag_vals(exec, prolong_diag_nnz);
    exec->run(pmis::make_direct_interpolation_fill_diag(
        diag_csr.get(), off_diag_csr.get(), row_maxabs.get_const_data(),
        threshold, coarse_map.get_const_data(),
        ghost_coarse_status_arr.get_const_data(),
        prolong_diag_rows.get_const_data(),
        prolong_diag_cols.get_data(), prolong_diag_vals.get_data()));

    array<IndexType> prolong_off_diag_cols(exec, prolong_off_diag_nnz);
    array<ValueType> prolong_off_diag_vals(exec, prolong_off_diag_nnz);
    exec->run(pmis::make_direct_interpolation_fill_off_diag(
        diag_csr.get(), off_diag_csr.get(), row_maxabs.get_const_data(),
        threshold, coarse_map.get_const_data(),
        ghost_coarse_local_idx_arr.get_const_data(),
        ghost_coarse_status_arr.get_const_data(),
        prolong_off_diag_rows.get_const_data(),
        prolong_off_diag_cols.get_data(), prolong_off_diag_vals.get_data()));

    // 13. Assemble distributed prolongation, restriction, coarse matrix
    const auto fine_size = gko::as<LinOp>(matrix)->get_size();
    const auto coarse_global_size = coarse_partition->get_size();
    const auto n_off_coarse = coarse_imap.get_non_local_size();

    auto prolong_diag_csr = share(csr_type::create(
        exec, dim<2>{nrows, static_cast<size_type>(num_local_coarse)},
        std::move(prolong_diag_vals), std::move(prolong_diag_cols),
        std::move(prolong_diag_rows)));

    auto prolong_off_diag_csr = share(csr_type::create(
        exec, dim<2>{nrows, n_off_coarse},
        std::move(prolong_off_diag_vals), std::move(prolong_off_diag_cols),
        std::move(prolong_off_diag_rows)));

    // Restriction local block: P_diag^T
    auto restrict_diag = share(as<csr_type>(prolong_diag_csr->transpose()));

    // Coarse matrix diag block: P_diag^T * A_diag * P_diag (local SpGeMM)
    // Output matrices must be pre-allocated with correct dims for Ginkgo SpGeMM.
    auto AP_diag = csr_type::create(exec, dim<2>{nrows, static_cast<size_type>(num_local_coarse)});
    diag_csr->apply(prolong_diag_csr.get(), AP_diag.get());
    auto A_c_diag = share(csr_type::create(exec, dim<2>{static_cast<size_type>(num_local_coarse), static_cast<size_type>(num_local_coarse)}));
    restrict_diag->apply(AP_diag.get(), A_c_diag.get());

    // Coarse matrix off-diag block: P_diag^T * (A_off * Q)
    // Q maps ghost fine → ghost coarse (Q[k, ghost_coarse_local_idx[k]] = 1 if coarse)
    auto A_c_off = share(csr_type::create(
        exec, dim<2>{static_cast<size_type>(num_local_coarse), n_off_coarse}));
    if (n_off_coarse > 0 && nghost > 0) {
        // Build Q on host from ghost_coarse arrays already in memory
        array<int> h_gcs(exec->get_master(), nghost);
        array<IndexType> h_gcl(exec->get_master(), nghost);
        exec->get_master()->copy_from(
            exec, nghost, ghost_coarse_status_arr.get_const_data(), h_gcs.get_data());
        exec->get_master()->copy_from(
            exec, nghost, ghost_coarse_local_idx_arr.get_const_data(), h_gcl.get_data());
        size_type q_nnz = 0;
        for (size_type k = 0; k < nghost; k++) {
            if (h_gcs.get_const_data()[k] == kernels::pmis::coarse) q_nnz++;
        }
        array<IndexType> h_q_rows(exec->get_master(), nghost + 1);
        array<IndexType> h_q_cols(exec->get_master(), q_nnz);
        array<ValueType> h_q_vals(exec->get_master(), q_nnz);
        h_q_rows.get_data()[0] = 0;
        size_type qi = 0;
        for (size_type k = 0; k < nghost; k++) {
            h_q_rows.get_data()[k + 1] = h_q_rows.get_data()[k];
            if (h_gcs.get_const_data()[k] == kernels::pmis::coarse) {
                h_q_cols.get_data()[qi] = h_gcl.get_const_data()[k];
                h_q_vals.get_data()[qi] = one<ValueType>();
                h_q_rows.get_data()[k + 1]++;
                qi++;
            }
        }
        array<IndexType> d_q_rows(exec, nghost + 1);
        array<IndexType> d_q_cols(exec, q_nnz);
        array<ValueType> d_q_vals(exec, q_nnz);
        exec->copy_from(exec->get_master(), nghost + 1,
                        h_q_rows.get_const_data(), d_q_rows.get_data());
        exec->copy_from(exec->get_master(), q_nnz,
                        h_q_cols.get_const_data(), d_q_cols.get_data());
        exec->copy_from(exec->get_master(), q_nnz,
                        h_q_vals.get_const_data(), d_q_vals.get_data());
        auto Q = share(csr_type::create(exec, dim<2>{nghost, n_off_coarse},
            std::move(d_q_vals), std::move(d_q_cols), std::move(d_q_rows)));
        auto AQ = csr_type::create(exec, dim<2>{nrows, n_off_coarse});
        off_diag_csr->apply(Q.get(), AQ.get());
        restrict_diag->apply(AQ.get(), A_c_off.get());
    }

    // Build a second imap for the coarse distributed matrix (same partition,
    // same remote connections as the prolongation's off-diag).
    auto coarse_imap_for_coarse =
        experimental::distributed::index_map<IndexType, GlobalIndexType>(
            exec, coarse_partition, comm.rank(), off_diag_coarse_valid);

    // Distributed coarse matrix
    auto coarse = share(dist_matrix_type::create(
        exec, comm, std::move(coarse_imap_for_coarse), A_c_diag, A_c_off));

    // Distributed prolongation (diag + off-diag blocks).
    // The imap constructor sets size to {coarse_global x coarse_global}, but
    // prolongation must be {fine_global x coarse_global}.  Fix via set_size
    // (accessible because Pmis is a friend of dist_matrix_type).
    auto prolongation_raw = dist_matrix_type::create(
        exec, comm, std::move(coarse_imap), prolong_diag_csr,
        prolong_off_diag_csr);
    prolongation_raw->set_size(dim<2>{fine_size[0], coarse_global_size});
    auto prolongation = share(std::move(prolongation_raw));

    // Distributed restriction (local block only: P_diag^T)
    auto restrict_op = share(dist_matrix_type::create(
        exec, comm,
        dim<2>{coarse_global_size, fine_size[0]},
        restrict_diag));

    this->set_multigrid_level(prolongation, coarse, restrict_op);
}


#endif  // GINKGO_BUILD_MPI


template <typename ValueType, typename IndexType>
void Pmis<ValueType, IndexType>::generate()
{
    using csr_type = matrix::Csr<ValueType, IndexType>;
    using real_type = remove_complex<ValueType>;
#if GINKGO_BUILD_MPI
    if (std::dynamic_pointer_cast<const experimental::distributed::DistributedBase>(
            system_matrix_)) {
        // Ensure fine op is a distributed::Matrix with CSR local blocks
        auto convert_fine_op = [&](auto matrix) {
            using global_index_type =
                typename std::decay_t<decltype(*matrix)>::result_type::global_index_type;
            auto exec = as<LinOp>(matrix)->get_executor();
            auto comm =
                as<experimental::distributed::DistributedBase>(matrix)
                    ->get_communicator();
            auto fine = share(
                experimental::distributed::Matrix<ValueType, IndexType,
                                                  global_index_type>::create(
                    exec, comm,
                    matrix::Csr<ValueType, IndexType>::create(exec),
                    matrix::Csr<ValueType, IndexType>::create(exec)));
            matrix->convert_to(fine);
            this->set_fine_op(fine);
        };
        auto setup_fine_op = [&](auto matrix) {
            auto diag_csr = std::dynamic_pointer_cast<const csr_type>(
                matrix->get_diag_matrix());
            auto off_diag_csr = std::dynamic_pointer_cast<const csr_type>(
                matrix->get_off_diag_matrix());
            if (!parameters_.skip_sorting || !diag_csr || !off_diag_csr) {
                using global_index_type =
                    typename std::decay_t<decltype(*matrix)>::global_index_type;
                convert_fine_op(
                    as<ConvertibleTo<experimental::distributed::Matrix<
                        ValueType, IndexType, global_index_type>>>(matrix));
            }
        };
        using fst_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, IndexType>;
        using snd_mtx_type =
            experimental::distributed::Matrix<ValueType, IndexType, int64>;
        if (auto obj = std::dynamic_pointer_cast<const fst_mtx_type>(system_matrix_)) {
            setup_fine_op(obj);
        } else if (auto obj = std::dynamic_pointer_cast<const snd_mtx_type>(system_matrix_)) {
            setup_fine_op(obj);
        } else {
            run<ConvertibleTo, fst_mtx_type, snd_mtx_type>(system_matrix_,
                                                            convert_fine_op);
        }
        auto distributed_setup = [&](auto matrix) {
            this->generate_distributed(
                std::dynamic_pointer_cast<
                    const std::decay_t<decltype(*matrix)>>(
                    this->get_fine_op()));
        };
        run<fst_mtx_type, snd_mtx_type>(this->get_fine_op(), distributed_setup);
        return;
    }
#endif  // GINKGO_BUILD_MPI
    auto exec = this->get_executor();
    // Only support csr matrix currently.
    auto pmis_op = std::dynamic_pointer_cast<const csr_type>(system_matrix_);
    // If system matrix is not csr or need sorting, generate the csr.
    if (!parameters_.skip_sorting || !pmis_op) {
        pmis_op = convert_to_with_sorting<csr_type>(exec, system_matrix_,
                                                    parameters_.skip_sorting);
        // keep the same precision data in fine_op
        this->set_fine_op(pmis_op);
    }

    array<IndexType> sparsity_rows(exec, pmis_op->get_size()[0] + 1);
    array<remove_complex<ValueType>> row_maxabs(exec, pmis_op->get_size()[0]);
    // weight = the number of strong dependence + rand[0, 1]
    gko::array<remove_complex<ValueType>> weight_(exec, pmis_op->get_size()[0]);
    exec->run(
        pmis::make_compute_row_maxabs(pmis_op.get(), row_maxabs.get_data()));
    // the number of nonzero in strong_dep of node i (#S_i) into sparsity_row i
    exec->run(pmis::make_compute_strong_dep_row(
        pmis_op.get(), row_maxabs.get_const_data(),
        this->get_parameters().strength_threshold, sparsity_rows.get_data()));
    // build offset
    exec->run(pmis::make_prefix_sum_nonnegative(sparsity_rows.get_data(),
                                                sparsity_rows.get_size()));
    auto nnz = get_element(sparsity_rows, pmis_op->get_size()[0]);
    array<IndexType> sparsity_cols(exec, nnz);
    auto strong_dep = matrix::SparsityCsr<ValueType, IndexType>::create(
        exec, pmis_op->get_size(), std::move(sparsity_cols),
        std::move(sparsity_rows));
    // fill column index into sparsity csr
    exec->run(pmis::make_compute_strong_dep(
        pmis_op.get(), row_maxabs.get_const_data(),
        this->get_parameters().strength_threshold, strong_dep.get()));
    auto transpose_strong_dep =
        as<matrix::SparsityCsr<ValueType, IndexType>>(strong_dep->transpose());
    // weight[i] = #S^T + rand(0, 1)
    // status -1: not assigned, 0: fine group 1: coarse group
    // status[i] = 0 if #S^T_i = 0 or -1
    gko::array<int> status(exec, this->get_size()[0]);
    gko::array<int> new_status(exec, this->get_size()[0]);
    auto status_ptr = status.get_data();
    auto new_status_ptr = new_status.get_data();
    exec->run(pmis::make_initialize_weight_and_status(
        transpose_strong_dep.get(), weight_.get_data(), status_ptr));
    size_type num_not_assigned = 0;

    exec->run(
        pmis::make_count(this->get_size()[0], status_ptr, &num_not_assigned));
    while (num_not_assigned != 0) {
        exec->run(pmis::make_classify(weight_.get_const_data(),
                                      strong_dep.get(), status_ptr,
                                      new_status_ptr));
        size_type new_num = 0;
        exec->run(
            pmis::make_count(this->get_size()[0], new_status_ptr, &new_num));
        GKO_THROW_IF_INVALID(new_num != num_not_assigned,
                             "no progress in Pmis");
        num_not_assigned = new_num;
        std::swap(new_status_ptr, status_ptr);
    }
    // finish classify points to fine and coarse group.
    array<IndexType> prolong_row_ptrs(exec, pmis_op->get_size()[0] + 1);
    exec->run(pmis::make_direct_interpolation_row_count(
        strong_dep.get(), status_ptr, prolong_row_ptrs.get_data()));

    // coarse_map[i] gives the coarse index from i if i will appear in coarse
    // grid. if i is not in coarse grid, coarse_map[i] has no meaning; In
    // theory, we can reuse status_ptr as coarse_map. We iterate the classify
    // process on status_ptr, so keeping that it in int not IndexType might give
    // some performance benefit.
    array<IndexType> coarse_map(exec, pmis_op->get_size()[0] + 1);
    static_assert(
        kernels::pmis::coarse == 1 && kernels::pmis::fine == 0,
        "we perform prefix sum directly by having fine == 0 and coarse == 1");
    exec->run(pmis::make_convert_precision(pmis_op->get_size()[0], status_ptr,
                                           coarse_map.get_data()));
    exec->run(pmis::make_prefix_sum_nonnegative(coarse_map.get_data(),
                                                this->get_size()[0] + 1));
    auto num_coarse =
        static_cast<size_type>(get_element(coarse_map, this->get_size()[0]));

    // the following implements direct interpolation, c is coarse_map, which map
    // the fine grid index k to coarse grid index c[k] if k will appear in
    // coarse grid. Construct interpolation W which contain value w_{i, c[i]} if
    // i will appear in coarse grid or w_{i, c[k]} if S_ik exists and k will
    // appear in coarse grid.

    exec->run(pmis::make_prefix_sum_nonnegative(prolong_row_ptrs.get_data(),
                                                this->get_size()[0] + 1));
    IndexType prolong_nnz = get_element(prolong_row_ptrs, this->get_size()[0]);
    array<IndexType> prolong_col_idxs(exec, prolong_nnz);
    array<ValueType> prolong_values(exec, prolong_nnz);

    // alpha_i = sum(a_ij which a_ij < 0) / sum(a_ik which s_ik exist, a_ik < 0,
    // k is coarse point)
    // beta_i = sum(a_ij which a_ij > 0) / sum(a_ik which
    // s_ik exist, a_ik > 0, k is coarse point)
    // If there is no entry for alpha_i or beta_i, we consider it is empty.
    // If the following formula uses empty alpha_i or beta_i, the weight will
    // not have that entry. finish weight construction w_{i, c[k]} = {-alpha_i
    // if a_ik is negative or -beta_i if it is positive} * a_ik/a_ii if i is
    // fine point. If i is coarse point w_{i, c[i]} = 1
    exec->run(pmis::make_direct_interpolation_fill(
        pmis_op.get(), row_maxabs.get_const_data(),
        this->get_parameters().strength_threshold, coarse_map.get_const_data(),
        prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
        prolong_values.get_data()));

    std::shared_ptr<csr_type> prolongation;
    const auto trunc_factor = this->get_parameters().truncation_factor;
    if (trunc_factor > zero<real_type>()) {
        // Build a temporary CSR from the full prolongation arrays, then
        // compact it by dropping entries below trunc_factor * row_max.
        auto tmp_prolong = csr_type::create(
            exec, dim<2>{pmis_op->get_size()[0], num_coarse},
            std::move(prolong_values), std::move(prolong_col_idxs),
            std::move(prolong_row_ptrs));
        const auto nrows = pmis_op->get_size()[0];
        array<IndexType> trunc_row_ptrs(exec, nrows + 1);
        exec->run(pmis::make_truncate_prolongation_count(
            tmp_prolong.get(), trunc_factor, trunc_row_ptrs.get_data()));
        exec->run(pmis::make_prefix_sum_nonnegative(trunc_row_ptrs.get_data(),
                                                    nrows + 1));
        IndexType trunc_nnz = get_element(trunc_row_ptrs, nrows);
        array<IndexType> trunc_cols(exec, trunc_nnz);
        array<ValueType> trunc_vals(exec, trunc_nnz);
        exec->run(pmis::make_truncate_prolongation_fill(
            tmp_prolong.get(), trunc_factor, trunc_row_ptrs.get_const_data(),
            trunc_cols.get_data(), trunc_vals.get_data()));
        prolongation = share(csr_type::create(
            exec, dim<2>{nrows, num_coarse},
            std::move(trunc_vals), std::move(trunc_cols),
            std::move(trunc_row_ptrs)));
    } else {
        prolongation = share(csr_type::create(
            exec, dim<2>{pmis_op->get_size()[0], num_coarse},
            std::move(prolong_values), std::move(prolong_col_idxs),
            std::move(prolong_row_ptrs)));
    }
    auto restriction = share(prolongation->transpose());
    auto internal = matrix::Csr<ValueType, IndexType>::create(
        exec, prolongation->get_size());
    auto coarse = share(matrix::Csr<ValueType, IndexType>::create(
        exec, dim<2>{num_coarse, num_coarse}));
    pmis_op->apply(prolongation, internal);
    restriction->apply(internal, coarse);
    this->set_multigrid_level(prolongation, coarse, restriction);
}


#define GKO_DECLARE_PMIS(_vtype, _itype) class Pmis<_vtype, _itype>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS);


}  // namespace multigrid
}  // namespace gko
