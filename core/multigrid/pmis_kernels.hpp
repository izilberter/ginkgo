// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace pmis {


constexpr int coarse = 1;
constexpr int fine = 0;
constexpr int unassigned = -1;


#define GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL(ValueType, IndexType)  \
    void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,  \
                            const matrix::Csr<ValueType, IndexType>* csr, \
                            remove_complex<ValueType>* row_maxabs)


#define GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL(ValueType, IndexType)  \
    void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,  \
                                const matrix::Csr<ValueType, IndexType>* csr, \
                                const remove_complex<ValueType>* row_maxabs,  \
                                remove_complex<ValueType> strength_threshold, \
                                IndexType* sparsity_rows)

#define GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL(ValueType, IndexType) \
    void compute_strong_dep(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* csr,                    \
        const remove_complex<ValueType>* row_maxabs,                     \
        remove_complex<ValueType> strength_threshold,                    \
        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)

#define GKO_DECLARE_PMIS_INITIALIZE_RANDOM_WEIGHT_KERNEL(ValueType)            \
    void initialize_random_weight(std::shared_ptr<const DefaultExecutor> exec, \
                                  size_type num, ValueType* weight)

#define GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL(ValueType,    \
                                                             IndexType)    \
    void initialize_weight_and_status(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep, \
        remove_complex<ValueType>* weight, int* status)

#define GKO_DECLARE_PMIS_CLASSIFY_KERNEL(ValueType, IndexType)                 \
    void classify(std::shared_ptr<const DefaultExecutor> exec,                 \
                  const remove_complex<ValueType>* weight,                     \
                  const matrix::SparsityCsr<ValueType, IndexType>* strong_dep, \
                  const int* status, int* new_status)

#define GKO_DECLARE_COUNT_KERNEL                                           \
    void count(std::shared_ptr<const DefaultExecutor> exec, size_type num, \
               const int* status, size_type* num_unassigned)

#define GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT(ValueType, IndexType) \
    void direct_interpolation_row_count(                                 \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,     \
        const int* status, IndexType* prolong_row_ptr)

#define GKO_DECLARE_DIRECT_INTERPOLATION_FILL(ValueType, IndexType)     \
    void direct_interpolation_fill(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType>* csr,                   \
        const remove_complex<ValueType>* row_maxabs,                    \
        const remove_complex<ValueType> strength_threshold,             \
        const IndexType* coarse_map, const IndexType* prolong_row_ptrs, \
        IndexType* prolong_col_idxs, ValueType* prolong_values)


// ---- Distributed (MPI) kernels ----

// Update row_maxabs by considering off-diagonal block entries (no diagonal to skip).
#define GKO_DECLARE_PMIS_UPDATE_ROW_MAXABS_OFF_DIAG_KERNEL(ValueType, IndexType) \
    void update_row_maxabs_off_diag(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                              \
        const matrix::Csr<ValueType, IndexType>* off_diag_csr,                   \
        remove_complex<ValueType>* row_maxabs)

// Count strong ghost (off-diagonal) neighbors per row.
#define GKO_DECLARE_PMIS_COMPUTE_STRONG_GHOST_DEP_ROW_KERNEL(ValueType, IndexType) \
    void compute_strong_ghost_dep_row(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                                \
        const matrix::Csr<ValueType, IndexType>* off_diag_csr,                     \
        const remove_complex<ValueType>* row_maxabs,                                \
        remove_complex<ValueType> strength_threshold,                               \
        IndexType* ghost_sparsity_rows)

// Fill ghost strong dep col indices (local ghost indices from off-diagonal CSR).
#define GKO_DECLARE_PMIS_COMPUTE_STRONG_GHOST_DEP_KERNEL(ValueType, IndexType) \
    void compute_strong_ghost_dep(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                            \
        const matrix::Csr<ValueType, IndexType>* off_diag_csr,                  \
        const remove_complex<ValueType>* row_maxabs,                             \
        remove_complex<ValueType> strength_threshold,                            \
        matrix::SparsityCsr<ValueType, IndexType>* strong_ghost_dep)

// Classify with ghost node weights and statuses (distributed version).
// ghost_weight[k] and ghost_status[k] are the communicated values for ghost node k.
// strong_ghost_dep has col indices that are local ghost indices (0..nghost-1).
#define GKO_DECLARE_PMIS_CLASSIFY_WITH_GHOSTS_KERNEL(ValueType, IndexType)            \
    void classify_with_ghosts(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                                   \
        const remove_complex<ValueType>* weight,                                       \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,                  \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_ghost_dep,            \
        const remove_complex<ValueType>* ghost_weight,                                 \
        const int* ghost_status,                                                       \
        const int* status, int* new_status)

// Count diag/off-diag prolongation entries per row in the distributed case.
#define GKO_DECLARE_PMIS_DIRECT_INTERP_ROW_COUNT_DIST_KERNEL(ValueType, IndexType) \
    void direct_interpolation_row_count_dist(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                                \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,               \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_ghost_dep,         \
        const int* status, const int* ghost_status,                                 \
        IndexType* diag_row_ptr, IndexType* off_diag_row_ptr)

// Fill the diag block of the prolongation (local fine -> local coarse).
// coarse_map[i..i+1] encodes whether i is coarse and its local coarse index.
// ghost_coarse_status[k] == pmis::coarse if ghost k is a coarse node.
#define GKO_DECLARE_PMIS_DIRECT_INTERP_FILL_DIAG_KERNEL(ValueType, IndexType)   \
    void direct_interpolation_fill_diag(                                         \
        std::shared_ptr<const DefaultExecutor> exec,                             \
        const matrix::Csr<ValueType, IndexType>* diag_csr,                       \
        const matrix::Csr<ValueType, IndexType>* off_diag_csr,                   \
        const remove_complex<ValueType>* row_maxabs,                              \
        remove_complex<ValueType> strength_threshold,                             \
        const IndexType* coarse_map,                                              \
        const int* ghost_coarse_status,                                           \
        const IndexType* prolong_row_ptrs,                                        \
        IndexType* prolong_col_idxs, ValueType* prolong_values)

// Truncate prolongation rows: for each row keep only entries whose absolute
// value is >= trunc_factor * row_max.  row_count[i] receives the surviving
// count (always >= 1 when trunc_factor < 1 and the row is non-empty).
#define GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_COUNT_KERNEL(ValueType, IndexType) \
    void truncate_prolongation_count(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                               \
        const matrix::Csr<ValueType, IndexType>* prolong,                          \
        remove_complex<ValueType> trunc_factor,                                    \
        IndexType* row_count)

// Fill compacted prolongation arrays from the (already-computed) full
// prolongation, keeping only entries that survive the truncation threshold.
#define GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_FILL_KERNEL(ValueType, IndexType) \
    void truncate_prolongation_fill(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                              \
        const matrix::Csr<ValueType, IndexType>* prolong,                         \
        remove_complex<ValueType> trunc_factor,                                   \
        const IndexType* new_row_ptrs,                                            \
        IndexType* new_col_idxs,                                                  \
        ValueType* new_values)

// Fill the off-diag block of the prolongation (local fine -> ghost coarse).
// ghost_coarse_local_idx[k] == local off-diag coarse index for ghost node k.
#define GKO_DECLARE_PMIS_DIRECT_INTERP_FILL_OFF_DIAG_KERNEL(ValueType, IndexType) \
    void direct_interpolation_fill_off_diag(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                               \
        const matrix::Csr<ValueType, IndexType>* diag_csr,                         \
        const matrix::Csr<ValueType, IndexType>* off_diag_csr,                     \
        const remove_complex<ValueType>* row_maxabs,                                \
        remove_complex<ValueType> strength_threshold,                               \
        const IndexType* coarse_map,                                                \
        const IndexType* ghost_coarse_local_idx,                                    \
        const int* ghost_coarse_status,                                             \
        const IndexType* prolong_row_ptrs,                                          \
        IndexType* prolong_col_idxs, ValueType* prolong_values)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                                \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL(ValueType, IndexType);               \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL(ValueType, IndexType);               \
    template <typename ValueType>                                                   \
    GKO_DECLARE_PMIS_INITIALIZE_RANDOM_WEIGHT_KERNEL(ValueType);                    \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_CLASSIFY_KERNEL(ValueType, IndexType);                         \
    GKO_DECLARE_COUNT_KERNEL;                                                       \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT(ValueType, IndexType);               \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL(ValueType, IndexType);                    \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_UPDATE_ROW_MAXABS_OFF_DIAG_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_GHOST_DEP_ROW_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_GHOST_DEP_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_CLASSIFY_WITH_GHOSTS_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_DIRECT_INTERP_ROW_COUNT_DIST_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_DIRECT_INTERP_FILL_DIAG_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_DIRECT_INTERP_FILL_OFF_DIAG_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_COUNT_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                               \
    GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_FILL_KERNEL(ValueType, IndexType)


}  // namespace pmis


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(pmis, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_
