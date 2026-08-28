// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <tuple>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The PMIS solver namespace.
 *
 * @ingroup pmis
 */
namespace pmis {


template <typename ValueType, typename IndexType>
void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        remove_complex<ValueType>* row_maxabs)
{
    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; row++) {
        // get the max in the row except diagonal
        auto max_abs = zero<remove_complex<ValueType>>();
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (col_idxs[idx] == row) {
                continue;
            }
            max_abs = std::max(max_abs, abs(vals[idx]));
        }
        row_maxabs[row] = max_abs;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Csr<ValueType, IndexType>* csr,
                            const remove_complex<ValueType>* row_maxabs,
                            remove_complex<ValueType> strength_threshold,
                            IndexType* sparsity_rows)
{
    const auto nrow = csr->get_size()[0];
    const auto row_ptrs = csr->get_const_row_ptrs();
    const auto col_idxs = csr->get_const_col_idxs();
    const auto vals = csr->get_const_values();

    for (IndexType row = 0; row < nrow; row++) {
        // count the number of strongest neighbor
        IndexType count = 0;
        auto max_abs = row_maxabs[row];
        if (max_abs == zero<remove_complex<ValueType>>()) {
            sparsity_rows[row] = zero<IndexType>();
            continue;
        }
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (col_idxs[idx] == row) {
                continue;
            }

            if (abs(vals[idx]) >= strength_threshold * max_abs) {
                count++;
            }
        }
        sparsity_rows[row] = count;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_dep(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        const remove_complex<ValueType>* row_maxabs,
                        remove_complex<ValueType> strength_threshold,
                        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)
{
    const auto vals = csr->get_const_values();
    for (IndexType row = 0; row < csr->get_size()[0]; row++) {
        auto s_idx = strong_dep->get_const_row_ptrs()[row];
        auto max_abs = row_maxabs[row];
        if (max_abs == zero<remove_complex<ValueType>>()) {
            continue;
        }
        for (auto idx = csr->get_const_row_ptrs()[row];
             idx < csr->get_const_row_ptrs()[row + 1]; idx++) {
            if (csr->get_const_col_idxs()[idx] == row) {
                continue;
            }
            if (abs(vals[idx]) >= strength_threshold * max_abs) {
                strong_dep->get_col_idxs()[s_idx] =
                    csr->get_const_col_idxs()[idx];
                s_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL);


template <typename ValueType>
void initialize_random_weight(std::shared_ptr<const DefaultExecutor> exec,
                              size_type num, ValueType* weight)
{
    std::default_random_engine gen(42);
    std::uniform_real_distribution<ValueType> dist(0.0, 1.0);
    for (size_type row = 0; row < num; row++) {
        weight[row] = dist(gen);
    }
}
GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(
    GKO_DECLARE_PMIS_INITIALIZE_RANDOM_WEIGHT_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_weight_and_status(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep,
    remove_complex<ValueType>* weight, int* status)
{
    // we can not use half, bfloat16 with random generator
    // generate it in double and then cast to corresponding type
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const auto nrows = static_cast<IndexType>(trans_strong_dep->get_size()[0]);
    const auto row_ptrs = trans_strong_dep->get_const_row_ptrs();

    for (size_type row = 0; row < nrows; row++) {
        weight[row] = static_cast<double>(row_ptrs[row + 1] - row_ptrs[row]);
        status[row] =
            (weight[row] == zero<ValueType>() ? kernels::pmis::fine
                                              : kernels::pmis::unassigned);
        weight[row] += static_cast<remove_complex<ValueType>>(dist(gen));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename ValueType, typename IndexType>
void classify(std::shared_ptr<const DefaultExecutor> exec,
              const remove_complex<ValueType>* weight,
              const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
              const int* status, int* new_status)
{
    const auto nrows = static_cast<IndexType>(strong_dep->get_size()[0]);
    const auto row_ptrs = strong_dep->get_const_row_ptrs();
    const auto col_idxs = strong_dep->get_const_col_idxs();

    for (IndexType row = 0; row < nrows; row++) {
        // -1 is unassigned yet
        auto ans = status[row];
        if (status[row] == kernels::pmis::unassigned) {
            // works on the strong graph
            const auto row_start = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            bool is_coarse = true;
            for (IndexType idx = row_start; idx < row_end; idx++) {
                auto col = col_idxs[idx];
                if (status[col] == -1 && weight[col] > weight[row]) {
                    is_coarse = false;
                    break;
                }
            }
            if (is_coarse) {
                ans = kernels::pmis::coarse;
            }
        }
        new_status[row] = ans;
    }
    // mark new fine point strongly influenced by the new coarse points
    for (IndexType row = 0; row < nrows; row++) {
        if (new_status[row] == kernels::pmis::unassigned) {
            for (auto idx = strong_dep->get_const_row_ptrs()[row];
                 idx < strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
                if (new_status[strong_dep->get_const_col_idxs()[idx]] ==
                    kernels::pmis::coarse) {
                    new_status[row] = kernels::pmis::fine;
                    break;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS_CLASSIFY_KERNEL);


void count(std::shared_ptr<const DefaultExecutor> exec, size_type num,
           const int* status, size_type* num_unassigned)
{
    size_type ans = 0;
    for (size_type i = 0; i < num; i++) {
        if (status[i] == kernels::pmis::unassigned) {
            ans++;
        }
    }
    *num_unassigned = ans;
}


template <typename ValueType, typename IndexType>
void direct_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const int* status, IndexType* prolong_row_ptr)
{
    for (size_type row = 0; row < strong_dep->get_size()[0]; row++) {
        IndexType num = 0;
        if (status[row] == 1) {
            prolong_row_ptr[row] = 1;
            continue;
        }
        for (auto idx = strong_dep->get_const_row_ptrs()[row];
             idx < strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
            auto col = strong_dep->get_const_col_idxs()[idx];
            if (status[col] == kernels::pmis::coarse) {
                num++;
            }
        }
        prolong_row_ptr[row] = num;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT);


template <typename ValueType, typename IndexType>
void direct_interpolation_fill(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* csr,
    const remove_complex<ValueType>* row_maxabs,
    const remove_complex<ValueType> strength_threshold,
    const IndexType* coarse_map, const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    auto csr_values = csr->get_const_values();
    auto csr_col_idxs = csr->get_const_col_idxs();
    auto csr_row_ptrs = csr->get_const_row_ptrs();
    for (size_type row = 0; row < csr->get_size()[0]; row++) {
        if (coarse_map[row] != coarse_map[row + 1]) {
            auto idx = prolong_row_ptrs[row];
            prolong_col_idxs[idx] = coarse_map[row];
            prolong_values[idx] = one<ValueType>();
            continue;
        }
        auto pos = zero<ValueType>();
        auto pos_divisor = zero<ValueType>();
        auto neg = zero<ValueType>();
        auto neg_divisor = zero<ValueType>();
        auto diag = zero<ValueType>();
        bool enable_neg = false;
        bool enable_pos = false;
        // first compute alpha/beta
        auto max_abs = row_maxabs[row];
        for (auto idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1]; idx++) {
            auto val = csr_values[idx];
            auto col = csr_col_idxs[idx];
            if (col == row) {
                diag = val;
                continue;
            }
            if (real(val) >= 0) {
                pos += val;
                if (coarse_map[col] != coarse_map[col + 1] &&
                    abs(val) >= strength_threshold * max_abs) {
                    pos_divisor += val;
                    enable_pos = true;
                }
            } else {
                neg += val;
                if (coarse_map[col] != coarse_map[col + 1] &&
                    abs(val) >= strength_threshold * max_abs) {
                    neg_divisor += val;
                    enable_neg = true;
                }
            }
        }
        pos = safe_divide(pos, pos_divisor);
        neg = safe_divide(neg, neg_divisor);
        if (!enable_neg && !enable_pos) {
            continue;
        }
        auto start = prolong_row_ptrs[row];
        for (auto idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1]; idx++) {
            auto val = csr_values[idx];
            auto col = csr_col_idxs[idx];
            if (col == row || abs(val) < strength_threshold * max_abs) {
                continue;
            }
            if (real(val) >= 0 && enable_pos &&
                coarse_map[col] != coarse_map[col + 1]) {
                prolong_col_idxs[start] = coarse_map[col];
                prolong_values[start] = -pos * val / diag;
                start++;
            }
            if (real(val) < 0 && enable_neg &&
                coarse_map[col] != coarse_map[col + 1]) {
                prolong_col_idxs[start] = coarse_map[col];
                prolong_values[start] = -neg * val / diag;
                start++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL);


template <typename ValueType, typename IndexType>
void truncate_prolongation_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* prolong,
    remove_complex<ValueType> trunc_factor,
    IndexType* row_count)
{
    const auto nrow = static_cast<IndexType>(prolong->get_size()[0]);
    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto values = prolong->get_const_values();
    for (IndexType row = 0; row < nrow; row++) {
        const auto row_start = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        if (row_end == row_start) {
            row_count[row] = 0;
            continue;
        }
        auto max_abs = zero<remove_complex<ValueType>>();
        for (auto idx = row_start; idx < row_end; idx++) {
            max_abs = std::max(max_abs, abs(values[idx]));
        }
        const auto threshold = trunc_factor * max_abs;
        IndexType count = 0;
        for (auto idx = row_start; idx < row_end; idx++) {
            if (abs(values[idx]) >= threshold) count++;
        }
        row_count[row] = count;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_COUNT_KERNEL);


template <typename ValueType, typename IndexType>
void truncate_prolongation_fill(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* prolong,
    remove_complex<ValueType> trunc_factor,
    const IndexType* new_row_ptrs,
    IndexType* new_col_idxs,
    ValueType* new_values)
{
    const auto nrow = static_cast<IndexType>(prolong->get_size()[0]);
    const auto row_ptrs = prolong->get_const_row_ptrs();
    const auto col_idxs = prolong->get_const_col_idxs();
    const auto values = prolong->get_const_values();
    for (IndexType row = 0; row < nrow; row++) {
        const auto row_start = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        if (row_end == row_start) continue;
        auto max_abs = zero<remove_complex<ValueType>>();
        for (auto idx = row_start; idx < row_end; idx++) {
            max_abs = std::max(max_abs, abs(values[idx]));
        }
        const auto threshold = trunc_factor * max_abs;
        auto out = new_row_ptrs[row];
        for (auto idx = row_start; idx < row_end; idx++) {
            if (abs(values[idx]) >= threshold) {
                new_col_idxs[out] = col_idxs[idx];
                new_values[out] = values[idx];
                out++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_FILL_KERNEL);


// ---- Distributed kernels ----

template <typename ValueType, typename IndexType>
void update_row_maxabs_off_diag(std::shared_ptr<const DefaultExecutor> exec,
                                const matrix::Csr<ValueType, IndexType>* off_diag_csr,
                                remove_complex<ValueType>* row_maxabs)
{
    const auto nrow = off_diag_csr->get_size()[0];
    const auto row_ptrs = off_diag_csr->get_const_row_ptrs();
    const auto vals = off_diag_csr->get_const_values();
    for (IndexType row = 0; row < nrow; row++) {
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            row_maxabs[row] = std::max(row_maxabs[row], abs(vals[idx]));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_UPDATE_ROW_MAXABS_OFF_DIAG_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_ghost_dep_row(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* off_diag_csr,
    const remove_complex<ValueType>* row_maxabs,
    remove_complex<ValueType> strength_threshold,
    IndexType* ghost_sparsity_rows)
{
    const auto nrow = off_diag_csr->get_size()[0];
    const auto row_ptrs = off_diag_csr->get_const_row_ptrs();
    const auto vals = off_diag_csr->get_const_values();
    for (IndexType row = 0; row < nrow; row++) {
        auto max_abs = row_maxabs[row];
        IndexType count = 0;
        if (max_abs != zero<remove_complex<ValueType>>()) {
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                if (abs(vals[idx]) >= strength_threshold * max_abs) {
                    count++;
                }
            }
        }
        ghost_sparsity_rows[row] = count;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_GHOST_DEP_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void compute_strong_ghost_dep(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* off_diag_csr,
    const remove_complex<ValueType>* row_maxabs,
    remove_complex<ValueType> strength_threshold,
    matrix::SparsityCsr<ValueType, IndexType>* strong_ghost_dep)
{
    const auto vals = off_diag_csr->get_const_values();
    const auto col_idxs = off_diag_csr->get_const_col_idxs();
    const auto row_ptrs = off_diag_csr->get_const_row_ptrs();
    for (IndexType row = 0; row < static_cast<IndexType>(off_diag_csr->get_size()[0]); row++) {
        auto max_abs = row_maxabs[row];
        if (max_abs == zero<remove_complex<ValueType>>()) continue;
        auto s_idx = strong_ghost_dep->get_const_row_ptrs()[row];
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
            if (abs(vals[idx]) >= strength_threshold * max_abs) {
                strong_ghost_dep->get_col_idxs()[s_idx++] = col_idxs[idx];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_GHOST_DEP_KERNEL);


template <typename ValueType, typename IndexType>
void classify_with_ghosts(
    std::shared_ptr<const DefaultExecutor> exec,
    const remove_complex<ValueType>* weight,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_ghost_dep,
    const remove_complex<ValueType>* ghost_weight,
    const int* ghost_status,
    const int* status, int* new_status)
{
    const auto nrows = static_cast<IndexType>(strong_dep->get_size()[0]);
    // Pass 1: mark coarse — node is coarse if it has the highest weight among
    // all strongly connected unassigned neighbors (local and ghost).
    for (IndexType row = 0; row < nrows; row++) {
        auto ans = status[row];
        if (status[row] == kernels::pmis::unassigned) {
            bool is_coarse = true;
            for (auto idx = strong_dep->get_const_row_ptrs()[row];
                 idx < strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
                auto col = strong_dep->get_const_col_idxs()[idx];
                if (status[col] == kernels::pmis::unassigned &&
                    weight[col] > weight[row]) {
                    is_coarse = false;
                    break;
                }
            }
            if (is_coarse) {
                for (auto idx = strong_ghost_dep->get_const_row_ptrs()[row];
                     idx < strong_ghost_dep->get_const_row_ptrs()[row + 1]; idx++) {
                    auto gcol = strong_ghost_dep->get_const_col_idxs()[idx];
                    if (ghost_status[gcol] == kernels::pmis::unassigned &&
                        ghost_weight[gcol] > weight[row]) {
                        is_coarse = false;
                        break;
                    }
                }
            }
            if (is_coarse) ans = kernels::pmis::coarse;
        }
        new_status[row] = ans;
    }
    // Pass 2: mark fine — any unassigned node strongly influenced by a new coarse.
    for (IndexType row = 0; row < nrows; row++) {
        if (new_status[row] != kernels::pmis::unassigned) continue;
        for (auto idx = strong_dep->get_const_row_ptrs()[row];
             idx < strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
            if (new_status[strong_dep->get_const_col_idxs()[idx]] ==
                kernels::pmis::coarse) {
                new_status[row] = kernels::pmis::fine;
                break;
            }
        }
        if (new_status[row] != kernels::pmis::unassigned) continue;
        // also check ghost coarse neighbors (using communicated prev-iter status)
        for (auto idx = strong_ghost_dep->get_const_row_ptrs()[row];
             idx < strong_ghost_dep->get_const_row_ptrs()[row + 1]; idx++) {
            if (ghost_status[strong_ghost_dep->get_const_col_idxs()[idx]] ==
                kernels::pmis::coarse) {
                new_status[row] = kernels::pmis::fine;
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_CLASSIFY_WITH_GHOSTS_KERNEL);


template <typename ValueType, typename IndexType>
void direct_interpolation_row_count_dist(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_ghost_dep,
    const int* status, const int* ghost_status,
    IndexType* diag_row_ptr, IndexType* off_diag_row_ptr)
{
    for (size_type row = 0; row < strong_dep->get_size()[0]; row++) {
        if (status[row] == kernels::pmis::coarse) {
            diag_row_ptr[row] = 1;
            off_diag_row_ptr[row] = 0;
            continue;
        }
        IndexType diag_cnt = 0, off_diag_cnt = 0;
        for (auto idx = strong_dep->get_const_row_ptrs()[row];
             idx < strong_dep->get_const_row_ptrs()[row + 1]; idx++) {
            if (status[strong_dep->get_const_col_idxs()[idx]] ==
                kernels::pmis::coarse) {
                diag_cnt++;
            }
        }
        for (auto idx = strong_ghost_dep->get_const_row_ptrs()[row];
             idx < strong_ghost_dep->get_const_row_ptrs()[row + 1]; idx++) {
            if (ghost_status[strong_ghost_dep->get_const_col_idxs()[idx]] ==
                kernels::pmis::coarse) {
                off_diag_cnt++;
            }
        }
        diag_row_ptr[row] = diag_cnt;
        off_diag_row_ptr[row] = off_diag_cnt;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_DIRECT_INTERP_ROW_COUNT_DIST_KERNEL);


// Helper: compute alpha (negative weight scaling) and beta (positive weight scaling)
// for row `row` of the distributed CSR (both diag and off-diag blocks).
// ghost_coarse_status[k] == pmis::coarse means ghost k is a coarse node.
template <typename ValueType, typename IndexType>
static void compute_alpha_beta(
    size_type row,
    const matrix::Csr<ValueType, IndexType>* diag_csr,
    const matrix::Csr<ValueType, IndexType>* off_diag_csr,
    const remove_complex<ValueType>* row_maxabs,
    remove_complex<ValueType> strength_threshold,
    const IndexType* coarse_map,
    const int* ghost_coarse_status,
    ValueType& diag_val, ValueType& pos, ValueType& pos_div,
    ValueType& neg, ValueType& neg_div,
    bool& enable_pos, bool& enable_neg)
{
    auto max_abs = row_maxabs[row];
    auto zero_v = zero<ValueType>();
    diag_val = zero_v; pos = zero_v; pos_div = zero_v;
    neg = zero_v; neg_div = zero_v;
    enable_pos = false; enable_neg = false;

    // Diagonal block
    for (auto idx = diag_csr->get_const_row_ptrs()[row];
         idx < diag_csr->get_const_row_ptrs()[row + 1]; idx++) {
        auto val = diag_csr->get_const_values()[idx];
        auto col = diag_csr->get_const_col_idxs()[idx];
        if (col == static_cast<IndexType>(row)) { diag_val = val; continue; }
        bool is_local_coarse = coarse_map[col] != coarse_map[col + 1];
        bool is_strong = abs(val) >= strength_threshold * max_abs;
        if (real(val) >= real(zero_v)) {
            pos += val;
            if (is_local_coarse && is_strong) { pos_div += val; enable_pos = true; }
        } else {
            neg += val;
            if (is_local_coarse && is_strong) { neg_div += val; enable_neg = true; }
        }
    }
    // Off-diagonal block
    if (off_diag_csr) {
        for (auto idx = off_diag_csr->get_const_row_ptrs()[row];
             idx < off_diag_csr->get_const_row_ptrs()[row + 1]; idx++) {
            auto val = off_diag_csr->get_const_values()[idx];
            auto gcol = off_diag_csr->get_const_col_idxs()[idx];
            bool is_ghost_coarse = (ghost_coarse_status[gcol] == kernels::pmis::coarse);
            bool is_strong = abs(val) >= strength_threshold * max_abs;
            if (real(val) >= real(zero_v)) {
                pos += val;
                if (is_ghost_coarse && is_strong) { pos_div += val; enable_pos = true; }
            } else {
                neg += val;
                if (is_ghost_coarse && is_strong) { neg_div += val; enable_neg = true; }
            }
        }
    }
    pos = safe_divide(pos, pos_div);
    neg = safe_divide(neg, neg_div);
}


template <typename ValueType, typename IndexType>
void direct_interpolation_fill_diag(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* diag_csr,
    const matrix::Csr<ValueType, IndexType>* off_diag_csr,
    const remove_complex<ValueType>* row_maxabs,
    remove_complex<ValueType> strength_threshold,
    const IndexType* coarse_map,
    const int* ghost_coarse_status,
    const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    auto zero_v = zero<ValueType>();
    for (size_type row = 0; row < diag_csr->get_size()[0]; row++) {
        // Coarse node: identity
        if (coarse_map[row] != coarse_map[row + 1]) {
            prolong_col_idxs[prolong_row_ptrs[row]] = coarse_map[row];
            prolong_values[prolong_row_ptrs[row]] = one<ValueType>();
            continue;
        }
        ValueType diag_val, pos, pos_div, neg, neg_div;
        bool enable_pos, enable_neg;
        compute_alpha_beta(row, diag_csr, off_diag_csr, row_maxabs,
                           strength_threshold, coarse_map, ghost_coarse_status,
                           diag_val, pos, pos_div, neg, neg_div,
                           enable_pos, enable_neg);
        if (!enable_pos && !enable_neg) continue;
        auto max_abs = row_maxabs[row];
        auto p_idx = prolong_row_ptrs[row];
        for (auto idx = diag_csr->get_const_row_ptrs()[row];
             idx < diag_csr->get_const_row_ptrs()[row + 1]; idx++) {
            auto val = diag_csr->get_const_values()[idx];
            auto col = diag_csr->get_const_col_idxs()[idx];
            if (col == static_cast<IndexType>(row) ||
                abs(val) < strength_threshold * max_abs) continue;
            if (coarse_map[col] == coarse_map[col + 1]) continue;
            if (real(val) >= real(zero_v) && enable_pos) {
                prolong_col_idxs[p_idx] = coarse_map[col];
                prolong_values[p_idx++] = -pos * val / diag_val;
            } else if (real(val) < real(zero_v) && enable_neg) {
                prolong_col_idxs[p_idx] = coarse_map[col];
                prolong_values[p_idx++] = -neg * val / diag_val;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_DIRECT_INTERP_FILL_DIAG_KERNEL);


template <typename ValueType, typename IndexType>
void direct_interpolation_fill_off_diag(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* diag_csr,
    const matrix::Csr<ValueType, IndexType>* off_diag_csr,
    const remove_complex<ValueType>* row_maxabs,
    remove_complex<ValueType> strength_threshold,
    const IndexType* coarse_map,
    const IndexType* ghost_coarse_local_idx,
    const int* ghost_coarse_status,
    const IndexType* prolong_row_ptrs,
    IndexType* prolong_col_idxs, ValueType* prolong_values)
{
    auto zero_v = zero<ValueType>();
    for (size_type row = 0; row < diag_csr->get_size()[0]; row++) {
        if (coarse_map[row] != coarse_map[row + 1]) continue; // coarse: no off-diag entries
        ValueType diag_val, pos, pos_div, neg, neg_div;
        bool enable_pos, enable_neg;
        compute_alpha_beta(row, diag_csr, off_diag_csr, row_maxabs,
                           strength_threshold, coarse_map, ghost_coarse_status,
                           diag_val, pos, pos_div, neg, neg_div,
                           enable_pos, enable_neg);
        if (!enable_pos && !enable_neg) continue;
        auto max_abs = row_maxabs[row];
        auto p_idx = prolong_row_ptrs[row];
        for (auto idx = off_diag_csr->get_const_row_ptrs()[row];
             idx < off_diag_csr->get_const_row_ptrs()[row + 1]; idx++) {
            auto val = off_diag_csr->get_const_values()[idx];
            auto gcol = off_diag_csr->get_const_col_idxs()[idx];
            if (abs(val) < strength_threshold * max_abs) continue;
            if (ghost_coarse_status[gcol] != kernels::pmis::coarse) continue;
            if (real(val) >= real(zero_v) && enable_pos) {
                prolong_col_idxs[p_idx] = ghost_coarse_local_idx[gcol];
                prolong_values[p_idx++] = -pos * val / diag_val;
            } else if (real(val) < real(zero_v) && enable_neg) {
                prolong_col_idxs[p_idx] = ghost_coarse_local_idx[gcol];
                prolong_values[p_idx++] = -neg * val / diag_val;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_DIRECT_INTERP_FILL_OFF_DIAG_KERNEL);


}  // namespace pmis
}  // namespace reference
}  // namespace kernels
}  // namespace gko
