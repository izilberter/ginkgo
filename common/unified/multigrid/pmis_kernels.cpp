// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <random>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Pmis namespace.
 *
 * @ingroup pmis
 */
namespace pmis {


// the number of threads working on the same row
constexpr int width = 32;


template <typename ValueType, typename IndexType>
void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* csr,
                        remove_complex<ValueType>* row_maxabs)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto row_ptrs, auto col_idxs,
                      auto values) {
            auto maxabs = zero(abs(values[0]));
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (row == col_idxs[idx]) {
                    continue;
                }
                maxabs = max(maxabs, abs(values[idx]));
            }
            return maxabs;
        },
        GKO_KERNEL_REDUCE_MAX(remove_complex<ValueType>), row_maxabs, 1,
        dim<2>{csr->get_size()[0], width}, csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values());
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
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto row_maxabs,
                      auto strength_threshold, auto row_ptrs, auto col_idxs,
                      auto values) {
            auto max_abs = row_maxabs[row];
            auto count = zero<IndexType>();
            if (max_abs == zero(max_abs)) {
                return count;
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (row == col_idxs[idx]) {
                    continue;
                }
                if (abs(values[idx]) >= strength_threshold * max_abs) {
                    count++;
                }
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), sparsity_rows, 1,
        dim<2>{csr->get_size()[0], width}, row_maxabs, strength_threshold,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values());
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
    // we handle this by one thread per row. It might get improved if we use a
    // warp with popcount and prefix for a row.
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto strength_threshold,
                      auto row_ptrs, auto col_idxs, auto values,
                      auto dep_row_ptrs, auto dep_col_idxs) {
            auto max_abs = row_maxabs[row];
            if (max_abs == zero(max_abs)) {
                return;
            }
            auto d_idx = dep_row_ptrs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                const auto col = col_idxs[idx];
                if (row == col) {
                    continue;
                }
                if (abs(values[idx]) >= strength_threshold * max_abs) {
                    dep_col_idxs[d_idx] = col;
                    d_idx++;
                }
            }
        },
        csr->get_size()[0], row_maxabs, strength_threshold,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), strong_dep->get_const_row_ptrs(),
        strong_dep->get_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_weight_and_status(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep,
    remove_complex<ValueType>* weight, int* status)
{
    auto num = trans_strong_dep->get_size()[0];
    array<float> random(exec, num);
    initialize_random_weight(exec, num, random.get_data());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto random, auto weight,
                      auto status) {
            using type = device_type<remove_complex<ValueType>>;
            auto w = static_cast<float>(row_ptrs[row + 1] - row_ptrs[row]);
            status[row] =
                (w == 0.0f ? kernels::pmis::fine : kernels::pmis::unassigned);
            weight[row] = static_cast<type>(random[row] + w);
        },
        num, trans_strong_dep->get_const_row_ptrs(), random.get_const_data(),
        weight, status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL);


template <typename ValueType, typename IndexType>
void classify(std::shared_ptr<const DefaultExecutor> exec,
              const remove_complex<ValueType>* weight,
              const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
              const int* status, int* new_status)
{
    static_assert(kernels::pmis::unassigned < kernels::pmis::coarse,
                  "we use min reduction to mark local maximum as coarse");
    // mark coarse point
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto status, auto weight,
                      auto row_ptrs, auto col_idxs) {
            auto ans = status[row];
            if (ans != kernels::pmis::unassigned) {
                return ans;
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                auto col = col_idxs[idx];
                if (status[col] == kernels::pmis::unassigned &&
                    weight[col] > weight[row]) {
                    return kernels::pmis::unassigned;
                }
            }
            return kernels::pmis::coarse;
        },
        [] GKO_KERNEL(auto a, auto b) { return a < b ? a : b; } /* minimun */,
        [] GKO_KERNEL(auto a) { return a; }, int{1}, new_status, 1,
        dim<2>{strong_dep->get_size()[0], width}, status, weight,
        strong_dep->get_const_row_ptrs(), strong_dep->get_const_col_idxs());
    // mark new fine point strongly influenced by the new coarse points
    // TODO: using warp vote function if implement in native way.
    static_assert(kernels::pmis::fine > kernels::pmis::unassigned,
                  "we use max reduction to mark new fine by any strong coarse");
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto new_status, auto row_ptrs,
                      auto col_idxs) {
            if (new_status[row] != kernels::pmis::unassigned) {
                return new_status[row];
            }
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                // we will only update new_status from -1 to 0 or keep -1, so
                // grabbing this value is fine no matter if it is updated or
                // not.
                if (new_status[col_idxs[idx]] == kernels::pmis::coarse) {
                    return kernels::pmis::fine;
                }
            }
            return kernels::pmis::unassigned;
        },
        [] GKO_KERNEL(auto a, auto b) { return a > b ? a : b; } /* maximum */,
        [] GKO_KERNEL(auto a) { return a; }, int{-1}, new_status, 1,
        dim<2>{strong_dep->get_size()[0], width}, new_status,
        strong_dep->get_const_row_ptrs(), strong_dep->get_const_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PMIS_CLASSIFY_KERNEL);


void count(std::shared_ptr<const DefaultExecutor> exec, size_type num,
           const int* status, size_type* num_unassigned)
{
    array<size_type> d_result(exec, 1);
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto status) {
            return static_cast<size_type>(status[i] ==
                                          kernels::pmis::unassigned);
        },
        GKO_KERNEL_REDUCE_SUM(size_type), d_result.get_data(), num, status);
    *num_unassigned = get_element(d_result, 0);
}


template <typename ValueType, typename IndexType>
void direct_interpolation_row_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,
    const int* status, IndexType* prolong_row_ptr)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto status, auto row_ptrs,
                      auto col_idxs) {
            if (status[row] == kernels::pmis::coarse) {
                return tid == 0 ? one<IndexType>() : zero<IndexType>();
            }
            auto count = zero<IndexType>();
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (status[col_idxs[idx]] == kernels::pmis::coarse) {
                    count++;
                }
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), prolong_row_ptr, 1,
        dim<2>{strong_dep->get_size()[0], width}, status,
        strong_dep->get_const_row_ptrs(), strong_dep->get_const_col_idxs());
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
    // currently use one thread per row. It might get improved by using a warp
    // for row with prefix and popcount
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto strength_threshold,
                      auto coarse_map, auto row_ptrs, auto col_idxs,
                      auto values, auto prolong_row_ptrs, auto prolong_col_idxs,
                      auto prolong_values) {
            if (coarse_map[row] != coarse_map[row + 1]) {
                auto idx = prolong_row_ptrs[row];
                prolong_col_idxs[idx] = coarse_map[row];
                prolong_values[idx] = one(prolong_values[idx]);
                return;
            }
            auto pos = zero(values[0]);
            auto zeroval = zero(values[0]);
            auto pos_divisor = zero(values[0]);
            auto neg = zero(values[0]);
            auto neg_divisor = zero(values[0]);
            auto diag = zero(values[0]);
            bool enable_neg = false;
            bool enable_pos = false;
            // first compute alpha/beta
            auto max_abs = row_maxabs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                auto val = values[idx];
                auto col = col_idxs[idx];
                if (col == row) {
                    diag = val;
                    continue;
                }
                if (real(val) >= real(zeroval)) {
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
                return;
            }

            auto p_idx = prolong_row_ptrs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                auto val = values[idx];
                auto col = col_idxs[idx];
                if (col == row || abs(val) < strength_threshold * max_abs) {
                    continue;
                }
                if (real(val) >= real(zeroval) && enable_pos &&
                    coarse_map[col] != coarse_map[col + 1]) {
                    prolong_col_idxs[p_idx] = coarse_map[col];
                    prolong_values[p_idx] = -pos * val / diag;
                    p_idx++;
                }
                if (real(val) < real(zeroval) && enable_neg &&
                    coarse_map[col] != coarse_map[col + 1]) {
                    prolong_col_idxs[p_idx] = coarse_map[col];
                    prolong_values[p_idx] = -neg * val / diag;
                    p_idx++;
                }
            }
        },
        csr->get_size()[0], row_maxabs, strength_threshold, coarse_map,
        csr->get_const_row_ptrs(), csr->get_const_col_idxs(),
        csr->get_const_values(), prolong_row_ptrs, prolong_col_idxs,
        prolong_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL);


// ---- Distributed kernels ----

template <typename ValueType, typename IndexType>
void update_row_maxabs_off_diag(std::shared_ptr<const DefaultExecutor> exec,
                                const matrix::Csr<ValueType, IndexType>* off_diag_csr,
                                remove_complex<ValueType>* row_maxabs)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto row_ptrs, auto values,
                      auto prev_max) {
            auto maxabs = prev_max[row];
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                maxabs = max(maxabs, abs(values[idx]));
            }
            return maxabs;
        },
        GKO_KERNEL_REDUCE_MAX(remove_complex<ValueType>), row_maxabs, 1,
        dim<2>{off_diag_csr->get_size()[0], width},
        off_diag_csr->get_const_row_ptrs(), off_diag_csr->get_const_values(),
        row_maxabs);
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
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto row_maxabs,
                      auto strength_threshold, auto row_ptrs, auto values) {
            auto max_abs = row_maxabs[row];
            auto count = zero<IndexType>();
            if (max_abs == zero(max_abs)) return count;
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (abs(values[idx]) >= strength_threshold * max_abs) count++;
            }
            return count;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), ghost_sparsity_rows, 1,
        dim<2>{off_diag_csr->get_size()[0], width}, row_maxabs,
        strength_threshold, off_diag_csr->get_const_row_ptrs(),
        off_diag_csr->get_const_values());
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
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto strength_threshold,
                      auto row_ptrs, auto col_idxs, auto values,
                      auto dep_row_ptrs, auto dep_col_idxs) {
            auto max_abs = row_maxabs[row];
            if (max_abs == zero(max_abs)) return;
            auto d_idx = dep_row_ptrs[row];
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                if (abs(values[idx]) >= strength_threshold * max_abs) {
                    dep_col_idxs[d_idx++] = col_idxs[idx];
                }
            }
        },
        off_diag_csr->get_size()[0], row_maxabs, strength_threshold,
        off_diag_csr->get_const_row_ptrs(), off_diag_csr->get_const_col_idxs(),
        off_diag_csr->get_const_values(), strong_ghost_dep->get_const_row_ptrs(),
        strong_ghost_dep->get_col_idxs());
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
    static_assert(kernels::pmis::unassigned < kernels::pmis::coarse,
                  "min reduction used to mark local max as coarse");
    // Pass 1: mark coarse
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto status, auto weight,
                      auto row_ptrs, auto col_idxs,
                      auto ghost_row_ptrs, auto ghost_col_idxs,
                      auto ghost_weight, auto ghost_status) {
            auto ans = status[row];
            if (ans != kernels::pmis::unassigned) return ans;
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                auto col = col_idxs[idx];
                if (status[col] == kernels::pmis::unassigned &&
                    weight[col] > weight[row]) {
                    return kernels::pmis::unassigned;
                }
            }
            for (auto idx = tid + ghost_row_ptrs[row]; idx < ghost_row_ptrs[row + 1];
                 idx += width) {
                auto gcol = ghost_col_idxs[idx];
                if (ghost_status[gcol] == kernels::pmis::unassigned &&
                    ghost_weight[gcol] > weight[row]) {
                    return kernels::pmis::unassigned;
                }
            }
            return kernels::pmis::coarse;
        },
        [] GKO_KERNEL(auto a, auto b) { return a < b ? a : b; },
        [] GKO_KERNEL(auto a) { return a; }, int{1}, new_status, 1,
        dim<2>{strong_dep->get_size()[0], width}, status, weight,
        strong_dep->get_const_row_ptrs(), strong_dep->get_const_col_idxs(),
        strong_ghost_dep->get_const_row_ptrs(),
        strong_ghost_dep->get_const_col_idxs(), ghost_weight, ghost_status);

    // Pass 2: mark fine by coarse influence (local + ghost, using communicated status)
    static_assert(kernels::pmis::fine > kernels::pmis::unassigned,
                  "max reduction used to mark new fine by any strong coarse");
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto row, auto tid, auto new_status, auto row_ptrs,
                      auto col_idxs, auto ghost_row_ptrs, auto ghost_col_idxs,
                      auto ghost_status) {
            if (new_status[row] != kernels::pmis::unassigned) return new_status[row];
            for (auto idx = tid + row_ptrs[row]; idx < row_ptrs[row + 1];
                 idx += width) {
                if (new_status[col_idxs[idx]] == kernels::pmis::coarse)
                    return kernels::pmis::fine;
            }
            for (auto idx = tid + ghost_row_ptrs[row]; idx < ghost_row_ptrs[row + 1];
                 idx += width) {
                if (ghost_status[ghost_col_idxs[idx]] == kernels::pmis::coarse)
                    return kernels::pmis::fine;
            }
            return kernels::pmis::unassigned;
        },
        [] GKO_KERNEL(auto a, auto b) { return a > b ? a : b; },
        [] GKO_KERNEL(auto a) { return a; }, int{-1}, new_status, 1,
        dim<2>{strong_dep->get_size()[0], width}, new_status,
        strong_dep->get_const_row_ptrs(), strong_dep->get_const_col_idxs(),
        strong_ghost_dep->get_const_row_ptrs(),
        strong_ghost_dep->get_const_col_idxs(), ghost_status);
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
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto status, auto row_ptrs, auto col_idxs,
                      auto ghost_row_ptrs, auto ghost_col_idxs,
                      auto ghost_status, auto diag_row_ptr, auto off_diag_row_ptr) {
            using idx_t = std::decay_t<decltype(diag_row_ptr[0])>;
            if (status[row] == kernels::pmis::coarse) {
                diag_row_ptr[row] = one<idx_t>();
                off_diag_row_ptr[row] = zero<idx_t>();
                return;
            }
            auto dcnt = zero<idx_t>();
            auto ocnt = zero<idx_t>();
            for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
                if (status[col_idxs[idx]] == kernels::pmis::coarse) dcnt++;
            }
            for (auto idx = ghost_row_ptrs[row]; idx < ghost_row_ptrs[row + 1]; idx++) {
                if (ghost_status[ghost_col_idxs[idx]] == kernels::pmis::coarse) ocnt++;
            }
            diag_row_ptr[row] = dcnt;
            off_diag_row_ptr[row] = ocnt;
        },
        strong_dep->get_size()[0], status,
        strong_dep->get_const_row_ptrs(), strong_dep->get_const_col_idxs(),
        strong_ghost_dep->get_const_row_ptrs(),
        strong_ghost_dep->get_const_col_idxs(),
        ghost_status, diag_row_ptr, off_diag_row_ptr);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_DIRECT_INTERP_ROW_COUNT_DIST_KERNEL);


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
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto strength_threshold,
                      auto coarse_map, auto ghost_coarse_status,
                      auto d_row_ptrs, auto d_col_idxs, auto d_values,
                      auto o_row_ptrs, auto o_col_idxs, auto o_values,
                      auto prolong_row_ptrs, auto prolong_col_idxs,
                      auto prolong_values) {
            if (coarse_map[row] != coarse_map[row + 1]) {
                prolong_col_idxs[prolong_row_ptrs[row]] = coarse_map[row];
                prolong_values[prolong_row_ptrs[row]] = one(prolong_values[0]);
                return;
            }
            auto max_abs = row_maxabs[row];
            auto zv = zero(d_values[0]);
            auto pos = zv, pos_div = zv, neg = zv, neg_div = zv, diag_val = zv;
            bool enable_pos = false, enable_neg = false;
            for (auto idx = d_row_ptrs[row]; idx < d_row_ptrs[row + 1]; idx++) {
                auto val = d_values[idx];
                auto col = d_col_idxs[idx];
                if (col == static_cast<decltype(col)>(row)) { diag_val = val; continue; }
                bool lc = coarse_map[col] != coarse_map[col + 1];
                bool s = abs(val) >= strength_threshold * max_abs;
                if (real(val) >= real(zv)) { pos += val; if (lc && s) { pos_div += val; enable_pos = true; } }
                else { neg += val; if (lc && s) { neg_div += val; enable_neg = true; } }
            }
            for (auto idx = o_row_ptrs[row]; idx < o_row_ptrs[row + 1]; idx++) {
                auto val = o_values[idx];
                auto gcol = o_col_idxs[idx];
                bool gc = ghost_coarse_status[gcol] == kernels::pmis::coarse;
                bool s = abs(val) >= strength_threshold * max_abs;
                if (real(val) >= real(zv)) { pos += val; if (gc && s) { pos_div += val; enable_pos = true; } }
                else { neg += val; if (gc && s) { neg_div += val; enable_neg = true; } }
            }
            pos = safe_divide(pos, pos_div);
            neg = safe_divide(neg, neg_div);
            if (!enable_pos && !enable_neg) return;
            auto p_idx = prolong_row_ptrs[row];
            for (auto idx = d_row_ptrs[row]; idx < d_row_ptrs[row + 1]; idx++) {
                auto val = d_values[idx];
                auto col = d_col_idxs[idx];
                if (col == static_cast<decltype(col)>(row) || abs(val) < strength_threshold * max_abs) continue;
                if (coarse_map[col] == coarse_map[col + 1]) continue;
                if (real(val) >= real(zv) && enable_pos) { prolong_col_idxs[p_idx] = coarse_map[col]; prolong_values[p_idx++] = -pos * val / diag_val; }
                else if (real(val) < real(zv) && enable_neg) { prolong_col_idxs[p_idx] = coarse_map[col]; prolong_values[p_idx++] = -neg * val / diag_val; }
            }
        },
        diag_csr->get_size()[0], row_maxabs, strength_threshold, coarse_map,
        ghost_coarse_status, diag_csr->get_const_row_ptrs(),
        diag_csr->get_const_col_idxs(), diag_csr->get_const_values(),
        off_diag_csr->get_const_row_ptrs(), off_diag_csr->get_const_col_idxs(),
        off_diag_csr->get_const_values(), prolong_row_ptrs, prolong_col_idxs,
        prolong_values);
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
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_maxabs, auto strength_threshold,
                      auto coarse_map, auto ghost_coarse_local_idx,
                      auto ghost_coarse_status,
                      auto d_row_ptrs, auto d_col_idxs, auto d_values,
                      auto o_row_ptrs, auto o_col_idxs, auto o_values,
                      auto prolong_row_ptrs, auto prolong_col_idxs,
                      auto prolong_values) {
            if (coarse_map[row] != coarse_map[row + 1]) return; // coarse: no off-diag
            auto max_abs = row_maxabs[row];
            auto zv = zero(d_values[0]);
            auto pos = zv, pos_div = zv, neg = zv, neg_div = zv, diag_val = zv;
            bool enable_pos = false, enable_neg = false;
            for (auto idx = d_row_ptrs[row]; idx < d_row_ptrs[row + 1]; idx++) {
                auto val = d_values[idx];
                auto col = d_col_idxs[idx];
                if (col == static_cast<decltype(col)>(row)) { diag_val = val; continue; }
                bool lc = coarse_map[col] != coarse_map[col + 1];
                bool s = abs(val) >= strength_threshold * max_abs;
                if (real(val) >= real(zv)) { pos += val; if (lc && s) { pos_div += val; enable_pos = true; } }
                else { neg += val; if (lc && s) { neg_div += val; enable_neg = true; } }
            }
            for (auto idx = o_row_ptrs[row]; idx < o_row_ptrs[row + 1]; idx++) {
                auto val = o_values[idx];
                auto gcol = o_col_idxs[idx];
                bool gc = ghost_coarse_status[gcol] == kernels::pmis::coarse;
                bool s = abs(val) >= strength_threshold * max_abs;
                if (real(val) >= real(zv)) { pos += val; if (gc && s) { pos_div += val; enable_pos = true; } }
                else { neg += val; if (gc && s) { neg_div += val; enable_neg = true; } }
            }
            pos = safe_divide(pos, pos_div);
            neg = safe_divide(neg, neg_div);
            if (!enable_pos && !enable_neg) return;
            auto p_idx = prolong_row_ptrs[row];
            for (auto idx = o_row_ptrs[row]; idx < o_row_ptrs[row + 1]; idx++) {
                auto val = o_values[idx];
                auto gcol = o_col_idxs[idx];
                if (abs(val) < strength_threshold * max_abs) continue;
                if (ghost_coarse_status[gcol] != kernels::pmis::coarse) continue;
                if (real(val) >= real(zv) && enable_pos) { prolong_col_idxs[p_idx] = ghost_coarse_local_idx[gcol]; prolong_values[p_idx++] = -pos * val / diag_val; }
                else if (real(val) < real(zv) && enable_neg) { prolong_col_idxs[p_idx] = ghost_coarse_local_idx[gcol]; prolong_values[p_idx++] = -neg * val / diag_val; }
            }
        },
        diag_csr->get_size()[0], row_maxabs, strength_threshold, coarse_map,
        ghost_coarse_local_idx, ghost_coarse_status,
        diag_csr->get_const_row_ptrs(), diag_csr->get_const_col_idxs(),
        diag_csr->get_const_values(), off_diag_csr->get_const_row_ptrs(),
        off_diag_csr->get_const_col_idxs(), off_diag_csr->get_const_values(),
        prolong_row_ptrs, prolong_col_idxs, prolong_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_DIRECT_INTERP_FILL_OFF_DIAG_KERNEL);


template <typename ValueType, typename IndexType>
void truncate_prolongation_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* prolong,
    remove_complex<ValueType> trunc_factor, IndexType* row_count)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto trunc_factor, auto row_ptrs, auto values,
                      auto row_count) {
            const auto row_start = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            if (row_end == row_start) {
                row_count[row] = zero<IndexType>();
                return;
            }
            auto max_abs =
                zero<remove_complex<std::decay_t<decltype(values[0])>>>();
            for (auto idx = row_start; idx < row_end; idx++) {
                max_abs = max(max_abs, abs(values[idx]));
            }
            const auto threshold = trunc_factor * max_abs;
            IndexType count = zero<IndexType>();
            for (auto idx = row_start; idx < row_end; idx++) {
                if (abs(values[idx]) >= threshold) count++;
            }
            row_count[row] = count;
        },
        prolong->get_size()[0], trunc_factor, prolong->get_const_row_ptrs(),
        prolong->get_const_values(), row_count);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_COUNT_KERNEL);


template <typename ValueType, typename IndexType>
void truncate_prolongation_fill(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* prolong,
    remove_complex<ValueType> trunc_factor, const IndexType* new_row_ptrs,
    IndexType* new_col_idxs, ValueType* new_values)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto trunc_factor, auto row_ptrs, auto col_idxs,
                      auto values, auto new_row_ptrs, auto new_col_idxs,
                      auto new_values) {
            const auto row_start = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            if (row_end == row_start) return;
            auto max_abs = zero<remove_complex<std::decay_t<decltype(values[0])>>>();
            for (auto idx = row_start; idx < row_end; idx++) {
                max_abs = max(max_abs, abs(values[idx]));
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
        },
        prolong->get_size()[0], trunc_factor, prolong->get_const_row_ptrs(),
        prolong->get_const_col_idxs(), prolong->get_const_values(),
        new_row_ptrs, new_col_idxs, new_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PMIS_TRUNCATE_PROLONGATION_FILL_KERNEL);


}  // namespace pmis
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
