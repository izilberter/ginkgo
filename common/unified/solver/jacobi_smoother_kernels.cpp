// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/jacobi_smoother_kernels.hpp"

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace jacobi_smoother {


template <typename ValueType, typename IndexType>
void compute_residual(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* row_ptrs, const IndexType* col_idxs,
                      const ValueType* values, size_type num_rows,
                      const ValueType* b, const ValueType* x, ValueType* r)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto values,
                      auto b, auto x, auto r) {
            auto acc = b[row];
            for (auto j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
                acc -= values[j] * x[col_idxs[j]];
            }
            r[row] = acc;
        },
        num_rows, row_ptrs, col_idxs, values, b, x, r);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SMOOTHER_RESIDUAL_KERNEL);


template <typename ValueType>
void apply_update(std::shared_ptr<const DefaultExecutor> exec,
                  size_type num_rows,
                  const ValueType* r, ValueType* x,
                  const ValueType* inv_diag, ValueType omega)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto r, auto x, auto inv_diag, auto omega) {
            x[row] += omega * r[row] * inv_diag[row];
        },
        num_rows, r, x, inv_diag, omega);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SMOOTHER_UPDATE_KERNEL);


template <typename ValueType>
void invert_diag(std::shared_ptr<const DefaultExecutor> exec,
                 size_type num_rows, ValueType* diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto diag) {
            diag[row] = one(diag[row]) / diag[row];
        },
        num_rows, diag);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SMOOTHER_INVERT_DIAG_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diag(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* row_ptrs, const IndexType* col_idxs,
                  const ValueType* values, size_type num_rows,
                  ValueType* diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto values,
                      auto diag) {
            for (auto j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
                if (col_idxs[j] == static_cast<decltype(col_idxs[j])>(row)) {
                    diag[row] = values[j];
                    return;
                }
            }
        },
        num_rows, row_ptrs, col_idxs, values, diag);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SMOOTHER_EXTRACT_DIAG_KERNEL);


}  // namespace jacobi_smoother
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
