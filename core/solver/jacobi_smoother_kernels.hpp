// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_JACOBI_SMOOTHER_KERNELS_HPP_
#define GKO_CORE_SOLVER_JACOBI_SMOOTHER_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace jacobi_smoother {


// Step 1 of the IR smoother: compute the residual into a pre-allocated buffer.
//   r[row] = b[row] - sum_j A[row,j] * x[j]
// Reads the full x before any writes, so there is no race condition.
#define GKO_DECLARE_JACOBI_SMOOTHER_RESIDUAL_KERNEL(ValueType, IndexType) \
    void compute_residual(std::shared_ptr<const DefaultExecutor> exec,    \
                          const IndexType* row_ptrs,                      \
                          const IndexType* col_idxs,                      \
                          const ValueType* values, size_type num_rows,    \
                          const ValueType* b, const ValueType* x,         \
                          ValueType* r)

// Step 2 of the IR smoother: fused inv-diagonal-scale + in-place AXPY update.
//   x[row] += omega * r[row] * inv_diag[row]   (multiply, not divide)
// Reads r (written by kernel 1, not x), so there is no race condition.
#define GKO_DECLARE_JACOBI_SMOOTHER_UPDATE_KERNEL(ValueType)    \
    void apply_update(std::shared_ptr<const DefaultExecutor> exec, \
                      size_type num_rows,                           \
                      const ValueType* r, ValueType* x,            \
                      const ValueType* inv_diag, ValueType omega)

// Invert array in-place: d[i] = 1/d[i].  Called once at construction.
#define GKO_DECLARE_JACOBI_SMOOTHER_INVERT_DIAG_KERNEL(ValueType) \
    void invert_diag(std::shared_ptr<const DefaultExecutor> exec,  \
                     size_type num_rows, ValueType* diag)

// Extract the main diagonal of a CSR matrix into a separate array.
#define GKO_DECLARE_JACOBI_SMOOTHER_EXTRACT_DIAG_KERNEL(ValueType, IndexType) \
    void extract_diag(std::shared_ptr<const DefaultExecutor> exec,             \
                      const IndexType* row_ptrs, const IndexType* col_idxs,    \
                      const ValueType* values, size_type num_rows,             \
                      ValueType* diag)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                                    \
    template <typename ValueType, typename IndexType>                                   \
    GKO_DECLARE_JACOBI_SMOOTHER_RESIDUAL_KERNEL(ValueType, IndexType);                 \
    template <typename ValueType>                                                       \
    GKO_DECLARE_JACOBI_SMOOTHER_UPDATE_KERNEL(ValueType);                              \
    template <typename ValueType>                                                       \
    GKO_DECLARE_JACOBI_SMOOTHER_INVERT_DIAG_KERNEL(ValueType);                        \
    template <typename ValueType, typename IndexType>                                   \
    GKO_DECLARE_JACOBI_SMOOTHER_EXTRACT_DIAG_KERNEL(ValueType, IndexType)


}  // namespace jacobi_smoother


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(jacobi_smoother,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_JACOBI_SMOOTHER_KERNELS_HPP_
