// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/jacobi_smoother_kernels.hpp"

#include <vector>

#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace reference {
namespace jacobi_smoother {


template <typename ValueType, typename IndexType>
void compute_residual(std::shared_ptr<const ReferenceExecutor> exec,
                      const IndexType* row_ptrs, const IndexType* col_idxs,
                      const ValueType* values, size_type num_rows,
                      const ValueType* b, const ValueType* x, ValueType* r)
{
    for (size_type row = 0; row < num_rows; ++row) {
        auto acc = b[row];
        for (auto j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
            acc -= values[j] * x[col_idxs[j]];
        }
        r[row] = acc;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SMOOTHER_RESIDUAL_KERNEL);


template <typename ValueType>
void apply_update(std::shared_ptr<const ReferenceExecutor> exec,
                  size_type num_rows,
                  const ValueType* r, ValueType* x,
                  const ValueType* inv_diag, ValueType omega)
{
    for (size_type row = 0; row < num_rows; ++row) {
        x[row] += omega * r[row] * inv_diag[row];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SMOOTHER_UPDATE_KERNEL);


template <typename ValueType>
void invert_diag(std::shared_ptr<const ReferenceExecutor> exec,
                 size_type num_rows, ValueType* diag)
{
    for (size_type row = 0; row < num_rows; ++row) {
        diag[row] = one<ValueType>() / diag[row];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SMOOTHER_INVERT_DIAG_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diag(std::shared_ptr<const ReferenceExecutor> exec,
                  const IndexType* row_ptrs, const IndexType* col_idxs,
                  const ValueType* values, size_type num_rows,
                  ValueType* diag)
{
    for (size_type row = 0; row < num_rows; ++row) {
        for (auto j = row_ptrs[row]; j < row_ptrs[row + 1]; j++) {
            if (col_idxs[j] == static_cast<IndexType>(row)) {
                diag[row] = values[j];
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SMOOTHER_EXTRACT_DIAG_KERNEL);


}  // namespace jacobi_smoother
}  // namespace reference
}  // namespace kernels
}  // namespace gko
