// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/jacobi_smoother.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/solver/jacobi_smoother_kernels.hpp"


namespace gko {
namespace solver {
namespace js_ops {
namespace {


GKO_REGISTER_OPERATION(compute_residual, jacobi_smoother::compute_residual);
GKO_REGISTER_OPERATION(apply_update, jacobi_smoother::apply_update);
GKO_REGISTER_OPERATION(invert_diag, jacobi_smoother::invert_diag);
GKO_REGISTER_OPERATION(extract_diag, jacobi_smoother::extract_diag);


}  // anonymous namespace
}  // namespace js_ops


// ---- Constructor (called by Factory::generate_impl) ------------------------

template <typename ValueType, typename IndexType>
JacobiSmoother<ValueType, IndexType>::JacobiSmoother(
    const Factory* factory, std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), system_matrix->get_size()),
      parameters_{factory->get_parameters()}
{
    auto exec = this->get_executor();

    auto A = std::dynamic_pointer_cast<const csr_type>(system_matrix);
    if (!A) {
        GKO_NOT_SUPPORTED(*system_matrix);
    }
    A_ = std::move(A);
    omega_ = parameters_.omega;
    num_iters_ = parameters_.num_iters;

    auto n_rows = A_->get_size()[0];
    // Extract diagonal, then invert: multiply is 10-20x faster than divide on GPU.
    inv_diag_ = array<ValueType>(exec, n_rows);
    inv_diag_.fill(one<ValueType>());  // safe default (=1) for rows with no diagonal
    // Pre-allocate residual vector and scalars to avoid per-apply allocation.
    r_dense_ = matrix::Dense<ValueType>::create(exec, dim<2>{n_rows, 1});
    neg_one_ = initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);
    one_     = initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);

    exec->run(js_ops::make_extract_diag(
        A_->get_const_row_ptrs(), A_->get_const_col_idxs(),
        A_->get_const_values(), n_rows, inv_diag_.get_data()));
    // Invert in place: inv_diag_[i] = 1 / diag_[i]
    exec->run(js_ops::make_invert_diag(n_rows, inv_diag_.get_data()));
}


// ---- apply_impl ------------------------------------------------------------
//
// Implements num_iters steps of IR(Jacobi) with pre-allocated scratch:
//   r = b - A*x          (compute_residual kernel, reads full x before writes)
//   x += omega * r / d   (apply_update kernel, fused scale+axpy)
//
// Algorithmically identical to Ginkgo's IR(Jacobi) smoother but avoids
// per-apply cudaMalloc/cudaFree and reduces kernel launches from 4 to 2
// per step.

template <typename ValueType, typename IndexType>
void JacobiSmoother<ValueType, IndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    auto exec = this->get_executor();
    auto dense_b = as<matrix::Dense<ValueType>>(b);
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto n_rows = A_->get_size()[0];
    auto*       x_vals   = dense_x->get_values();
    const auto* inv_diag = inv_diag_.get_const_data();

    for (int iter = 0; iter < num_iters_; ++iter) {
        // Step 1: r = b - A*x  using Ginkgo's optimized SpMV (cuSPARSE on CUDA)
        r_dense_->copy_from(dense_b);
        A_->apply(neg_one_.get(), dense_x, one_.get(), r_dense_.get());
        // Step 2: x += omega * r * inv_diag  (multiply, not divide — much faster on GPU)
        exec->run(js_ops::make_apply_update(
            n_rows, r_dense_->get_const_values(), x_vals, inv_diag, omega_));
    }
}


template <typename ValueType, typename IndexType>
void JacobiSmoother<ValueType, IndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    auto dense_alpha = as<matrix::Dense<ValueType>>(alpha);
    auto dense_beta = as<matrix::Dense<ValueType>>(beta);
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    apply_impl(b, x_clone.get());
    dense_x->scale(dense_beta);
    dense_x->add_scaled(dense_alpha, x_clone.get());
}


// ---- Explicit instantiations -----------------------------------------------

#define GKO_DECLARE_JACOBI_SMOOTHER(V, I) class JacobiSmoother<V, I>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_SMOOTHER);


}  // namespace solver
}  // namespace gko
