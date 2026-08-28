// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_JACOBI_SMOOTHER_HPP_
#define GKO_PUBLIC_CORE_SOLVER_JACOBI_SMOOTHER_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace solver {


/**
 * Fused damped-Jacobi smoother for multigrid.
 *
 * Each apply() call performs num_iters iterations of:
 *   x[i] += omega * (b[i] - sum_j A[i,j]*x[j]) / A[i,i]
 *
 * Both the residual computation and the diagonal scaling are fused into a
 * single GPU kernel launch per iteration, saving two extra memory passes
 * compared to separate SpMV + Jacobi-apply + AXPY kernels.
 *
 * The system matrix must be gko::matrix::Csr<ValueType, IndexType>.
 * apply() treats x as the initial iterate (apply_uses_initial_guess() == true).
 *
 * @tparam ValueType  floating-point type of matrix and vector entries
 * @tparam IndexType  integer type for CSR indices (default int32)
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class JacobiSmoother : public LinOp {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using csr_type = matrix::Csr<ValueType, IndexType>;

    bool apply_uses_initial_guess() const override { return true; }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /** Damping factor omega (Richardson relaxation). */
        ValueType GKO_FACTORY_PARAMETER_SCALAR(omega, static_cast<ValueType>(0.9));

        /** Number of smoother iterations per apply() call. */
        int GKO_FACTORY_PARAMETER_SCALAR(num_iters, 2);
    };
    GKO_ENABLE_LIN_OP_FACTORY(JacobiSmoother, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;
    void apply_impl(const LinOp* alpha, const LinOp* b,
                    const LinOp* beta, LinOp* x) const override;

    explicit JacobiSmoother(std::shared_ptr<const Executor> exec)
        : LinOp(std::move(exec))
    {}

    explicit JacobiSmoother(const Factory* factory,
                            std::shared_ptr<const LinOp> system_matrix);

private:
    std::shared_ptr<const csr_type> A_;
    array<ValueType> inv_diag_;  // 1/A[i,i], pre-computed at construction
    // Pre-allocated residual vector and scalars: avoids cudaMalloc/cudaFree per apply.
    mutable std::shared_ptr<matrix::Dense<ValueType>> r_dense_;
    std::shared_ptr<matrix::Dense<ValueType>> neg_one_;
    std::shared_ptr<matrix::Dense<ValueType>> one_;
    ValueType omega_;
    int num_iters_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_JACOBI_SMOOTHER_HPP_
