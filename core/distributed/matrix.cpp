// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/matrix.hpp"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/assembly.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/distributed/matrix_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace matrix {
namespace {


GKO_REGISTER_OPERATION(separate_diag_off_diag,
                       distributed_matrix::separate_diag_off_diag);


}  // namespace
}  // namespace matrix


namespace {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void append_global_matrix_data(
    std::shared_ptr<const Executor> exec,
    const matrix_data<ValueType, LocalIndexType>& local_data,
    const index_map<LocalIndexType, GlobalIndexType>& imap,
    index_space col_space, matrix_data<ValueType, GlobalIndexType>& result)
{
    if (local_data.nonzeros.empty()) {
        return;
    }

    auto device_data =
        device_matrix_data<ValueType, LocalIndexType>::create_from_host(
            exec, local_data);
    const auto nnz = device_data.get_num_stored_elements();
    auto local_row_idxs =
        make_array_view(exec, nnz, device_data.get_row_idxs());
    auto local_col_idxs =
        make_array_view(exec, nnz, device_data.get_col_idxs());
    auto global_row_idxs =
        imap.map_to_global(local_row_idxs, index_space::local);
    auto global_col_idxs = imap.map_to_global(local_col_idxs, col_space);
    auto values = make_array_view(exec, nnz, device_data.get_values());
    auto global_data =
        device_matrix_data<ValueType, GlobalIndexType>{
            exec, result.size, std::move(global_row_idxs),
            std::move(global_col_idxs), std::move(values)}
            .copy_to_host();

    result.nonzeros.insert(result.nonzeros.end(), global_data.nonzeros.begin(),
                           global_data.nonzeros.end());
}


}  // namespace


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
    : Matrix(exec, RowGatherer<LocalIndexType>::create(exec, comm),
             gko::matrix::Csr<ValueType, LocalIndexType>::create(exec),
             gko::matrix::Csr<ValueType, LocalIndexType>::create(exec))
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const RowGatherer<LocalIndexType>> row_gather_template,
    ptr_param<const LinOp> diag_matrix_template,
    ptr_param<const LinOp> off_diag_matrix_template)
    : LinOp{exec},
      DistributedBase{row_gather_template->get_communicator()},
      row_gatherer_{clone(exec, row_gather_template)},
      imap_{exec},
      one_scalar_{exec, 1.0},
      diag_mtx_{clone(exec, diag_matrix_template.get())},
      off_diag_mtx_{clone(exec, off_diag_matrix_template.get())}
{
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            diag_mtx_.get())));
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            off_diag_mtx_.get())));
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> diag_linop)
    : LinOp{exec},
      DistributedBase{comm},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          exec, mpi::detail::create_default_collective_communicator(comm))},
      imap_{exec},
      one_scalar_{exec, 1.0},
      off_diag_mtx_(::gko::matrix::Coo<ValueType, LocalIndexType>::create(
          exec, dim<2>{diag_linop->get_size()[0], 0}))
{
    this->set_size(size);
    auto partition =
        share(build_partition_from_local_size<LocalIndexType, GlobalIndexType>(
            exec, comm, diag_linop->get_size()[0]));
    imap_ = index_map<local_index_type, global_index_type>(
        exec, partition, comm.rank(), array<global_index_type>{exec});
    diag_mtx_ = std::move(diag_linop);
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    index_map<LocalIndexType, GlobalIndexType> imap,
    std::shared_ptr<LinOp> diag_linop, std::shared_ptr<LinOp> off_diag_linop)
    : LinOp{exec},
      DistributedBase{comm},
      row_gatherer_(RowGatherer<LocalIndexType>::create(
          exec,
          mpi::detail::create_default_collective_communicator(comm)
              ->create_with_same_type(comm, &imap),
          imap)),
      imap_(std::move(imap)),
      one_scalar_{exec, 1.0}
{
    this->set_size({imap_.get_global_size(), imap_.get_global_size()});
    diag_mtx_ = std::move(diag_linop);
    off_diag_mtx_ = std::move(off_diag_linop);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
{
    return std::unique_ptr<Matrix>{new Matrix{exec, comm}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> matrix_template)
{
    return create(exec, comm, matrix_template, matrix_template);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> diag_matrix_template,
    ptr_param<const LinOp> off_diag_matrix_template)
{
    return std::unique_ptr<Matrix>{new Matrix{
        exec,
        RowGatherer<LocalIndexType>::create(
            exec, mpi::detail::create_default_collective_communicator(comm)),
        diag_matrix_template, off_diag_matrix_template}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> diag_linop)
{
    return std::unique_ptr<Matrix>{new Matrix{exec, comm, size, diag_linop}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> diag_linop, std::shared_ptr<LinOp> off_diag_linop,
    std::vector<comm_index_type> recv_sizes,
    std::vector<comm_index_type> recv_offsets,
    array<local_index_type> recv_gather_idxs)
{
    array<comm_index_type> part_ids(exec->get_master(), comm.size());
    std::iota(part_ids.get_data(), part_ids.get_data() + part_ids.get_size(),
              0);
    auto contiguous_partition =
        share(build_partition_from_local_size<LocalIndexType, GlobalIndexType>(
            exec, comm, diag_linop->get_size()[0]));
    array<global_index_type> global_recv_gather_idxs(
        exec, recv_gather_idxs.get_size());
    for (int rank = 0; rank < comm.size(); ++rank) {
        if (recv_sizes[rank] > 0) {
            auto map = index_map<LocalIndexType, GlobalIndexType>(
                exec, contiguous_partition, rank, array<GlobalIndexType>{exec});
            auto local_view = make_array_view(
                exec, recv_sizes[rank],
                recv_gather_idxs.get_data() + recv_offsets[rank]);
            auto global_idxs =
                map.map_to_global(local_view, index_space::local);
            exec->copy(recv_sizes[rank], global_idxs.get_const_data(),
                       global_recv_gather_idxs.get_data() + recv_offsets[rank]);
        }
    }

    return Matrix::create(
        exec, comm,
        index_map<LocalIndexType, GlobalIndexType>(
            exec, contiguous_partition, comm.rank(), global_recv_gather_idxs),
        std::move(diag_linop), std::move(off_diag_linop));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    index_map<LocalIndexType, GlobalIndexType> imap,
    std::shared_ptr<LinOp> diag_linop, std::shared_ptr<LinOp> off_diag_linop)
{
    return std::unique_ptr<Matrix>{
        new Matrix{std::move(exec), comm, std::move(imap),
                   std::move(diag_linop), std::move(off_diag_linop)}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    as<Cloneable>(result->diag_mtx_)->copy_from(as<Cloneable>(this->diag_mtx_));
    as<Cloneable>(result->off_diag_mtx_)
        ->copy_from(as<Cloneable>(this->off_diag_mtx_));
    as<Cloneable>(result->row_gatherer_)
        ->copy_from(as<Cloneable>(this->row_gatherer_));
    result->imap_ = this->imap_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Matrix<next_precision<value_type>, local_index_type, global_index_type>*
        result)
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    as<Cloneable>(result->diag_mtx_)->move_from(as<Cloneable>(this->diag_mtx_));
    as<Cloneable>(result->off_diag_mtx_)
        ->move_from(as<Cloneable>(this->off_diag_mtx_));
    as<Cloneable>(result->row_gatherer_)
        ->move_from(as<Cloneable>(this->row_gatherer_));
    result->imap_ = std::move(this->imap_);
    result->set_size(this->get_size());
    this->set_size({});
}


#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type, 2>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    as<Cloneable>(result->diag_mtx_)
        ->copy_from(as<Cloneable>(this->diag_mtx_.get()));
    as<Cloneable>(result->off_diag_mtx_)
        ->copy_from(as<Cloneable>(this->off_diag_mtx_.get()));
    as<Cloneable>(result->row_gatherer_)
        ->copy_from(as<Cloneable>(this->row_gatherer_));
    result->imap_ = this->imap_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Matrix<next_precision<value_type, 2>, local_index_type, global_index_type>*
        result)
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    as<Cloneable>(result->diag_mtx_)
        ->move_from(as<Cloneable>(this->diag_mtx_.get()));
    as<Cloneable>(result->off_diag_mtx_)
        ->move_from(as<Cloneable>(this->off_diag_mtx_.get()));
    result->row_gatherer_->move_from(this->row_gatherer_);
    result->imap_ = std::move(this->imap_);
    result->set_size(this->get_size());
    this->set_size({});
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type, 3>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    as<Cloneable>(result->diag_mtx_)
        ->copy_from(as<Cloneable>(this->diag_mtx_.get()));
    as<Cloneable>(result->off_diag_mtx_)
        ->copy_from(as<Cloneable>(this->off_diag_mtx_.get()));
    result->row_gatherer_->copy_from(this->row_gatherer_);
    result->imap_ = this->imap_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Matrix<next_precision<value_type, 3>, local_index_type, global_index_type>*
        result)
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    as<Cloneable>(result->diag_mtx_)
        ->move_from(as<Cloneable>(this->diag_mtx_.get()));
    as<Cloneable>(result->off_diag_mtx_)
        ->move_from(as<Cloneable>(this->off_diag_mtx_.get()));
    result->row_gatherer_->move_from(this->row_gatherer_);
    result->imap_ = std::move(this->imap_);
    result->set_size(this->get_size());
    this->set_size({});
}
#endif


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        row_partition,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        col_partition,
    assembly_mode assembly_type)
{
    const auto comm = this->get_communicator();
    GKO_ASSERT_EQ(data.get_size()[0], row_partition->get_size());
    GKO_ASSERT_EQ(data.get_size()[1], col_partition->get_size());
    GKO_ASSERT_EQ(comm.size(), row_partition->get_num_parts());
    GKO_ASSERT_EQ(comm.size(), col_partition->get_num_parts());
    auto exec = this->get_executor();
    auto local_part = comm.rank();
    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    auto tmp_row_partition = make_temporary_clone(exec, row_partition);
    auto tmp_col_partition = make_temporary_clone(exec, col_partition);

    const device_matrix_data<value_type, global_index_type>* all_data_ptr =
        &data;
    device_matrix_data<value_type, global_index_type> assembled_data(exec);
    if (assembly_type == assembly_mode::communicate) {
        assembled_data = assemble_rows_from_neighbors<ValueType, LocalIndexType,
                                                      GlobalIndexType>(
            this->get_communicator(), data, row_partition);
        all_data_ptr = &assembled_data;
    }

    // set up LinOp sizes
    auto global_num_rows = row_partition->get_size();
    auto global_num_cols = col_partition->get_size();
    dim<2> global_dim{global_num_rows, global_num_cols};
    this->set_size(global_dim);

    // temporary storage for the output
    array<local_index_type> diag_row_idxs{exec};
    array<local_index_type> diag_col_idxs{exec};
    array<value_type> diag_values{exec};
    array<local_index_type> off_diag_row_idxs{exec};
    array<global_index_type> global_off_diag_col_idxs{exec};
    array<value_type> off_diag_values{exec};

    // separate input into diag and off-diag block
    // The rows and columns of the diag block are mapped into local indexing,
    // as well as the rows of the off-diag block. The columns of the off-diag
    // block are still in global indices.
    exec->run(matrix::make_separate_diag_off_diag(
        *all_data_ptr, tmp_row_partition.get(), tmp_col_partition.get(),
        local_part, diag_row_idxs, diag_col_idxs, diag_values,
        off_diag_row_idxs, global_off_diag_col_idxs, off_diag_values));

    imap_ = index_map<local_index_type, global_index_type>(
        exec, col_partition, comm.rank(), global_off_diag_col_idxs);

    auto off_diag_col_idxs =
        imap_.map_to_local(global_off_diag_col_idxs, index_space::non_local);

    // read the diag matrix data
    const auto num_local_rows =
        static_cast<size_type>(row_partition->get_part_size(local_part));
    const auto num_local_cols =
        static_cast<size_type>(col_partition->get_part_size(local_part));
    device_matrix_data<value_type, local_index_type> diag_data{
        exec, dim<2>{num_local_rows, num_local_cols}, std::move(diag_row_idxs),
        std::move(diag_col_idxs), std::move(diag_values)};
    device_matrix_data<value_type, local_index_type> off_diag_data{
        exec, dim<2>{num_local_rows, imap_.get_remote_global_idxs().get_size()},
        std::move(off_diag_row_idxs), std::move(off_diag_col_idxs),
        std::move(off_diag_values)};
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->diag_mtx_)
        ->read(std::move(diag_data));
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->off_diag_mtx_)
        ->read(std::move(off_diag_data));

    row_gatherer_ = RowGatherer<LocalIndexType>::create(
        row_gatherer_->get_executor(),
        row_gatherer_->get_collective_communicator()->create_with_same_type(
            comm, &imap_),
        imap_);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        row_partition,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        col_partition,
    assembly_mode assembly_type)
{
    return this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        row_partition, col_partition, assembly_type);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition,
    assembly_mode assembly_type)
{
    return this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition, partition, assembly_type);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition,
    assembly_mode assembly_type)
{
    return this->read_distributed(data, partition, partition, assembly_type);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::write(
    matrix_data<value_type, global_index_type>& data) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this);
    auto diag_data = matrix_data<value_type, local_index_type>{};
    auto off_diag_data = matrix_data<value_type, local_index_type>{};
    as<WritableToMatrixData<ValueType, LocalIndexType>>(this->diag_mtx_)
        ->write(diag_data);
    as<WritableToMatrixData<ValueType, LocalIndexType>>(this->off_diag_mtx_)
        ->write(off_diag_data);

    data = {this->get_size(), {}};
    auto exec = this->get_executor();
    append_global_matrix_data(exec, diag_data, imap_, index_space::local, data);
    append_global_matrix_data(exec, off_diag_data, imap_,
                              index_space::non_local, data);
    data.sort_row_major();
}


template <typename ValueType, typename LocalIndexType>
std::pair<std::shared_ptr<Vector<ValueType>>,
          std::shared_ptr<Vector<ValueType>>>
init_recv_buffers(std::shared_ptr<const Executor> exec,
                  const RowGatherer<LocalIndexType>* row_gatherer,
                  size_type num_cols, const detail::GenericVectorCache& buffer,
                  const detail::GenericVectorCache& host_buffer)
{
    auto comm =
        row_gatherer->get_collective_communicator()->get_base_communicator();
    auto global_recv_dim =
        dim<2>{static_cast<size_type>(row_gatherer->get_size()[0]), num_cols};
    auto local_recv_dim = dim<2>{
        static_cast<size_type>(
            row_gatherer->get_collective_communicator()->get_recv_size()),
        num_cols};

    auto vector = buffer.template get<ValueType>(exec, comm, global_recv_dim,
                                                 local_recv_dim);
    auto host_vector = host_buffer.template get<ValueType>(
        exec->get_master(), comm, global_recv_dim, local_recv_dim);
    return std::make_pair(vector, host_vector);
}


// Helper: build a local CSR from host COO data (rows already sorted).
// Computes CSR row pointers from the COO row array.
template <typename ValueType, typename LocalIndexType>
static std::shared_ptr<gko::matrix::Csr<ValueType, LocalIndexType>>
coo_to_csr(std::shared_ptr<const Executor> exec,
           std::shared_ptr<const Executor> host,
           size_type nrows, size_type ncols,
           std::vector<LocalIndexType>& rows,
           std::vector<LocalIndexType>& cols,
           std::vector<ValueType>& vals)
{
    using csr_type = gko::matrix::Csr<ValueType, LocalIndexType>;
    const auto nnz = rows.size();
    std::vector<LocalIndexType> rp(nrows + 1, 0);
    for (auto r : rows) rp[r + 1]++;
    for (size_type r = 0; r < nrows; r++) rp[r + 1] += rp[r];

    array<LocalIndexType> d_rp{exec, nrows + 1};
    array<LocalIndexType> d_ci{exec, nnz};
    array<ValueType>      d_vs{exec, nnz};
    exec->copy_from(host, nrows + 1, rp.data(),   d_rp.get_data());
    exec->copy_from(host, nnz,       cols.data(),  d_ci.get_data());
    exec->copy_from(host, nnz,       vals.data(),  d_vs.get_data());
    return csr_type::create(exec, dim<2>{nrows, ncols},
                            std::move(d_vs), std::move(d_ci), std::move(d_rp));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::add(
    const Matrix* B, Matrix* C) const
{
    // Computes C = A + B (this = A).
    //
    // Algorithm (all host-side):
    //  1. Copy all CSR blocks (diag, off-diag) to host.
    //  2. Merge A's and B's ghost global column indices into a sorted unique
    //     list.  For contiguous partitions the imap flat arrays are already
    //     globally sorted, so std::merge suffices.
    //  3. Build per-row COO for C_diag by merging the two sorted CSR rows.
    //  4. Build per-row COO for C_off by remapping each matrix's off-diagonal
    //     column indices to the merged ghost space, then merging sorted rows.
    //  5. Construct C's index_map and RowGatherer from the merged ghost cols.
    //  6. Write all fields of C.

    using csr_type = gko::matrix::Csr<ValueType, LocalIndexType>;

    GKO_ASSERT_EQ(this->get_size()[0], B->get_size()[0]);
    GKO_ASSERT_EQ(this->get_size()[1], B->get_size()[1]);

    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto comm = this->get_communicator();
    const int my_rank = comm.rank();

    const auto* A_diag = gko::as<const csr_type>(this->diag_mtx_).get();
    const auto* A_off  = gko::as<const csr_type>(this->off_diag_mtx_).get();
    const auto* B_diag = gko::as<const csr_type>(B->diag_mtx_).get();
    const auto* B_off  = gko::as<const csr_type>(B->off_diag_mtx_).get();

    const auto n_local      = static_cast<size_type>(A_diag->get_size()[0]);
    const auto n_local_cols = static_cast<size_type>(A_diag->get_size()[1]);
    const auto n_ghost_A    = static_cast<size_type>(A_off->get_size()[1]);
    const auto n_ghost_B    = static_cast<size_type>(B_off->get_size()[1]);

    auto d2h = [&](const auto* ptr, size_type n) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
        std::vector<T> v(n);
        if (n > 0) host->copy_from(exec, n, ptr, v.data());
        return v;
    };

    // ---- Copy all CSR blocks to host ----
    auto A_d_rp = d2h(A_diag->get_const_row_ptrs(), n_local + 1);
    auto A_d_ci = d2h(A_diag->get_const_col_idxs(), A_diag->get_num_stored_elements());
    auto A_d_vs = d2h(A_diag->get_const_values(),   A_diag->get_num_stored_elements());
    auto B_d_rp = d2h(B_diag->get_const_row_ptrs(), n_local + 1);
    auto B_d_ci = d2h(B_diag->get_const_col_idxs(), B_diag->get_num_stored_elements());
    auto B_d_vs = d2h(B_diag->get_const_values(),   B_diag->get_num_stored_elements());

    auto A_o_rp = d2h(A_off->get_const_row_ptrs(), n_local + 1);
    auto A_o_ci = d2h(A_off->get_const_col_idxs(), A_off->get_num_stored_elements());
    auto A_o_vs = d2h(A_off->get_const_values(),   A_off->get_num_stored_elements());
    auto B_o_rp = d2h(B_off->get_const_row_ptrs(), n_local + 1);
    auto B_o_ci = d2h(B_off->get_const_col_idxs(), B_off->get_num_stored_elements());
    auto B_o_vs = d2h(B_off->get_const_values(),   B_off->get_num_stored_elements());

    // ---- Merge ghost global column indices ----
    // For contiguous partitions the imap flat arrays are globally sorted
    // (gcols from rank j < rank k are all smaller), so std::merge is valid.
    auto A_ghost_gcol = d2h(
        this->imap_.get_remote_global_idxs().get_const_flat_data(), n_ghost_A);
    auto B_ghost_gcol = d2h(
        B->imap_.get_remote_global_idxs().get_const_flat_data(), n_ghost_B);

    std::vector<GlobalIndexType> C_ghost_gcol;
    C_ghost_gcol.reserve(n_ghost_A + n_ghost_B);
    std::merge(A_ghost_gcol.begin(), A_ghost_gcol.end(),
               B_ghost_gcol.begin(), B_ghost_gcol.end(),
               std::back_inserter(C_ghost_gcol));
    C_ghost_gcol.erase(
        std::unique(C_ghost_gcol.begin(), C_ghost_gcol.end()),
        C_ghost_gcol.end());
    const auto n_c_ghost = C_ghost_gcol.size();

    // Build maps: A/B local ghost index → position in C_ghost_gcol.
    // Monotone because A/B ghost lists are subsets of the merged sorted list.
    auto make_remap = [&](const std::vector<GlobalIndexType>& src) {
        std::vector<LocalIndexType> remap(src.size());
        for (size_type i = 0; i < src.size(); i++) {
            remap[i] = static_cast<LocalIndexType>(
                std::lower_bound(C_ghost_gcol.begin(), C_ghost_gcol.end(),
                                 src[i]) -
                C_ghost_gcol.begin());
        }
        return remap;
    };
    const auto A_to_c = make_remap(A_ghost_gcol);
    const auto B_to_c = make_remap(B_ghost_gcol);

    // ---- Build C diag COO: merge two sorted CSR rows ----
    std::vector<LocalIndexType> c_d_rows, c_d_cols;
    std::vector<ValueType> c_d_vals;

    for (size_type r = 0; r < n_local; r++) {
        auto ai = A_d_rp[r], ae = A_d_rp[r + 1];
        auto bi = B_d_rp[r], be = B_d_rp[r + 1];
        while (ai < ae && bi < be) {
            auto lr = static_cast<LocalIndexType>(r);
            if (A_d_ci[ai] < B_d_ci[bi]) {
                c_d_rows.push_back(lr);
                c_d_cols.push_back(A_d_ci[ai]);
                c_d_vals.emplace_back(A_d_vs[ai]);
                ++ai;
            } else if (A_d_ci[ai] > B_d_ci[bi]) {
                c_d_rows.push_back(lr);
                c_d_cols.push_back(B_d_ci[bi]);
                c_d_vals.emplace_back(B_d_vs[bi]);
                ++bi;
            } else {
                c_d_rows.push_back(lr);
                c_d_cols.push_back(A_d_ci[ai]);
                c_d_vals.emplace_back(A_d_vs[ai] + B_d_vs[bi]);
                ++ai; ++bi;
            }
        }
        for (; ai < ae; ++ai) {
            c_d_rows.push_back(static_cast<LocalIndexType>(r));
            c_d_cols.push_back(A_d_ci[ai]);
            c_d_vals.emplace_back(A_d_vs[ai]);
        }
        for (; bi < be; ++bi) {
            c_d_rows.push_back(static_cast<LocalIndexType>(r));
            c_d_cols.push_back(B_d_ci[bi]);
            c_d_vals.emplace_back(B_d_vs[bi]);
        }
    }

    // ---- Build C off-diag COO: remap then merge sorted CSR rows ----
    // After applying A_to_c / B_to_c (both monotone), each row is still
    // sorted by column, so the merge remains valid.
    std::vector<LocalIndexType> c_o_rows, c_o_cols;
    std::vector<ValueType> c_o_vals;

    for (size_type r = 0; r < n_local; r++) {
        auto ai = A_o_rp[r], ae = A_o_rp[r + 1];
        auto bi = B_o_rp[r], be = B_o_rp[r + 1];
        while (ai < ae && bi < be) {
            auto lr = static_cast<LocalIndexType>(r);
            auto ac = A_to_c[A_o_ci[ai]];
            auto bc = B_to_c[B_o_ci[bi]];
            if (ac < bc) {
                c_o_rows.push_back(lr);
                c_o_cols.push_back(ac);
                c_o_vals.emplace_back(A_o_vs[ai]);
                ++ai;
            } else if (ac > bc) {
                c_o_rows.push_back(lr);
                c_o_cols.push_back(bc);
                c_o_vals.emplace_back(B_o_vs[bi]);
                ++bi;
            } else {
                c_o_rows.push_back(lr);
                c_o_cols.push_back(ac);
                c_o_vals.emplace_back(A_o_vs[ai] + B_o_vs[bi]);
                ++ai; ++bi;
            }
        }
        for (; ai < ae; ++ai) {
            c_o_rows.push_back(static_cast<LocalIndexType>(r));
            c_o_cols.push_back(A_to_c[A_o_ci[ai]]);
            c_o_vals.emplace_back(A_o_vs[ai]);
        }
        for (; bi < be; ++bi) {
            c_o_rows.push_back(static_cast<LocalIndexType>(r));
            c_o_cols.push_back(B_to_c[B_o_ci[bi]]);
            c_o_vals.emplace_back(B_o_vs[bi]);
        }
    }

    // ---- Build C's index_map from merged ghost gcols ----
    auto col_part = gko::share(
        experimental::distributed::build_partition_from_local_size<
            LocalIndexType, GlobalIndexType>(exec, comm, n_local_cols));

    array<GlobalIndexType> d_c_ghost{exec, n_c_ghost};
    if (n_c_ghost > 0)
        exec->copy_from(host, n_c_ghost, C_ghost_gcol.data(),
                        d_c_ghost.get_data());

    auto c_imap =
        experimental::distributed::index_map<LocalIndexType, GlobalIndexType>(
            exec, col_part, my_rank, d_c_ghost);

    // Remap c_o_cols from C_ghost_gcol positions to imap non-local indices.
    // For contiguous partitions these are identical, but we remap via the
    // imap for correctness in the general case.
    auto c_ghost_nloc = c_imap.map_to_local(
        d_c_ghost, experimental::distributed::index_space::non_local);
    std::vector<LocalIndexType> gcol_to_nloc(n_c_ghost);
    if (n_c_ghost > 0)
        host->copy_from(exec, n_c_ghost, c_ghost_nloc.get_const_data(),
                        gcol_to_nloc.data());
    for (auto& col : c_o_cols)
        col = gcol_to_nloc[col];

    const auto n_nonlocal = c_imap.get_non_local_size();

    // ---- Build CSR matrices and RowGatherer for C ----
    auto C_diag_csr = share(coo_to_csr<ValueType, LocalIndexType>(
        exec, host, n_local, n_local_cols, c_d_rows, c_d_cols, c_d_vals));
    auto C_off_csr  = share(coo_to_csr<ValueType, LocalIndexType>(
        exec, host, n_local, n_nonlocal, c_o_rows, c_o_cols, c_o_vals));

    auto c_rg = RowGatherer<LocalIndexType>::create(
        exec,
        C->row_gatherer_->get_collective_communicator()
            ->create_with_same_type(comm, &c_imap),
        c_imap);

    // ---- Write C ----
    const size_type C_global_size = this->get_size()[0];
    C->set_size(dim<2>{C_global_size, C_global_size});
    C->imap_         = std::move(c_imap);
    C->diag_mtx_     = std::move(C_diag_csr);
    C->off_diag_mtx_ = std::move(C_off_csr);
    C->row_gatherer_ = std::move(c_rg);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_spgemm(
    const Matrix* B, Matrix* C) const
{
    // Computes C = A * B (this = A).
    //
    // Algorithm:
    //  1. Exchange B-row nnz counts via A's existing CollectiveCommunicator.
    //  2. Exchange B-row data (global col + value) via MPI_Alltoallv.
    //  3. Build A_ext = [A_diag | A_off] and B_ext (local + ghost rows of B)
    //     with a shared local column space.
    //  4. Local SpGeMM: C_tmp = A_ext * B_ext  (uses cuSPARSE on CUDA).
    //  5. Split C_tmp into diag/off-diag, build new index_map and RowGatherer.
    //
    // Requirements: diag_mtx_ and off_diag_mtx_ must be Csr on both A and B.
    // Assumes contiguous 1D row/column partitions.

    using csr_type = gko::matrix::Csr<ValueType, LocalIndexType>;

    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto comm = this->get_communicator();
    const int nproc   = comm.size();
    const int my_rank = comm.rank();

    // ---- Local CSR blocks ----
    const auto* A_diag = gko::as<const csr_type>(this->diag_mtx_).get();
    const auto* A_off  = gko::as<const csr_type>(this->off_diag_mtx_).get();
    const auto* B_diag = gko::as<const csr_type>(B->diag_mtx_).get();
    const auto* B_off  = gko::as<const csr_type>(B->off_diag_mtx_).get();

    const auto n_local   = static_cast<size_type>(A_diag->get_size()[0]);
    const auto n_ghost_A = static_cast<size_type>(A_off->get_size()[1]);
    const auto n_local_B = static_cast<size_type>(B_diag->get_size()[1]);
    const auto n_ghost_B = static_cast<size_type>(B_off->get_size()[1]);

    // B_col_offset: global index of B's first local column on this rank
    // (exclusive prefix sum of local B column counts across ranks).
    GlobalIndexType B_col_offset = 0;
    {
        long long local_B = static_cast<long long>(n_local_B);
        long long scan_B  = 0;
        MPI_Exscan(&local_B, &scan_B, 1, MPI_LONG_LONG, MPI_SUM, comm.get());
        B_col_offset = (my_rank == 0) ? 0 : static_cast<GlobalIndexType>(scan_B);
    }

    // ---- Copy blocks to host ----
    auto d2h = [&](const auto* ptr, size_type n) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
        std::vector<T> v(n);
        if (n > 0) host->copy_from(exec, n, ptr, v.data());
        return v;
    };

    auto A_d_rp = d2h(A_diag->get_const_row_ptrs(), n_local + 1);
    auto A_d_ci = d2h(A_diag->get_const_col_idxs(), A_diag->get_num_stored_elements());
    auto A_d_vs = d2h(A_diag->get_const_values(),   A_diag->get_num_stored_elements());
    auto A_o_rp = d2h(A_off->get_const_row_ptrs(),  n_local + 1);
    auto A_o_ci = d2h(A_off->get_const_col_idxs(),  A_off->get_num_stored_elements());
    auto A_o_vs = d2h(A_off->get_const_values(),    A_off->get_num_stored_elements());

    auto B_d_rp = d2h(B_diag->get_const_row_ptrs(), n_local + 1);
    auto B_d_ci = d2h(B_diag->get_const_col_idxs(), B_diag->get_num_stored_elements());
    auto B_d_vs = d2h(B_diag->get_const_values(),   B_diag->get_num_stored_elements());
    auto B_o_rp = d2h(B_off->get_const_row_ptrs(),  n_local + 1);
    auto B_o_ci = d2h(B_off->get_const_col_idxs(),  B_off->get_num_stored_elements());
    auto B_o_vs = d2h(B_off->get_const_values(),    B_off->get_num_stored_elements());

    // B's ghost column global indices (from B's imap, flat)
    auto B_ghost_gcol = d2h(
        B->imap_.get_remote_global_idxs().get_const_flat_data(), n_ghost_B);

    // A's communication pattern
    auto coll_comm = this->row_gatherer_->get_collective_communicator();
    const auto send_size = static_cast<size_type>(coll_comm->get_send_size());
    const auto recv_size = static_cast<size_type>(coll_comm->get_recv_size());
    GKO_ASSERT(recv_size == n_ghost_A);

    auto send_idxs_h = d2h(this->row_gatherer_->get_const_send_idxs(), send_size);

    // ============================================================
    // Phase 1: Exchange B-row nnz counts using existing coll_comm.
    // (One int per row — same topology as SpMV but different type.)
    // ============================================================
    std::vector<int> send_nnz(send_size, 0);
    for (size_type i = 0; i < send_size; i++) {
        LocalIndexType r = send_idxs_h[i];
        send_nnz[i] = static_cast<int>(
            (B_d_rp[r + 1] - B_d_rp[r]) + (B_o_rp[r + 1] - B_o_rp[r]));
    }

    exec->synchronize();
    std::vector<int> recv_nnz(recv_size, 0);
    coll_comm->i_all_to_all_v(host, send_nnz.data(), recv_nnz.data()).wait();

    int total_send_data = 0;
    for (int n : send_nnz) total_send_data += n;
    int total_recv_data = 0;
    for (int n : recv_nnz) total_recv_data += n;

    // Offset into recv data buffer for ghost row k
    std::vector<int> recv_nnz_offs(recv_size + 1, 0);
    for (size_type k = 0; k < recv_size; k++)
        recv_nnz_offs[k + 1] = recv_nnz_offs[k] + recv_nnz[k];

    // ============================================================
    // Build per-rank row counts to determine per-rank data counts.
    // recv_row_counts[r]: rows of B received from rank r.
    // send_row_counts[r]: rows of B sent to rank r.
    // ============================================================
    std::vector<int> recv_row_counts(nproc, 0);
    {
        const auto& rim   = this->imap_.get_remote_global_idxs();
        const auto& rtids = this->imap_.get_remote_target_ids();
        const auto  n_seg = rim.get_segment_count();
        std::vector<int64> seg_offs(n_seg + 1);
        std::vector<comm_index_type> tids(n_seg);
        host->copy_from(exec,
                        n_seg + 1, rim.get_offsets().get_const_data(),
                        seg_offs.data());
        host->copy_from(exec,
                        n_seg, rtids.get_const_data(), tids.data());
        for (size_type s = 0; s < n_seg; s++)
            recv_row_counts[tids[s]] =
                static_cast<int>(seg_offs[s + 1] - seg_offs[s]);
    }

    std::vector<int> send_row_counts(nproc, 0);
    MPI_Alltoall(recv_row_counts.data(), 1, MPI_INT,
                 send_row_counts.data(), 1, MPI_INT, comm.get());

    // Cumulative per-rank row offsets into send_idxs (send topology is sorted
    // by ascending destination rank, matching NeighborhoodCommunicator ordering).
    std::vector<int> send_row_offs(nproc + 1, 0);
    for (int r = 0; r < nproc; r++)
        send_row_offs[r + 1] = send_row_offs[r] + send_row_counts[r];

    // Per-rank data counts for MPI_Alltoallv
    std::vector<int> send_dcnt(nproc, 0), recv_dcnt(nproc, 0);
    {
        const auto& rim   = this->imap_.get_remote_global_idxs();
        const auto& rtids = this->imap_.get_remote_target_ids();
        const auto  n_seg = rim.get_segment_count();
        std::vector<int64> seg_offs(n_seg + 1);
        std::vector<comm_index_type> tids(n_seg);
        host->copy_from(exec,
                        n_seg + 1, rim.get_offsets().get_const_data(),
                        seg_offs.data());
        host->copy_from(exec,
                        n_seg, rtids.get_const_data(), tids.data());
        for (size_type s = 0; s < n_seg; s++)
            for (int64 k = seg_offs[s]; k < seg_offs[s + 1]; k++)
                recv_dcnt[tids[s]] += recv_nnz[k];
    }
    for (int r = 0; r < nproc; r++)
        for (int i = send_row_offs[r]; i < send_row_offs[r + 1]; i++)
            send_dcnt[r] += send_nnz[i];

    std::vector<int> send_doffs(nproc + 1, 0), recv_doffs(nproc + 1, 0);
    for (int r = 0; r < nproc; r++) {
        send_doffs[r + 1] = send_doffs[r] + send_dcnt[r];
        recv_doffs[r + 1] = recv_doffs[r] + recv_dcnt[r];
    }

    // ============================================================
    // Phase 2: Exchange B-row data (global_col, value) via Alltoallv.
    // ============================================================
    std::vector<GlobalIndexType> send_cols(total_send_data);
    std::vector<ValueType>       send_vals(total_send_data);
    {
        int pos = 0;
        for (size_type i = 0; i < send_size; i++) {
            LocalIndexType r = send_idxs_h[i];
            for (auto j = B_d_rp[r]; j < B_d_rp[r + 1]; j++) {
                send_cols[pos] =
                    B_col_offset + static_cast<GlobalIndexType>(B_d_ci[j]);
                send_vals[pos] = B_d_vs[j];
                pos++;
            }
            for (auto j = B_o_rp[r]; j < B_o_rp[r + 1]; j++) {
                send_cols[pos] = B_ghost_gcol[B_o_ci[j]];
                send_vals[pos] = B_o_vs[j];
                pos++;
            }
        }
    }

    std::vector<GlobalIndexType> recv_cols(total_recv_data);
    std::vector<ValueType>       recv_vals(total_recv_data);

    auto gidx_mpi = experimental::mpi::type_impl<GlobalIndexType>::get_type();
    auto val_mpi  = experimental::mpi::type_impl<ValueType>::get_type();
    MPI_Alltoallv(send_cols.data(), send_dcnt.data(), send_doffs.data(), gidx_mpi,
                  recv_cols.data(), recv_dcnt.data(), recv_doffs.data(), gidx_mpi,
                  comm.get());
    MPI_Alltoallv(send_vals.data(), send_dcnt.data(), send_doffs.data(), val_mpi,
                  recv_vals.data(), recv_dcnt.data(), recv_doffs.data(), val_mpi,
                  comm.get());

    // ============================================================
    // Build A_ext = [A_diag | A_off] (n_local x n_local+n_ghost_A).
    // A_diag cols unchanged; A_off cols shifted by n_local.
    // Rows are already col-sorted within each block and the two blocks
    // cover disjoint col ranges, so the merged row is also sorted.
    // ============================================================
    const auto A_ext_nnz =
        A_diag->get_num_stored_elements() + A_off->get_num_stored_elements();
    std::vector<LocalIndexType> A_ext_rp(n_local + 1);
    std::vector<LocalIndexType> A_ext_ci(A_ext_nnz);
    std::vector<ValueType>      A_ext_vs(A_ext_nnz);
    {
        size_type pos = 0;
        for (size_type r = 0; r < n_local; r++) {
            A_ext_rp[r] = static_cast<LocalIndexType>(pos);
            for (auto j = A_d_rp[r]; j < A_d_rp[r + 1]; j++) {
                A_ext_ci[pos] = A_d_ci[j];
                A_ext_vs[pos] = A_d_vs[j];
                pos++;
            }
            for (auto j = A_o_rp[r]; j < A_o_rp[r + 1]; j++) {
                A_ext_ci[pos] =
                    static_cast<LocalIndexType>(n_local) + A_o_ci[j];
                A_ext_vs[pos] = A_o_vs[j];
                pos++;
            }
        }
        A_ext_rp[n_local] = static_cast<LocalIndexType>(pos);
    }

    // ============================================================
    // Collect unique global columns across all B_ext rows, build
    // global→local bijection for the SpGeMM column space.
    // ============================================================
    std::vector<GlobalIndexType> all_gcols;
    all_gcols.reserve(B_diag->get_num_stored_elements() +
                      B_off->get_num_stored_elements() + total_recv_data);
    for (size_type r = 0; r < n_local; r++) {
        for (auto j = B_d_rp[r]; j < B_d_rp[r + 1]; j++)
            all_gcols.push_back(
                B_col_offset + static_cast<GlobalIndexType>(B_d_ci[j]));
        for (auto j = B_o_rp[r]; j < B_o_rp[r + 1]; j++)
            all_gcols.push_back(B_ghost_gcol[B_o_ci[j]]);
    }
    for (int k = 0; k < total_recv_data; k++)
        all_gcols.push_back(recv_cols[k]);

    std::sort(all_gcols.begin(), all_gcols.end());
    all_gcols.erase(std::unique(all_gcols.begin(), all_gcols.end()),
                    all_gcols.end());
    const size_type m = all_gcols.size();  // unique columns in C

    // Map a global column to its local index in [0, m)
    auto to_local_col = [&](GlobalIndexType gc) -> LocalIndexType {
        return static_cast<LocalIndexType>(
            std::lower_bound(all_gcols.begin(), all_gcols.end(), gc) -
            all_gcols.begin());
    };

    // ============================================================
    // Build B_ext (n_local + n_ghost_A rows x m cols).
    // Rows 0..n_local-1   : local B rows.
    // Rows n_local..n_local+n_ghost_A-1: fetched ghost B rows.
    // Each row is sorted by local col index (required by cuSPARSE SpGeMM).
    // ============================================================
    const auto B_ext_nrows =
        static_cast<size_type>(n_local + n_ghost_A);
    const auto B_ext_nnz =
        B_diag->get_num_stored_elements() + B_off->get_num_stored_elements() +
        static_cast<size_type>(total_recv_data);
    std::vector<LocalIndexType> B_ext_rp(B_ext_nrows + 1);
    std::vector<LocalIndexType> B_ext_ci(B_ext_nnz);
    std::vector<ValueType>      B_ext_vs(B_ext_nnz);
    {
        size_type pos = 0;
        auto emit_sorted_row =
            [&](std::vector<std::pair<LocalIndexType, ValueType>>& row) {
                std::sort(row.begin(), row.end(),
                          [](const auto& a, const auto& b) {
                              return a.first < b.first;
                          });
                for (auto& entry : row) {
                    B_ext_ci[pos] = entry.first;
                    B_ext_vs[pos] = entry.second;
                    pos++;
                }
            };

        // Local B rows
        for (size_type r = 0; r < n_local; r++) {
            B_ext_rp[r] = static_cast<LocalIndexType>(pos);
            std::vector<std::pair<LocalIndexType, ValueType>> row;
            row.reserve((B_d_rp[r + 1] - B_d_rp[r]) +
                        (B_o_rp[r + 1] - B_o_rp[r]));
            for (auto j = B_d_rp[r]; j < B_d_rp[r + 1]; j++)
                row.emplace_back(
                    to_local_col(B_col_offset +
                                 static_cast<GlobalIndexType>(B_d_ci[j])),
                    B_d_vs[j]);
            for (auto j = B_o_rp[r]; j < B_o_rp[r + 1]; j++)
                row.emplace_back(to_local_col(B_ghost_gcol[B_o_ci[j]]),
                                 B_o_vs[j]);
            emit_sorted_row(row);
        }

        // Ghost B rows (received)
        for (size_type k = 0; k < n_ghost_A; k++) {
            B_ext_rp[n_local + k] = static_cast<LocalIndexType>(pos);
            std::vector<std::pair<LocalIndexType, ValueType>> row;
            row.reserve(recv_nnz[k]);
            for (int j = recv_nnz_offs[k]; j < recv_nnz_offs[k + 1]; j++)
                row.emplace_back(to_local_col(recv_cols[j]), recv_vals[j]);
            emit_sorted_row(row);
        }
        B_ext_rp[B_ext_nrows] = static_cast<LocalIndexType>(pos);
    }

    // Transfer A_ext and B_ext to device
    auto make_dev_csr =
        [&](size_type nrows, size_type ncols,
            std::vector<LocalIndexType>& rp, std::vector<LocalIndexType>& ci,
            std::vector<ValueType>& vs) -> std::shared_ptr<csr_type> {
            const auto nnz = ci.size();
            array<LocalIndexType> d_rp{exec, nrows + 1};
            array<LocalIndexType> d_ci{exec, nnz};
            array<ValueType>      d_vs{exec, nnz};
            exec->copy_from(host, nrows + 1, rp.data(), d_rp.get_data());
            if (nnz > 0) {
                exec->copy_from(host, nnz, ci.data(), d_ci.get_data());
                exec->copy_from(host, nnz, vs.data(), d_vs.get_data());
            }
            return share(csr_type::create(
                exec, dim<2>{nrows, ncols},
                std::move(d_vs), std::move(d_ci), std::move(d_rp)));
        };

    auto A_ext_dev = make_dev_csr(n_local, n_local + n_ghost_A,
                                  A_ext_rp, A_ext_ci, A_ext_vs);
    auto B_ext_dev = make_dev_csr(B_ext_nrows, m,
                                  B_ext_rp, B_ext_ci, B_ext_vs);

    // ============================================================
    // Local SpGeMM: C_tmp = A_ext * B_ext  (uses cuSPARSE on CUDA)
    // ============================================================
    auto C_tmp = csr_type::create(exec, dim<2>{n_local, m});
    A_ext_dev->apply(B_ext_dev.get(), C_tmp.get());

    // ============================================================
    // Split C_tmp into diag and off-diag blocks.
    // Diag: global col in [B_col_offset, B_col_offset + n_local_B).
    // Off-diag: all other global cols.
    // ============================================================
    const auto C_tmp_nnz = C_tmp->get_num_stored_elements();
    auto C_rp_h = d2h(C_tmp->get_const_row_ptrs(), n_local + 1);
    auto C_ci_h = d2h(C_tmp->get_const_col_idxs(), C_tmp_nnz);
    auto C_vs_h = d2h(C_tmp->get_const_values(),   C_tmp_nnz);

    std::vector<LocalIndexType>  c_d_rows, c_d_cols;
    std::vector<ValueType>       c_d_vals;
    std::vector<LocalIndexType>  c_o_rows;
    std::vector<GlobalIndexType> c_o_gcols;
    std::vector<ValueType>       c_o_vals;

    const GlobalIndexType B_col_end =
        B_col_offset + static_cast<GlobalIndexType>(n_local_B);
    for (size_type r = 0; r < n_local; r++) {
        for (auto j = C_rp_h[r]; j < C_rp_h[r + 1]; j++) {
            GlobalIndexType gc = all_gcols[C_ci_h[j]];
            auto lr = static_cast<LocalIndexType>(r);
            if (gc >= B_col_offset && gc < B_col_end) {
                c_d_rows.push_back(lr);
                c_d_cols.push_back(
                    static_cast<LocalIndexType>(gc - B_col_offset));
                c_d_vals.push_back(C_vs_h[j]);
            } else {
                c_o_rows.push_back(lr);
                c_o_gcols.push_back(gc);
                c_o_vals.push_back(C_vs_h[j]);
            }
        }
    }

    // Build C's index map from the off-diagonal global column indices.
    // Reconstruct B's column partition so index_map can classify columns.
    auto B_col_partition = gko::share(
        experimental::distributed::build_partition_from_local_size<
            LocalIndexType, GlobalIndexType>(exec, comm, n_local_B));

    array<GlobalIndexType> d_o_gcols{exec, c_o_gcols.size()};
    if (!c_o_gcols.empty())
        exec->copy_from(host, c_o_gcols.size(), c_o_gcols.data(),
                        d_o_gcols.get_data());

    auto c_imap =
        experimental::distributed::index_map<LocalIndexType, GlobalIndexType>(
            exec, B_col_partition, my_rank, d_o_gcols);

    // Map ghost global cols to local off-diagonal indices
    auto c_o_local = c_imap.map_to_local(
        d_o_gcols, experimental::distributed::index_space::non_local);

    // Pull c_o_local back to host for CSR construction
    const auto n_o_entries = c_o_gcols.size();
    std::vector<LocalIndexType> c_o_cols_h(n_o_entries);
    if (n_o_entries > 0)
        host->copy_from(exec, n_o_entries, c_o_local.get_const_data(),
                        c_o_cols_h.data());

    // Build C_diag and C_off CSR matrices
    const auto C_n_local_rows =
        static_cast<size_type>(n_local);
    const auto C_n_off_cols =
        static_cast<size_type>(c_imap.get_non_local_size());

    auto C_diag_csr = share(coo_to_csr<ValueType, LocalIndexType>(
        exec, host, C_n_local_rows, n_local_B,
        c_d_rows, c_d_cols, c_d_vals));
    auto C_off_csr  = share(coo_to_csr<ValueType, LocalIndexType>(
        exec, host, C_n_local_rows, C_n_off_cols,
        c_o_rows, c_o_cols_h, c_o_vals));

    // Build RowGatherer for C
    auto c_rg = RowGatherer<LocalIndexType>::create(
        exec,
        C->row_gatherer_->get_collective_communicator()
            ->create_with_same_type(comm, &c_imap),
        c_imap);

    // Write output C
    const size_type C_global_rows = this->get_size()[0];
    const size_type C_global_cols = B->get_size()[1];
    C->set_size(dim<2>{C_global_rows, C_global_cols});
    C->imap_         = std::move(c_imap);
    C->diag_mtx_     = std::move(C_diag_csr);
    C->off_diag_mtx_ = std::move(C_off_csr);
    C->row_gatherer_ = std::move(c_rg);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    // SpGeMM: C = A * B where B and C are distributed matrices (same type)
    if (auto b_mat = dynamic_cast<const Matrix*>(b)) {
        if (auto x_mat = dynamic_cast<Matrix*>(x)) {
            this->apply_spgemm(b_mat, x_mat);
            return;
        }
    }
    distributed::mixed_precision_dispatch_real_complex<ValueType>(
        [this](const auto dense_b, auto dense_x) {
            using x_value_type =
                typename std::decay_t<decltype(*dense_x)>::value_type;
            using b_value_type =
                typename std::decay_t<decltype(*dense_b)>::value_type;
            auto x_exec = dense_x->get_executor();
            auto local_x = gko::matrix::Dense<x_value_type>::create(
                x_exec, dense_x->get_local_vector()->get_size(),
                gko::make_array_view(
                    x_exec,
                    dense_x->get_local_vector()->get_num_stored_elements(),
                    dense_x->get_local_values()),
                dense_x->get_local_vector()->get_stride());

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto [recv_vector, host_recv_vector] =
                init_recv_buffers<b_value_type>(
                    exec, row_gatherer_.get(), dense_b->get_size()[1],
                    recv_buffer_, host_recv_buffer_);
            auto recv_ptr = mpi::requires_host_buffer(exec, comm)
                                ? host_recv_vector.get()
                                : recv_vector.get();
            if (dense_b->get_executor() ==
                dense_b->get_executor()->get_master()) {
                // reference and omp executor does not have event, so we still
                // submit the mpi first.
                auto req = this->row_gatherer_->apply_async(dense_b, recv_ptr);
                diag_mtx_->apply(dense_b->get_local_vector(), local_x);
                req.wait();
            } else {
                // we use event here such that we can submit spmv job first
                // without waiting for synchronization from the row gatherer.
                auto ev = this->row_gatherer_->apply_prepare(dense_b);
                diag_mtx_->apply(dense_b->get_local_vector(), local_x);
                auto req =
                    this->row_gatherer_->apply_finalize(dense_b, recv_ptr, ev);
                req.wait();
            }

            if (recv_ptr != recv_vector.get()) {
                recv_vector->copy_from(host_recv_vector);
            }
            if (auto coo = std::dynamic_pointer_cast<
                    const ::gko::matrix::Coo<ValueType, LocalIndexType>>(
                    off_diag_mtx_)) {
                coo->apply2(recv_vector->get_local_vector(), local_x);
            } else {
                off_diag_mtx_->apply(
                    one_scalar_.template get<ValueType>().get(),
                    recv_vector->get_local_vector(),
                    one_scalar_.template get<x_value_type>().get(), local_x);
            }
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    distributed::mixed_precision_dispatch_real_complex<ValueType>(
        [this, alpha, beta](const auto dense_b, auto dense_x) {
            using x_value_type =
                typename std::decay_t<decltype(*dense_x)>::value_type;
            using b_value_type =
                typename std::decay_t<decltype(*dense_b)>::value_type;
            const auto x_exec = dense_x->get_executor();
            auto local_alpha = gko::make_temporary_conversion<ValueType>(alpha);
            auto local_beta =
                gko::make_temporary_conversion<x_value_type>(beta);
            auto local_x = gko::matrix::Dense<x_value_type>::create(
                x_exec, dense_x->get_local_vector()->get_size(),
                gko::make_array_view(
                    x_exec,
                    dense_x->get_local_vector()->get_num_stored_elements(),
                    dense_x->get_local_values()),
                dense_x->get_local_vector()->get_stride());

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto [recv_vector, host_recv_vector] =
                init_recv_buffers<b_value_type>(
                    exec, row_gatherer_.get(), dense_b->get_size()[1],
                    recv_buffer_, host_recv_buffer_);
            auto recv_ptr = mpi::requires_host_buffer(exec, comm)
                                ? host_recv_vector.get()
                                : recv_vector.get();
            if (dense_b->get_executor() ==
                dense_b->get_executor()->get_master()) {
                // reference and omp executor does not have event, so we still
                // submit the mpi first.
                auto req = this->row_gatherer_->apply_async(dense_b, recv_ptr);
                diag_mtx_->apply(local_alpha.get(), dense_b->get_local_vector(),
                                 local_beta.get(), local_x);
                req.wait();
            } else {
                // we use event here such that we can submit spmv job first
                // without waiting for synchronization from the row gatherer.
                auto ev = this->row_gatherer_->apply_prepare(dense_b);
                diag_mtx_->apply(local_alpha.get(), dense_b->get_local_vector(),
                                 local_beta.get(), local_x);
                auto req =
                    this->row_gatherer_->apply_finalize(dense_b, recv_ptr, ev);
                req.wait();
            }

            if (recv_ptr != recv_vector.get()) {
                recv_vector->copy_from(host_recv_vector);
            }
            if (auto coo = std::dynamic_pointer_cast<
                    const ::gko::matrix::Coo<ValueType, LocalIndexType>>(
                    off_diag_mtx_)) {
                coo->apply2(local_alpha.get(), recv_vector->get_local_vector(),
                            local_x);
            } else {
                off_diag_mtx_->apply(
                    local_alpha.get(), recv_vector->get_local_vector(),
                    one_scalar_.template get<x_value_type>().get(), local_x);
            }
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::col_scale(
    ptr_param<const global_vector_type> scaling_factors)
{
    GKO_ASSERT_CONFORMANT(this, scaling_factors.get());
    GKO_ASSERT_EQ(scaling_factors->get_size()[1], 1);
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    size_type n_local_cols = diag_mtx_->get_size()[1];
    size_type n_off_diag_cols = off_diag_mtx_->get_size()[1];

    std::unique_ptr<global_vector_type> scaling_factors_single_stride;
    auto scaling_stride = scaling_factors->get_stride();
    if (scaling_stride != 1) {
        scaling_factors_single_stride = global_vector_type::create(exec, comm);
        scaling_factors_single_stride->copy_from(scaling_factors.get());
    }
    const global_vector_type* scaling_factors_ptr =
        scaling_stride == 1 ? scaling_factors.get()
                            : scaling_factors_single_stride.get();
    const auto scale_diag = gko::matrix::Diagonal<ValueType>::create_const(
        exec, n_local_cols,
        make_const_array_view(exec, n_local_cols,
                              scaling_factors_ptr->get_const_local_values()));

    auto [recv_vector, host_recv_vector] = init_recv_buffers<ValueType>(
        exec, row_gatherer_.get(), scaling_factors->get_size()[1], recv_buffer_,
        host_recv_buffer_);
    auto recv_ptr = mpi::requires_host_buffer(exec, comm)
                        ? host_recv_vector.get()
                        : recv_vector.get();

    if (scaling_factors->get_executor() ==
        scaling_factors->get_executor()->get_master()) {
        // reference and omp executor does not have event, so we still
        // submit the mpi first.
        auto req =
            this->row_gatherer_->apply_async(scaling_factors_ptr, recv_ptr);
        scale_diag->rapply(diag_mtx_, diag_mtx_);
        req.wait();
    } else {
        // we use event here such that we can submit diag matrix scaling job
        // first without waiting for synchronization from the row gatherer.
        auto ev = this->row_gatherer_->apply_prepare(scaling_factors_ptr);
        scale_diag->rapply(diag_mtx_, diag_mtx_);
        auto req = this->row_gatherer_->apply_finalize(scaling_factors_ptr,
                                                       recv_ptr, ev);
        req.wait();
    }
    if (n_off_diag_cols > 0) {
        if (recv_ptr != recv_vector.get()) {
            recv_vector->copy_from(host_recv_vector);
        }
        const auto off_diag_scale_diag =
            gko::matrix::Diagonal<ValueType>::create_const(
                exec, n_off_diag_cols,
                make_const_array_view(exec, n_off_diag_cols,
                                      recv_vector->get_const_local_values()));
        off_diag_scale_diag->rapply(off_diag_mtx_, off_diag_mtx_);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::row_scale(
    ptr_param<const global_vector_type> scaling_factors)
{
    GKO_ASSERT_EQUAL_ROWS(this, scaling_factors.get());
    GKO_ASSERT_EQ(scaling_factors->get_size()[1], 1);
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    size_type n_local_rows = diag_mtx_->get_size()[0];
    std::unique_ptr<global_vector_type> scaling_factors_single_stride;
    auto stride = scaling_factors->get_stride();
    if (stride != 1) {
        scaling_factors_single_stride = global_vector_type::create(exec, comm);
        scaling_factors_single_stride->copy_from(scaling_factors.get());
    }
    const auto scale_values =
        stride == 1 ? scaling_factors->get_const_local_values()
                    : scaling_factors_single_stride->get_const_local_values();
    const auto scale_diag = gko::matrix::Diagonal<ValueType>::create_const(
        exec, n_local_rows,
        make_const_array_view(exec, n_local_rows, scale_values));

    scale_diag->apply(diag_mtx_, diag_mtx_);
    scale_diag->apply(off_diag_mtx_, off_diag_mtx_);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(const Matrix& other)
    : LinOp{other.get_executor()},
      DistributedBase{other.get_communicator()},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          other.get_executor(), other.get_communicator())},
      imap_(other.get_executor()),
      one_scalar_(other.get_executor(), 1.0)
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    Matrix&& other) noexcept
    : LinOp{other.get_executor()},
      DistributedBase{other.get_communicator()},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          other.get_executor(), other.get_communicator())},
      imap_(other.get_executor()),
      one_scalar_(other.get_executor(), 1.0)
{
    *this = std::move(other);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>&
Matrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(
    const Matrix& other)
{
    if (this != &other) {
        GKO_ASSERT_EQ(other.get_communicator().size(),
                      this->get_communicator().size());
        this->set_size(other.get_size());
        as<Cloneable>(diag_mtx_)->copy_from(as<Cloneable>(other.diag_mtx_));
        as<Cloneable>(off_diag_mtx_)
            ->copy_from(as<Cloneable>(other.off_diag_mtx_));
        as<Cloneable>(row_gatherer_)
            ->copy_from(as<Cloneable>(other.row_gatherer_));
        imap_ = other.imap_;
    }
    return *this;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>&
Matrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(Matrix&& other)
{
    if (this != &other) {
        GKO_ASSERT_EQ(other.get_communicator().size(),
                      this->get_communicator().size());
        this->set_size(other.get_size());
        other.set_size({});
        as<Cloneable>(diag_mtx_)->move_from(as<Cloneable>(other.diag_mtx_));
        as<Cloneable>(off_diag_mtx_)
            ->move_from(as<Cloneable>(other.off_diag_mtx_));
        as<Cloneable>(row_gatherer_)
            ->move_from(as<Cloneable>(other.row_gatherer_));
        imap_ = std::move(other.imap_);
    }
    return *this;
}


#define GKO_DECLARE_DISTRIBUTED_MATRIX(ValueType, LocalIndexType, \
                                       GlobalIndexType)           \
    class Matrix<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_MATRIX);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
