// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <mpi.h>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>

#include "core/test/utils.hpp"
#include "test/utils/mpi/common_fixture.hpp"


#if GINKGO_DPCPP_SINGLE_MODE
using solver_value_type = float;
#else
using solver_value_type = double;
#endif  // GINKGO_DPCPP_SINGLE_MODE


template <typename ValueLocalGlobalIndexType>
class Pmis : public CommonMpiTestFixture {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        2, decltype(ValueLocalGlobalIndexType())>::type;
    using dist_mtx_type =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;
    using local_matrix_type = gko::matrix::Csr<value_type, local_index_type>;
    using Partition =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using matrix_data = gko::matrix_data<value_type, global_index_type>;
    using pmis = gko::multigrid::Pmis<value_type, local_index_type>;

    Pmis()
        : size{8, 8}, mat_input{size, {{0, 0, 5},  {0, 1, -1}, {1, 0, -1},
                                       {1, 1, 5},  {2, 2, 5},  {3, 3, 5},
                                       {4, 4, 5},  {4, 6, -2}, {5, 5, 5},
                                       {5, 7, -2}, {6, 4, -2}, {6, 6, 5},
                                       {7, 5, -2}, {7, 7, 5},  {0, 2, -3},
                                       {0, 4, 1},  {0, 5, 2},  {0, 6, 3},
                                       {1, 3, -4}, {1, 5, 4},  {1, 6, 5},
                                       {1, 7, 6},  {2, 0, -3}, {2, 5, -1},
                                       {2, 6, -2}, {3, 1, -4}, {3, 7, -5},
                                       {4, 0, 1},  {5, 0, 2},  {5, 1, 4},
                                       {5, 2, -1}, {6, 0, 3},  {6, 1, 5},
                                       {6, 2, -2}, {7, 1, 6},  {7, 3, -5}}}
    {
        row_part = Partition::build_from_contiguous(
            exec, gko::array<global_index_type>(
                      exec, I<global_index_type>{0, 2, 4, 8}));

        dist_mat = dist_mtx_type::create(exec, comm);
        dist_mat->read_distributed(mat_input, row_part);
    }

    void SetUp() override { ASSERT_EQ(comm.size(), 3); }

    gko::dim<2> size;
    std::shared_ptr<Partition> row_part;

    gko::matrix_data<value_type, global_index_type> mat_input;

    std::shared_ptr<dist_mtx_type> dist_mat;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(Pmis, CanGenerateFromDistributedMatrix)
{
    using pmis = typename TestFixture::pmis;
    using dist_mtx_type = typename TestFixture::dist_mtx_type;
    using local_matrix_type = typename TestFixture::local_matrix_type;
    auto rank = this->comm.rank();
    // PMIS uses random weights so coarsening is non-deterministic.
    // Retry a few times to avoid intermittent failures.
    std::shared_ptr<pmis> result;
    for (int attempt = 0; attempt < 5; ++attempt) {
        try {
            result = gko::share(pmis::build().on(this->exec)->generate(
                this->dist_mat));
            break;
        } catch (const gko::InvalidStateError&) {
            if (attempt == 4) throw;
        }
    }

    auto coarse = gko::as<dist_mtx_type>(result->get_coarse_op());
    auto prolong_op = result->get_prolong_op();
    auto restrict_op = result->get_restrict_op();

    // Coarse global size must be <= fine global size (no enlargement)
    EXPECT_LE(coarse->get_size()[0], this->dist_mat->get_size()[0]);
    // Coarse must be square
    EXPECT_EQ(coarse->get_size()[0], coarse->get_size()[1]);
    // Prolongation: rows = fine_global, cols = coarse_global
    EXPECT_EQ(prolong_op->get_size()[0], this->dist_mat->get_size()[0]);
    EXPECT_EQ(prolong_op->get_size()[1], coarse->get_size()[0]);
    // Restriction: rows = coarse_global, cols = fine_global
    EXPECT_EQ(restrict_op->get_size()[0], coarse->get_size()[0]);
    EXPECT_EQ(restrict_op->get_size()[1], this->dist_mat->get_size()[0]);
    // Local coarse diag block must have correct row count
    auto diag = gko::as<local_matrix_type>(coarse->get_diag_matrix());
    auto off_diag = gko::as<local_matrix_type>(coarse->get_off_diag_matrix());
    EXPECT_EQ(diag->get_size()[0], diag->get_size()[1]);
    EXPECT_EQ(off_diag->get_size()[0], diag->get_size()[0]);
}
