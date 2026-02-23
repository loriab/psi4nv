/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2025 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

/*!
** \file
** \brief Diagnoalize a symmetrix square matrix
** \ingroup CIOMR
*/

#include "psi4/libqt/qt.h"
#include "libciomr.h"
#include <vector>

namespace psi {
/*!
** DSYEV_ascending(): diagonalize a symmetric square matrix ('array') using
** LAPACK's divide-and-conquer DSYEVD for better performance on large matrices.
**
** \param n      = number of rows (and columns)
** \param array  = matrix to diagonalize (2D row major array)
** \param e_vals = array to hold eigenvalues (returned in ascending order)
** \param e_vecs = (optional) matrix of eigenvectors (2D row major array, one column for each eigvector)
**
** \ingroup CIOMR
*/
[[nodiscard]] int DSYEV_ascending(const int N, const double* const* const array, double* e_vals,
                                  double* const* const e_vecs /* = nullptr*/) {
    // Flatten row-major input to column-major 1D array for LAPACK.
    std::vector<double> tmp_matrix(N * N);
    for (int64_t i = 0, ij = 0; i < N; i++) {
        for (int64_t j = 0; j < N; j++, ij++) {
            tmp_matrix[ij] = array[j][i];
        }
    }
    const char jobtype = (e_vecs != nullptr) ? 'V' : 'N';

    // Query DSYEVD for optimal workspace sizes.
    double work_query;
    int iwork_query;
    C_DSYEVD(jobtype, 'U', N, tmp_matrix.data(), N, e_vals, &work_query, -1, &iwork_query, -1);
    const int lwork = static_cast<int>(work_query);
    const int liwork = iwork_query;

    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
    const int info = C_DSYEVD(jobtype, 'U', N, tmp_matrix.data(), N, e_vals, work.data(), lwork, iwork.data(), liwork);

    if ((info == 0) && (e_vecs != nullptr)) {
        // Column-major eigenvectors back to row-major 2D array.
        for (int64_t j = 0, ij = 0; j < N; j++) {
            for (int64_t i = 0; i < N; i++, ij++) {
                e_vecs[i][j] = tmp_matrix[ij];
            }
        }
    }
    return info;
}
}  // namespace psi
