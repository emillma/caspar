// CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
// This source code is under the Apache 2.0 license found in the LICENSE file.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// TODO improve performance of get_opcounts and get_ops by presorting

extern "C" __global__ void
get_n_matches(const int *const crows,
              const int *const cols,
              int *const out_n_match_per_row,
              int n_rows)
{
    const auto gtrank = cg::this_grid().thread_rank();
    if (gtrank >= n_rows)
        return;
    int to_test = n_rows / 2 - (n_rows % 2 ? 0 : gtrank >= (n_rows / 2));

    int a_start = crows[gtrank];
    int a_end = crows[gtrank + 1];
    int total = 0;

    for (int offset = 0; offset <= to_test; offset++)
    {
        int a_ptr = a_start;
        int a_elem = cols[a_ptr];

        int b_start = crows[(gtrank + offset) % n_rows];
        int b_end = crows[(gtrank + offset + 1) == n_rows ? n_rows : (gtrank + offset + 1) % n_rows];

        int b_ptr = b_start;
        int b_elem = cols[b_ptr];

        while (a_ptr < a_end && b_ptr < b_end)
        {
            if (a_elem == b_elem)
            {
                total++;
                break;
            }
            if (a_elem <= b_elem && ++a_ptr < a_end)
                a_elem = cols[a_ptr];
            else if (++b_ptr < b_end)
                b_elem = cols[b_ptr];
        }
    }
    out_n_match_per_row[gtrank] = total;
}
extern "C" __global__ void
get_n_ops(const int *const crows,
          const int *const cols,
          const int *const n_match_per_row_cum,
          int *const out_right_rows,
          int *const out_n_ops_per_match,
          int n_rows)
{
    const auto gtrank = cg::this_grid().thread_rank();
    if (gtrank >= n_rows)
        return;
    int to_test = n_rows / 2 - (n_rows % 2 ? 0 : gtrank >= (n_rows / 2));

    int a_start = crows[gtrank];
    int a_end = crows[gtrank + 1];
    int out_idx = n_match_per_row_cum[gtrank];
    for (int offset = 0; offset <= to_test; offset++)
    {
        int ops_per_match = 0;
        int any = false;
        int a_ptr = a_start;
        int a_elem = cols[a_ptr];

        int b_start = crows[(gtrank + offset) % n_rows];
        int b_end = crows[(gtrank + offset + 1) == n_rows ? n_rows : (gtrank + offset + 1) % n_rows];

        int b_ptr = b_start;
        int b_elem = cols[b_ptr];

        while (a_ptr < a_end && b_ptr < b_end)
        {
            if (a_elem == b_elem)
            {
                ops_per_match++;
                any = true;
            }
            if (a_elem <= b_elem && ++a_ptr < a_end)
                a_elem = cols[a_ptr];
            else if (++b_ptr < b_end)
                b_elem = cols[b_ptr];
        }
        if (any)
        {
            out_right_rows[out_idx] = (gtrank + offset) % n_rows;
            out_n_ops_per_match[out_idx++] = ops_per_match;
        }
    }
}
extern "C" __global__ void
get_ops(const int *const crows,
        const int *const cols,
        const int2 *const res_idxs,
        const int *const n_ops_per_match_cum,
        int2 *const out_ops,
        int n_match_total)
{
    const auto gtrank = cg::this_grid().thread_rank();
    if (gtrank >= n_match_total)
        return;

    const int2 res_idx = res_idxs[gtrank];

    int a_ptr = crows[res_idx.x];
    int a_elem = cols[a_ptr];
    int a_end = crows[res_idx.x + 1];

    int b_ptr = crows[res_idx.y];
    int b_elem = cols[b_ptr];
    int b_end = crows[res_idx.y + 1];

    int out_idx = n_ops_per_match_cum[gtrank];

    while (a_ptr < a_end && b_ptr < b_end)
    {
        if (a_elem == b_elem)
        {
            out_ops[out_idx++] = {a_ptr, b_ptr};
        }
        if (a_elem <= b_elem && ++a_ptr < a_end)
            a_elem = cols[a_ptr];
        else if (++b_ptr < b_end)
            b_elem = cols[b_ptr];
    }
}
extern "C" __global__ void
evaluate(

    const float *const values,

    const int *const n_ops_per_match_cum,
    const int2 *const ops,
    float *const out_values,
    const int *const out_values_ord,
    int n_match_total)
{
    const auto gtrank = cg::this_grid().thread_rank();
    if (gtrank >= n_match_total)
        return;
    float total = 0;
    int op_start = n_ops_per_match_cum[gtrank];
    int op_end = n_ops_per_match_cum[gtrank + 1];
    for (int op_idx = op_start; op_idx < op_end; op_idx++)
    {
        int a_ptr = ops[op_idx].x;
        int b_ptr = ops[op_idx].y;
        total += values[a_ptr] * values[b_ptr];
    }
    out_values[out_values_ord[gtrank]] = total;
}

// extern "C" __global__ void
// evaluate(

//     const int *const c_rows_outer,
//     const int *const cols_outer,
//     const int *const c_rows_inner_full,
//     const int *const cols_inner_full,
//     const int *const c_rows_inner_diag,
//     const int *const cols_inner_diag,
//     const int dim,
//     const int *const c_idx_out,
//     int2 *const idx_out,
//     int *const lower_out,
//     int *const diag_out,
//     const int n_rows)
// {
//     const auto gtrank = cg::this_grid().thread_rank();
//     if (gtrank >= n_rows)
//         return;

//     const int n_diag = c_rows_inner_diag[dim];

//     int idx_out_idx = c_idx_out[gtrank];
//     int lower_out_idx = idx_out_idx - n_diag * gtrank;
//     int diag_out_idx = n_diag * gtrank;

//     const int outer_col_start = c_rows_outer[gtrank];
//     const int outer_col_end = c_rows_outer[gtrank + 1];

//     for (int sub_row = 0; sub_row < dim; sub_row++)
//     {
//         for (int outer_col = outer_col_start; outer_col < outer_col_end; outer_col++)
//         {

//         }
//     }
// }
