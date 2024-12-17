# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
import torch
from src import torch_tools as tt

from . import get_n_matches, get_n_ops, get_ops, evaluate


class MTM:
    """res = mat @ mat.t()"""

    def __init__(self, indices: torch.Tensor, shape: tuple[int, int]):
        # Check if indices are a column-major sorted 2D tensor
        # assert (indices.view(torch.long).sort()[0] == indices.view(torch.long)).all()

        t_shape = (shape[1], shape[0])
        t_crows = tt.compress_index(indices[:, 1], t_shape[0])
        t_cols = indices[:, 0].contiguous()
        # self.vals = mat.values()

        assert t_crows.dtype == torch.int32
        assert t_cols.dtype == torch.int32
        # assert self.vals.dtype == torch.float32

        self.n_rows, n_cols = t_shape

        n_match_per_row = tt.empty_i32(self.n_rows) * 0
        get_n_matches.launch_n(self.n_rows)(
            t_crows,
            t_cols,
            n_match_per_row,
            self.n_rows,
        )
        n_match_per_row_cum = tt.cumsum_inc_i32(n_match_per_row)
        self.res_nnz = n_match_per_row_cum[-1].item()
        left_rows = torch.repeat_interleave(
            tt.arange_i32(self.n_rows),
            n_match_per_row,
        )
        right_rows = tt.empty_i32(self.res_nnz)
        self.n_ops_per_match = tt.empty_i32(self.res_nnz)
        get_n_ops.launch_n(self.n_rows)(
            t_crows,
            t_cols,
            n_match_per_row_cum,
            right_rows,
            self.n_ops_per_match,
            self.n_rows,
        )
        tmp = torch.stack([left_rows, right_rows], 1).sort(1, descending=True).values
        res_idx, res_ord = tmp.view(torch.long).ravel().sort()
        self.res_idx = res_idx.view(torch.int32).reshape(-1, 2)
        self.n_ops_per_match = self.n_ops_per_match[res_ord]
        n_ops_per_match_cum = tt.cumsum_inc_i32(self.n_ops_per_match)
        self.ops = tt.empty_i32(n_ops_per_match_cum[-1], 2)
        get_ops.launch_n(self.res_nnz)(
            t_crows,
            t_cols,
            self.res_idx,
            n_ops_per_match_cum,
            self.ops,
            self.res_nnz,
        )

        self.ops_ord = torch.repeat_interleave(
            tt.arange_i32(self.res_idx.shape[0]),
            self.n_ops_per_match,
        )

        eq = self.res_idx[:, 0] == self.res_idx[:, 1]
        eq_ops = eq[self.ops_ord]
        self.diag_idx = self.res_idx[eq]
        self.n_diag = self.diag_idx.shape[0]
        diag_ops = self.ops.T[:, eq_ops]
        diag_ord = self.ops_ord[eq_ops]
        self.diag_vec_idx = [*diag_ops, diag_ord]
        self.diag_nops = diag_ops.shape[1]

        neq_ops = ~eq_ops
        self.lower_idx = self.res_idx[~eq]
        self.n_lower = self.lower_idx.shape[0]
        lower_ops = self.ops.T[:, neq_ops]
        lower_ord = self.ops_ord[neq_ops]
        self.lower_vec_idx = [*lower_ops, lower_ord]
        self.lower_nops = lower_ops.shape[1]

        # res_idxT = self.res_idx.T.contiguous()
        # self.res_matmul_vec_idx = [res_ord, res_idxT[1], res_idxT[0]]

    def find_ord(self, target: torch.Tensor):
        idx_long = self.res_idx.view(torch.long).ravel()
        target_long = target.view(torch.long).ravel()

        hits = torch.searchsorted(idx_long, target_long, out_int32=True)
        assert idx_long[hits].equal(target_long)
        return hits
