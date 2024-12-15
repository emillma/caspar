# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from ctypes import c_double, cdll, pointer, c_void_p, c_int64, c_int32, cast, sizeof
import torch

CUDSS_CONFIG_REORDERING_ALG = 0
CUDSS_CONFIG_FACTORIZATION_ALG = 1
CUDSS_CONFIG_SOLVE_ALG = 2
CUDSS_CONFIG_MATCHING_TYPE = 3
CUDSS_CONFIG_SOLVE_MODE = 4
CUDSS_CONFIG_IR_N_STEPS = 5
CUDSS_CONFIG_IR_TOL = 6
CUDSS_CONFIG_PIVOT_TYPE = 7
CUDSS_CONFIG_PIVOT_THRESHOLD = 8
CUDSS_CONFIG_PIVOT_EPSILON = 9
CUDSS_CONFIG_MAX_LU_NNZ = 10
CUDSS_CONFIG_HYBRID_MODE = 11
CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT = 12
CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY = 13

CUDSS_DATA_INFO = 0
CUDSS_DATA_LU_NNZ = 1
CUDSS_DATA_NPIVOTS = 2
CUDSS_DATA_INERTIA = 3
CUDSS_DATA_PERM_REORDER_ROW = 4
CUDSS_DATA_PERM_REORDER_COL = 5
CUDSS_DATA_PERM_ROW = 6
CUDSS_DATA_PERM_COL = 7
CUDSS_DATA_DIAG = 8
CUDSS_DATA_USER_PERM = 9
CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN = 10
CUDSS_DATA_COMM = 11


CUDSS_PHASE_ANALYSIS = 1
CUDSS_PHASE_FACTORIZATION = 2
CUDSS_PHASE_REFACTORIZATION = 4
CUDSS_PHASE_SOLVE = 8
CUDSS_PHASE_SOLVE_FWD = 16
CUDSS_PHASE_SOLVE_DIAG = 32
CUDSS_PHASE_SOLVE_BWD = 64


CUDA_R_32F = 0
CUDA_R_64F = 1
CUDA_R_32I = 10
CUDA_R_64I = 24

CUDSS_BASE_ZERO = 0

CUDSS_MTYPE_GENERAL = 0
CUDSS_MTYPE_SYMMETRIC = 1
CUDSS_MTYPE_HERMITIAN = 2
CUDSS_MTYPE_SPD = 3
CUDSS_MTYPE_HPD = 4

CUDSS_LAYOUT_COL_MAJOR = 0
CUDSS_LAYOUT_ROW_MAJOR = 1

CUDSS_MVIEW_FULL = 0
CUDSS_MVIEW_LOWER = 1
CUDSS_MVIEW_UPPER = 2

CUDSS_PIVOT_COL = 0
CUDSS_PIVOT_ROW = 1
CUDSS_PIVOT_NONE = 2

CUDSS_ALG_DEFAULT = 0
CUDSS_ALG_1 = 1
CUDSS_ALG_2 = 2
CUDSS_ALG_3 = 3

dtype_map = {
    torch.float32: CUDA_R_32F,
    torch.float64: CUDA_R_64F,
    torch.int32: CUDA_R_32I,
    torch.int64: CUDA_R_64I,
}

torch.cuda.init()
torch.cuda.synchronize()

cudss = cdll.LoadLibrary("libcudss.so")

handle = c_void_p()
assert not cudss.cudssCreate(pointer(handle))


def create_symmetric_csr(csr: torch.Tensor):
    out = c_void_p()

    assert csr.crow_indices().dtype == csr.col_indices().dtype
    assert csr.crow_indices().is_contiguous() and csr.crow_indices().is_cuda
    assert csr.col_indices().is_contiguous() and csr.col_indices().is_cuda
    assert csr.values().is_contiguous() and csr.values().is_cuda

    assert not cudss.cudssMatrixCreateCsr(
        pointer(out),
        csr.shape[0],
        csr.shape[1],
        csr._nnz(),
        c_void_p(csr.crow_indices().data_ptr()),
        c_void_p(),
        c_void_p(csr.col_indices().data_ptr()),
        c_void_p(csr.values().data_ptr()),
        dtype_map[csr.crow_indices().dtype],
        dtype_map[csr.dtype],
        CUDSS_MTYPE_SYMMETRIC,
        CUDSS_MVIEW_LOWER,
        CUDSS_BASE_ZERO,
    )
    return out


def create_dn(ten: torch.Tensor):
    assert ten.is_contiguous() and ten.is_cuda and ten.stride()[0] == 1
    out = c_void_p()
    assert not cudss.cudssMatrixCreateDn(
        pointer(out),
        ten.shape[0],
        ten.shape[1] if ten.dim() > 1 else 1,
        ten.shape[0],
        c_void_p(ten.data_ptr()),
        dtype_map[ten.dtype],
        CUDSS_LAYOUT_COL_MAJOR,
    )
    return out


class ProblemDSS:

    def __init__(
        self,
        A: torch.Tensor,  # csr
        x: torch.Tensor,  # dn vector
        b: torch.Tensor,  # dn vector
        order: torch.Tensor = None,
        alg_all=CUDSS_ALG_3,
        reorder_alg=None,
        fact_alg=None,
        solve_alg=None,
    ):

        self.mats_torch = A, x, b
        self.A = A
        self.x = x
        self.b = b
        self.indextype = A.crow_indices().dtype
        self.mats = [create_symmetric_csr(A), create_dn(x), create_dn(b)]

        self.config = c_void_p()
        cudss.cudssConfigCreate(pointer(self.config))

        self.set_config(CUDSS_CONFIG_REORDERING_ALG, reorder_alg or alg_all)
        self.set_config(CUDSS_CONFIG_FACTORIZATION_ALG, fact_alg or alg_all)
        self.set_config(CUDSS_CONFIG_SOLVE_ALG, solve_alg or alg_all)
        # self.set_config(CUDSS_CONFIG_PIVOT_TYPE, CUDSS_PIVOT_ROW)
        # self.set_config(CUDSS_CONFIG_PIVOT_THRESHOLD, 1.0, c_double)
        # self.set_config(CUDSS_CONFIG_PIVOT_EPSILON, 1e-5, c_double)
        # self.get_config(CUDSS_CONFIG_PIVOT_EPSILON, torch.float64)
        self.data = c_void_p()

        cudss.cudssDataCreate(handle, pointer(self.data))

        if order is not None:
            assert order.dtype == A.crow_indices().dtype
            assert order.shape[0] == A.shape[0]
            self.set_data(CUDSS_DATA_USER_PERM, order)

        assert not cudss.cudssExecute(
            handle,
            CUDSS_PHASE_ANALYSIS,
            self.config,
            self.data,
            *self.mats,
        )
        self.perm = self.get_data(CUDSS_DATA_PERM_REORDER_ROW, torch.int32)
        self.lu_nnz = self.get_data(CUDSS_DATA_LU_NNZ, torch.long)

    def set_config(self, field, value, dtype=c_int32):
        size = sizeof(dtype)
        assert not cudss.cudssConfigSet(self.config, field, pointer(dtype(value)), size)

    def get_config(self, field, dtype: torch.dtype = torch.int32):
        size_p = pointer(c_int64())
        assert not cudss.cudssConfigGet(self.config, field, c_void_p(), 0, size_p)
        out = torch.empty(size_p.contents.value, dtype=torch.uint8, device="cpu")
        assert not cudss.cudssConfigGet(
            self.config,
            field,
            cast(out.data_ptr(), c_void_p),
            size_p.contents.value,
            c_void_p(),
        )
        return out.view(dtype)

    def set_data(self, field, data: torch.Tensor):
        assert not cudss.cudssDataSet(
            handle,
            self.data,
            field,
            cast(data.data_ptr(), c_void_p),
            data.nbytes,
        )

    def get_data(self, field, dtype: torch.dtype):
        size_p = pointer(c_int64())
        assert not cudss.cudssDataGet(handle, self.data, field, c_void_p(), 0, size_p)
        out = torch.empty(size_p.contents.value, dtype=torch.uint8, device="cuda")
        assert not cudss.cudssDataGet(
            handle,
            self.data,
            field,
            cast(out.data_ptr(), c_void_p),
            size_p.contents.value,
            size_p,
        )
        return out.view(dtype)

    def inertia(self):
        return self.get_data(CUDSS_DATA_INERTIA, self.indextype)

    def n_pivots(self):
        return self.get_data(CUDSS_DATA_NPIVOTS, self.indextype)

    def solve(self):
        cudss.cudssSetStream(handle, torch.cuda.current_stream()._as_parameter_)
        assert not cudss.cudssExecute(
            handle,
            CUDSS_PHASE_FACTORIZATION,
            self.config,
            self.data,
            *self.mats,
        )
        assert not cudss.cudssExecute(
            handle,
            CUDSS_PHASE_SOLVE,
            self.config,
            self.data,
            *self.mats,
        )
        return self.x

    def __del__(self):
        for m in self.mats:
            cudss.cudssMatrixDestroy(m)
        cudss.cudssConfigDestroy(self.config)
        cudss.cudssDataDestroy(handle, self.data)
