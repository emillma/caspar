# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
import torch


def tensor_f32(data):
    return torch.tensor(data, dtype=torch.float32, device="cuda")


def tensor_i32(data):
    return torch.tensor(data, dtype=torch.int32, device="cuda")


def tensor_i64(data):
    return torch.tensor(data, dtype=torch.int64, device="cuda")


def empty_f32(*shape):
    return torch.empty(*shape, dtype=torch.float32, device="cuda")


def zeros_f32(*shape):
    return torch.zeros(*shape, dtype=torch.float32, device="cuda")


def zeros_f64(*shape):
    return torch.zeros(*shape, dtype=torch.float64, device="cuda")


def zeros_i32(*shape):
    return torch.zeros(*shape, dtype=torch.int32, device="cuda")


def empty_f64(*shape):
    return torch.empty(*shape, dtype=torch.float64, device="cuda")


def empty_i32(*shape):
    return torch.empty(*shape, dtype=torch.int32, device="cuda")


def arange_i32(*args):
    return torch.arange(*args, dtype=torch.int32, device="cuda")


def argsort_inv_i32(data: torch.Tensor):
    out = empty_i32(data.numel())
    out[torch.argsort(data)] = arange_i32(data.numel())
    return out


def cumsum_inc_i32(data: torch.Tensor):
    assert data.dim() == 1
    out = empty_i32(data.numel() + 1)
    out[0] = 0
    out[1:] = torch.cumsum(data, 0)
    return out


def compress_index(index, minlength=0):
    counts = torch.bincount(index, minlength=minlength)
    compressed = torch.zeros(counts.numel() + 1, dtype=torch.int32, device="cuda")
    torch.cumsum(counts, 0, out=compressed[1:])
    return compressed


def mgrid(*sizes):
    return torch.meshgrid(*[arange_i32(n) for n in sizes], indexing="ij")


def index_tools1d(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert indices.dtype == torch.int32 and indices.dim() == 1
    unq, inv = torch.unique(indices, return_inverse=True)
    return unq.int(), inv.int()


def index_tools2d(
    indices: torch.Tensor, is_sorted: bool = False, flip: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    # use unique_consecutive
    # assert indices.is_cuda and indices.is_contiguous() and indices.dtype == torch.int32
    # assert indices.dtype == torch.int32 and indices.dim() == 2 and indices.shape[1] == 2
    if flip:
        indices[:] = indices[:, [1, 0]]
    indices_long = indices.view(torch.long).ravel()
    if is_sorted:
        unq, inv = torch.unique_consecutive(indices_long, return_inverse=True)
    else:
        unq, inv = torch.unique(indices_long, return_inverse=True)
    unq_pair = unq.view(torch.int32).reshape(-1, 2)
    if flip:
        unq_pair[:] = unq_pair[:, [1, 0]]
    return unq_pair, inv.int()


def indices2d(shape_a, shape_b, indexing="ij"):
    foo = torch.meshgrid(arange_i32(shape_a), arange_i32(shape_b), indexing=indexing)
    return torch.stack(foo, -1).flatten(0, -2)


def tril_indices(n, m=None, offset=0, transpose=True):
    m = m if m is not None else n
    tril = torch.tril_indices(n, m, offset, dtype=torch.int32, device="cuda")
    if transpose:
        return tril.t().contiguous()
    return tril


def combinations(a, b, indexing="ij"):
    a, b = arange_i32(a), arange_i32(b)
    mesh = torch.meshgrid(a, b, indexing=indexing)
    return torch.stack(mesh, -1).reshape(-1, 2)
