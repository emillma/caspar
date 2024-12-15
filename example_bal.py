# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from dataclasses import dataclass
from pathlib import Path
import time
from urllib import request
import numpy as np
import torch
import symforce

symforce.set_epsilon_to_number(1e-6)

import symforce.symbolic as sf
from caspar import Solver, CuFuncLib, FactorLib, torch_tools as tt

from foxglove_websocket import run_cancellable
from caspar.cuda.kernel import Kernel
from symforce.geo.unsupported.pose3_se3 import Pose3

torch.set_grad_enabled(False)

# fpath = Path(__file__).parent / "data/final/problem-4585-1324582-pre.txt.bz2"
fpath = Path(__file__).parent / "data/final/problem-13682-4456117-pre.txt.bz2"
# fpath = Path(__file__).parent / "data/venice/problem-1778-993923-pre.txt.bz2"
fpath.parent.mkdir(exist_ok=True, parents=True)

if not fpath.exists():
    print("Downloading data... (this may take a while the first time)")
    pre = "https://grail.cs.washington.edu/projects/bal"
    ret = request.urlretrieve(f"{pre}/{'/'.join(fpath.parts[-3:])}", fpath)


if not (pt_path := fpath.with_suffix(".pt")).exists():
    print("Loading data... (this may take a while the first time)")

    def load(file, typ, start, len, cols=None):
        a = np.loadtxt(file, typ, skiprows=start, max_rows=len, usecols=cols)
        tn = torch.from_numpy(a).cuda()
        return tn

    n_cams, n_points, n_facs = load(fpath, np.int32, 0, 1).t().t().contiguous()
    cam_ids, point_ids = load(fpath, np.int32, 1, n_facs, (0, 1)).t().contiguous()
    pixels = load(fpath, np.float32, 1, n_facs, (2, 3)).t().contiguous()
    camdata = load(fpath, np.float32, 1 + n_facs, n_cams * 9)
    camdata = camdata.reshape(n_cams, 9).t().contiguous()
    points = load(fpath, np.float32, 1 + n_facs + n_cams * 9, n_points * 3)
    points = points.reshape(n_points, 3).t().contiguous()

    torch.save((cam_ids, point_ids, camdata, points, pixels), pt_path)
else:
    cam_ids, point_ids, camdata, points, pixels = torch.load(
        pt_path,
        weights_only=False,
        map_location="cuda",
    )
    n_cams, n_points = cam_ids.amax() + 1, point_ids.amax() + 1


@dataclass  # you are absolutely right Aaron, this works :)
class Cam:
    pose: Pose3
    calib: sf.V3


class Point(sf.V3):
    pass


wdir = Path(__file__).parent / "generated" / Path(__file__).stem
klib = CuFuncLib(wdir / "kernels")


@klib.add()
def to_cam(data: sf.V9):
    pose = Pose3(sf.Rot3.from_tangent(data[:3]), sf.V3(data[3:6]))
    return Cam(pose=pose, calib=data[6:9])


klib.load()


def huber_norm(e, k):
    if k == 0:
        return e
    other = sf.sqrt(k * (2 * e * sf.sign(e) - k)) * sf.sign(e)
    return sf.Piecewise((e, e * sf.sign(e) < k), (other, True))


flib = FactorLib(wdir / "factors", [Cam, Point])


@flib.with_consts("pixel")
def snavely_reprojection_residual(
    cam: Cam,
    point: Point,
    pixel: sf.V2,
):
    cam_T_world = cam.pose
    intrinsics = cam.calib
    focal_length, k1, k2 = intrinsics
    point_cam = cam_T_world * point
    d = point_cam[2]
    p = -sf.V2(point_cam[:2]) / (d + sf.epsilon() * sf.sign(d))
    r = 1 + k1 * p.squared_norm() + k2 * p.squared_norm() ** 2
    pixel_projected = focal_length * r * p
    err = pixel_projected - pixel
    return [huber_norm(e, 0) for e in err]


flib.load()

cams = to_cam(camdata, prob_size=n_cams)

cam_median, _ = torch.median(cams[4:7, :], 1)

if False:  # remove outliers
    dist = (cams[4:7, :] - cam_median[:, None]).norm(dim=0)
    valid = dist < torch.quantile(dist, 0.95)
    invalid = torch.nonzero(~valid).ravel()
    valid_fac = ~(cam_ids[:, None] == invalid[None, :]).any(1)
    cam_ids = cam_ids[valid_fac]
    point_ids = point_ids[valid_fac]
    pixels = pixels[:, valid_fac]

a, cam_ids = cam_ids.unique(return_inverse=True)
c, point_ids = point_ids.unique(return_inverse=True)
cams = cams[:, a]
points = points[:, c]

typ2storage_init = {
    Cam: cams.clone(),
    Point: points.clone(),
}
typ2storage = {
    Cam: cams,
    Point: points,
}

factors = [
    snavely_reprojection_residual(
        cam_ids.int(),
        point_ids.int(),
        pixels,
    ),
]

solver = Solver(flib, typ2storage, factors)
final = solver.solve()
solver.print_stats()

# Visualize the results
if True:
    points = final[Point]
    jtr_points = solver.facs.typ2njtr[Point]
    jtr_norm = jtr_points.norm(dim=0)
    valid = jtr_points.norm(dim=0) <= jtr_norm.quantile(0.90)
    points_valid = points[:, valid]
    mean = points_valid.median(1, keepdim=True).values
    points_valid_zeroed = points_valid - mean
    points_cov = points_valid_zeroed @ points_valid_zeroed.T
    eig = torch.linalg.eig(points_valid_zeroed @ points_valid_zeroed.T)
    rot = torch.linalg.inv(eig.eigenvectors.real.float())

    newpoints = rot @ points_valid_zeroed

    orig_points = rot @ (typ2storage_init[Point] - mean)
