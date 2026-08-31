# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import torch

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

##########################加
# def rot_from_axisangle(vec):
#     """Convert an axisangle rotation into a rotation matrix.
#     """
#     angle = torch.norm(vec, 2, dim=1, keepdim=True)
#     axis = vec / (angle + 1e-7)
#     ca = torch.cos(angle)
#     sa = torch.sin(angle)
#     C = 1 - ca
#
#     x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
#
#     # 罗德里格斯公式 (Rodrigues' rotation formula)
#     R = torch.stack([
#         ca + x*x*C,      x*y*C - z*sa,    x*z*C + y*sa,
#         y*x*C + z*sa,    ca + y*y*C,      y*z*C - x*sa,
#         z*x*C - y*sa,    z*y*C + x*sa,    ca + z*z*C
#     ], dim=1).view(-1, 3, 3)
#
#     return R
#
# def transformation_from_parameters(axisangle, translation, invert=False):
#     """Convert the network's (axisangle, translation) output into a 4x4 matrix
#     """
#     R = rot_from_axisangle(axisangle)
#     t = translation.clone()
#
#     if invert:
#         R = R.transpose(1, 2)
#         t = -torch.matmul(R, t)
#
#     # 【核心修复】这里的 unsqueeze(2) 会把 [B, 3] 变成 [B, 3, 1]，和 R 拼接成 [B, 3, 4]
#     T = torch.cat([R, t.unsqueeze(2)], dim=2)
#
#     return T

