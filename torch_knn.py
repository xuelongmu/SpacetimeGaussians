"""Pure-PyTorch drop-in replacement for mmcv.ops.knn.

The upstream repo pins an mmcv commit that does not build against torch 2.x.
knn() is only used to find each point's nearest neighbour during init-time
point sampling, so a chunked cdist gives identical results without the
mmcv CUDA dependency.
"""

import torch


def knn(k, xyz, center_xyz, transposed=False, chunk=4096):
    """Matches mmcv.ops.knn: returns (B, k, npoint) int64 neighbour indices
    into xyz, sorted by increasing distance."""
    if transposed:
        xyz = xyz.transpose(1, 2).contiguous()
        center_xyz = center_xyz.transpose(1, 2).contiguous()

    idx = []
    for start in range(0, center_xyz.shape[1], chunk):
        block = center_xyz[:, start:start + chunk, :]
        dist = torch.cdist(block, xyz)  # B x chunk x N
        idx.append(dist.topk(k, dim=-1, largest=False).indices)

    return torch.cat(idx, dim=1).transpose(1, 2).contiguous().long()
