import torch


def normalize_center_torch(pcd: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    valid_sum = mask.sum(dim=0, keepdim=True).clamp(min=eps)
    center = (pcd * mask).sum(dim=0, keepdim=True) / valid_sum
    return (pcd - center) * mask

def estimate_principal_normal_torch(
    pcd: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    deg_thres: float = 0.01
) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)

    valid_pts = pcd[mask.squeeze(-1) > 0]
    if valid_pts.shape[0] < 3:
        return torch.tensor([0., 0., 1.], device=pcd.device, dtype=pcd.dtype)

    cov = (valid_pts.T @ valid_pts) / (valid_pts.shape[0] + eps)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    deg_ratio = (eigvals[1] - eigvals[0]) / (eigvals[1].abs() + eps)
    if deg_ratio < deg_thres:
        return torch.tensor([0., 0., 1.], device=pcd.device, dtype=pcd.dtype)

    normal = eigvecs[:, 0]
    normal = normal / (torch.norm(normal) + eps)
    
    if normal[2] < 0:
        normal = -normal

    return normal

def rotation_from_vector_to_z_torch(n: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    device, dtype = n.device, n.dtype
    z = torch.tensor([0., 0., 1.], device=device, dtype=dtype)

    n = n / (torch.norm(n) + eps)
    v = torch.cross(n, z, dim=0)
    s = torch.norm(v)
    c = torch.clamp(torch.dot(n, z), -1.0, 1.0)

    if s < eps:
        if c > 0:
            return torch.eye(3, device=device, dtype=dtype)
        # n 与 z 反向时绕X轴旋转180°
        return torch.tensor(
            [[1., 0., 0.],
             [0., -1., 0.],
             [0., 0., -1.]],
            device=device,
            dtype=dtype
        )

    vx = torch.stack([
        torch.stack([torch.zeros((), device=device, dtype=dtype), -v[2], v[1]]),
        torch.stack([v[2], torch.zeros((), device=device, dtype=dtype), -v[0]]),
        torch.stack([-v[1], v[0], torch.zeros((), device=device, dtype=dtype)])
    ])

    R = torch.eye(3, device=device, dtype=dtype) + vx + vx @ vx * ((1 - c) / (s ** 2 + eps))
    return R

def estimate_xy_principal_direction_torch(
    pcd: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    deg_thres: float = 0.01
) -> torch.Tensor:
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)

    xy = pcd[mask.squeeze(-1) > 0][:, :2]

    if xy.shape[0] < 3:
        return torch.tensor([1., 0.], device=pcd.device, dtype=pcd.dtype)

    cov = (xy.T @ xy) / (xy.shape[0] + eps)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    deg_ratio = (eigvals[1] - eigvals[0]) / (eigvals[1].abs() + eps)
    if deg_ratio < deg_thres:
        return torch.tensor([1., 0.], device=pcd.device, dtype=pcd.dtype)

    d = eigvecs[:, 1]
    d = d / (torch.norm(d) + eps)

    if d[0] < 0:
        d = -d

    return d

def rotation_xy_to_x_axis_torch(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    device, dtype = d.device, d.dtype
    d = d / (torch.norm(d) + eps)

    angle = torch.atan2(d[1], d[0])
    cos_a = torch.cos(-angle)
    sin_a = torch.sin(-angle)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)

    return torch.stack([
        torch.stack([cos_a, -sin_a, zero]),
        torch.stack([sin_a, cos_a, zero]),
        torch.stack([zero, zero, one])
    ])


def enforce_right_handed_rotation_torch(R: torch.Tensor) -> torch.Tensor:
    if torch.det(R) < 0:
        R = R.clone()
        R[1, :] = -R[1, :]
    return R

def canonicalize_point_cloud(
    pcd: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    deg_thres: float = 0.01
) -> torch.Tensor:
    if mask.dim() == 2 and mask.shape[0] == 1:
        mask = mask.transpose(0, 1)
    elif mask.dim() == 1:
        mask = mask.unsqueeze(-1)

    pcd = normalize_center_torch(pcd, mask)
    n = estimate_principal_normal_torch(pcd, mask, eps, deg_thres)
    R1 = rotation_from_vector_to_z_torch(n, eps)
    pcd = (R1 @ pcd.T).T * mask
    d = estimate_xy_principal_direction_torch(pcd, mask, eps, deg_thres)
    R2 = rotation_xy_to_x_axis_torch(d, eps)
    R = R2 @ R1
    R = enforce_right_handed_rotation_torch(R)
    pcd = (R @ pcd.T).T * mask

    return pcd
