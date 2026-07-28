"""Semi-Lagrangian PINN for the 2D-2V Vlasov-Poisson system.

We solve the Vlasov-Poisson equations in four-dimensional phase space
z = (x, y, vx, vy) by learning the inverse characteristic flow map
Psi(t, z) -> z0 and reconstructing the distribution as f = f0(Psi). The
electrostatic field is obtained from a spectral Poisson solve on an Eulerian
grid at each iteration.

Three parameterisations of the map are provided:

    "symplectic"  volume-preserving coupling flow, det(dPsi/dz) = 1 exactly;
                  mass and all Casimir invariants are conserved by construction.
    "normalizing" affine coupling flow with analytic log-determinant; mass is
                  conserved through the change-of-variables reconstruction
                  f = f0(Psi) |det dPsi/dz| (exact when the map preserves the domain).
    "soft"        unconstrained residual network Psi = z + t*NN(t, z) with a
                  soft penalty (det - 1)^2 (the formulation of the original
                  paper), for comparison.

The reference solution is a spectral semi-Lagrangian scheme (Cheng-Knorr,
Strang splitting, exact FFT sub-steps, float64).

Requires only torch, numpy and matplotlib.
"""

from __future__ import annotations

import hashlib
import math
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn


### device

def pick_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    return torch.float32


def solver_dtype(device: torch.device) -> torch.dtype:
    """Reference solver runs in double precision except on MPS."""
    return torch.float32 if device.type == "mps" else torch.float64


### problem definition


_INV_SQRT_2PI = 1.0 / (2.0 * torch.pi) ** 0.5


def _wrap(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Wrap a coordinate into the periodic interval [lo, hi)."""
    return torch.remainder(x - lo, hi - lo) + lo


@dataclass
class Problem:
    name: str
    dim: int
    t_final: float
    x_lo: float
    x_hi: float
    f0: callable
    phase_velocity: callable
    v_lo: float = 0.0
    v_hi: float = 0.0
    space_dim: int = 2
    has_field: bool = False
    external_field: callable | None = None
    f_exact: callable | None = None
    map_exact: callable | None = None
    plot_times: list = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.x_hi - self.x_lo

    @property
    def periodic_half_length(self) -> float:
        return 0.5 * self.length

    def domain_lo(self) -> list:
        return [self.x_lo] * self.space_dim + [self.v_lo] * self.space_dim

    def domain_hi(self) -> list:
        return [self.x_hi] * self.space_dim + [self.v_hi] * self.space_dim


def _blob_2d(u=(0.5, 0.0)):
    def f0(z):
        x = _wrap(z[:, 0], -6.0, 6.0)
        y = _wrap(z[:, 1], -6.0, 6.0)
        return _INV_SQRT_2PI * torch.exp(-0.5 * (x ** 2 + y ** 2
                                                 + (z[:, 2] - u[0]) ** 2
                                                 + (z[:, 3] - u[1]) ** 2))
    return f0


def _vlasov_velocity_2d(t, z, E):
    # z = (x, y, vx, vy); E: [N, 2]
    return torch.cat([z[:, 2:4], E], dim=-1)


def _gaussian_vp_2d() -> Problem:
    return Problem(
        name="gaussian_vp_2d", dim=4, space_dim=2, t_final=2.0,
        x_lo=-6.0, x_hi=6.0, v_lo=-6.0, v_hi=6.0,
        f0=_blob_2d((0.5, 0.0)), phase_velocity=_vlasov_velocity_2d,
        has_field=True, plot_times=[0.0, 1.0, 2.0],
    )


def _free_streaming_2d() -> Problem:
    """E = 0; the exact map is the shear (x - v t, v). Used as a check."""
    u = (0.5, 0.0)

    def f0(z):
        x = _wrap(z[:, 0], -6.0, 6.0)
        y = _wrap(z[:, 1], -6.0, 6.0)
        return torch.exp(-0.5 * (x ** 2 + y ** 2
                                 + (z[:, 2] - u[0]) ** 2 + (z[:, 3] - u[1]) ** 2))

    def velocity(t, z, E=None):
        v = z[:, 2:4]
        return torch.cat([v, torch.zeros_like(v)], dim=-1)

    def map_exact(t, z):
        ts = t.unsqueeze(-1)
        return torch.cat([z[:, 0:2] - z[:, 2:4] * ts, z[:, 2:4]], dim=-1)

    def f_exact(t, z):
        return f0(map_exact(t, z))

    return Problem(
        name="free_streaming_2d", dim=4, space_dim=2, t_final=1.0,
        x_lo=-6.0, x_hi=6.0, v_lo=-6.0, v_hi=6.0,
        f0=f0, phase_velocity=velocity, f_exact=f_exact, map_exact=map_exact,
        plot_times=[0.0, 0.5, 1.0],
    )


_PROBLEMS = {
    "gaussian_vp_2d": _gaussian_vp_2d,
    "free_streaming_2d": _free_streaming_2d,
}


def make_problem(name: str) -> Problem:
    if name not in _PROBLEMS:
        raise ValueError(f"unknown problem {name!r}; choose from "
                         f"{sorted(_PROBLEMS)}")
    return _PROBLEMS[name]()


### spectral Poisson (periodic 2D grid)


def wavenumbers(n: int, length: float, device, dtype) -> torch.Tensor:
    return 2.0 * torch.pi * torch.fft.fftfreq(n, d=length / n,
                                              device=device, dtype=dtype)


def electric_field_2d(rho: torch.Tensor, lengths) -> torch.Tensor:
    """E = -grad phi from -Laplace(phi) = rho - mean(rho).
    rho: [..., Nx, Ny]; returns [..., 2, Nx, Ny].
    """
    nx, ny = rho.shape[-2], rho.shape[-1]
    kx = wavenumbers(nx, lengths[0], rho.device, rho.dtype).view(nx, 1)
    ky = wavenumbers(ny, lengths[1], rho.device, rho.dtype).view(1, ny)
    rho_hat = torch.fft.fft2(rho, dim=(-2, -1))
    k2 = kx ** 2 + ky ** 2
    k2_safe = torch.where(k2 == 0, torch.ones_like(k2), k2)
    phi_hat = torch.where(k2 == 0, torch.zeros_like(rho_hat), rho_hat / k2_safe)
    ex = torch.fft.ifft2(-1j * kx * phi_hat, dim=(-2, -1)).real
    ey = torch.fft.ifft2(-1j * ky * phi_hat, dim=(-2, -1)).real
    return torch.stack([ex, ey], dim=-3)


def interp_periodic_2d(values, x0s, lengths, xq, row_index):
    """Bilinear interpolation on a uniform periodic 2D grid.
    values: [Nt, C, Nx, Ny]; xq: [N, 2]; row_index: [N]; returns [N, C].
    """
    nx, ny = values.shape[-2], values.shape[-1]
    ux = (xq[:, 0] - x0s[0]) / (lengths[0] / nx)
    uy = (xq[:, 1] - x0s[1]) / (lengths[1] / ny)
    ix0 = torch.floor(ux).long()
    iy0 = torch.floor(uy).long()
    wx = (ux - ix0.to(ux.dtype)).unsqueeze(-1)
    wy = (uy - iy0.to(uy.dtype)).unsqueeze(-1)
    ix0, iy0 = ix0 % nx, iy0 % ny
    ix1, iy1 = (ix0 + 1) % nx, (iy0 + 1) % ny
    r = row_index
    v00 = values[r, :, ix0, iy0]
    v10 = values[r, :, ix1, iy0]
    v01 = values[r, :, ix0, iy1]
    v11 = values[r, :, ix1, iy1]
    return (v00 * (1 - wx) * (1 - wy) + v10 * wx * (1 - wy)
            + v01 * (1 - wx) * wy + v11 * wx * wy)



### velocity-space quadrature and charge density

def uniform_grid(lo, hi, n, device, dtype) -> torch.Tensor:
    """Uniform periodic grid on [lo, hi) (no right endpoint)."""
    step = (hi - lo) / n
    return lo + step * torch.arange(n, device=device, dtype=dtype)


def trapezoid_weights(v: torch.Tensor) -> torch.Tensor:
    dv = v[1] - v[0]
    w = torch.full_like(v, dv)
    w[0] = 0.5 * dv
    w[-1] = 0.5 * dv
    return w


def eval_on_mesh(f_fn, t_slices, grids, chunk_size=1 << 18,
                 checkpoint=False) -> torch.Tensor:
    """Evaluate f_fn on (t_slices x grids[0] x ... ) -> [Nt, *grid_shapes]."""
    shape = [len(t_slices)] + [len(g) for g in grids]
    mesh = torch.meshgrid(t_slices, *grids, indexing="ij")
    tt = mesh[0].reshape(-1)
    z = torch.stack([m.reshape(-1) for m in mesh[1:]], dim=-1)
    del mesh
    use_ckpt = checkpoint and torch.is_grad_enabled()
    outs = []
    for i in range(0, len(tt), chunk_size):
        t_c, z_c = tt[i:i + chunk_size], z[i:i + chunk_size]
        if use_ckpt:
            outs.append(torch.utils.checkpoint.checkpoint(
                f_fn, t_c, z_c, use_reentrant=False))
        else:
            outs.append(f_fn(t_c, z_c))
    return torch.cat(outs).view(*shape)


def rho_on_grid_2d(f_fn, t_slices, x_grids, v_grids, v_weights,
                   chunk_size=1 << 18) -> torch.Tensor:
    f = eval_on_mesh(f_fn, t_slices, x_grids + v_grids, chunk_size)
    return torch.einsum("txyuw,u,w->txy", f, v_weights[0], v_weights[1])


def velocity_nodes(kind, n_nodes, space_dim, mean, sigma, device, dtype,
                   generator=None):
    """Nodes and weights for the velocity integral int f dv over R^s.

    Returns v [M, s] and w [M] with  int f dv  ~=  sum_m f(v_m) w_m.

    Accuracy measured against a Nv=96 tensor grid on gaussian_vp_2d at t=1,
    relative L2 of rho (nodes per spatial point in brackets):

        grid  Nv=24  [576]   2.9e-08     <- spectrally accurate, the baseline
        qmc   M=64   [ 64]   1.8e-03
        qmc   M=256  [256]   8.1e-04
        qmc   M=1024 [1024]  1.7e-04
        mc    M=1024 [1024]  2.7e-03
        uniform-box M=1024   6.3e-02     <- why the proposal matters
    """
    if kind == "qmc":
        # SobolEngine is CPU-only and its seed must come from a CPU generator;
        # the nodes are moved to `device` after they are drawn.
        seed = None if generator is None else int(
            torch.randint(0, 2 ** 31 - 1, (1,),
                          generator=generator,
                          device=generator.device).item())
        eng = torch.quasirandom.SobolEngine(space_dim, scramble=True, seed=seed)
        u01 = eng.draw(n_nodes).to(device=device, dtype=dtype)
        u01 = u01.clamp(1e-7, 1.0 - 1e-7)
        eps = torch.erfinv(2.0 * u01 - 1.0) * math.sqrt(2.0)
    elif kind == "mc":
        eps = torch.randn(n_nodes, space_dim, device=device, dtype=dtype,
                          generator=generator)
    else:
        raise ValueError(f"unknown velocity quadrature {kind!r}")
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype).expand(space_dim)
    v = mean + sigma * eps
    # q(v) = prod_j N(v_j; mean_j, sigma_j^2);  w = 1 / (M q(v))
    log_q = (-0.5 * (eps ** 2).sum(-1) - torch.log(sigma).sum()
             - 0.5 * space_dim * math.log(2.0 * math.pi))
    return v, torch.exp(-log_q) / n_nodes


def eval_on_nodes_2d(f_fn, t_slices, x_grids, v_nodes, chunk_size=1 << 18,
                     checkpoint=False) -> torch.Tensor:
    """f on (t_slices x x_grid x y_grid x v_nodes) -> [Nt, Nx, Ny, M]."""
    nt, nx, ny, m = len(t_slices), len(x_grids[0]), len(x_grids[1]), len(v_nodes)
    tt = t_slices.view(nt, 1, 1, 1).expand(nt, nx, ny, m).reshape(-1)
    xx = x_grids[0].view(1, nx, 1, 1).expand(nt, nx, ny, m).reshape(-1)
    yy = x_grids[1].view(1, 1, ny, 1).expand(nt, nx, ny, m).reshape(-1)
    s = v_nodes.shape[-1]
    vv = v_nodes.view(1, 1, 1, m, s).expand(nt, nx, ny, m, s).reshape(-1, s)
    z = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), vv], dim=-1)
    use_ckpt = checkpoint and torch.is_grad_enabled()
    outs = []
    for i in range(0, len(tt), chunk_size):
        t_c, z_c = tt[i:i + chunk_size], z[i:i + chunk_size]
        if use_ckpt:
            outs.append(torch.utils.checkpoint.checkpoint(
                f_fn, t_c, z_c, use_reentrant=False))
        else:
            outs.append(f_fn(t_c, z_c))
    return torch.cat(outs).view(nt, nx, ny, m)


def moments_on_nodes_2d(f_fn, t_slices, x_grids, v_nodes, v_w,
                        chunk_size=1 << 18, checkpoint=False):
    """(rho, J, K) from a velocity point cloud instead of a tensor grid."""
    f = eval_on_nodes_2d(f_fn, t_slices, x_grids, v_nodes, chunk_size,
                         checkpoint)
    rho = torch.einsum("txym,m->txy", f, v_w)
    J = torch.einsum("txym,m,mj->tjxy", f, v_w, v_nodes)
    vsq = (v_nodes ** 2).sum(-1)
    kin = 0.5 * torch.einsum("txym,m,m->txy", f, v_w, vsq)
    return rho, J, kin


def moments_on_grid_2d(f_fn, t_slices, x_grids, v_grids, v_weights,
                       chunk_size=1 << 18, checkpoint=False):
    """First velocity moments on the (x, y) grid.
    Returns rho [Nt, Nx, Ny], J [Nt, 2, Nx, Ny] and K [Nt, Nx, Ny], i.e.
    int f dv, int v f dv and int |v|^2/2 f dv.
    """
    f = eval_on_mesh(f_fn, t_slices, x_grids + v_grids, chunk_size, checkpoint)
    wu, ww = v_weights
    rho = torch.einsum("txyuw,u,w->txy", f, wu, ww)
    jx = torch.einsum("txyuw,u,u,w->txy", f, v_grids[0], wu, ww)
    jy = torch.einsum("txyuw,u,w,w->txy", f, wu, v_grids[1], ww)
    vsq = v_grids[0].view(-1, 1) ** 2 + v_grids[1].view(1, -1) ** 2
    kin = 0.5 * torch.einsum("txyuw,uw,u,w->txy", f, vsq, wu, ww)
    return rho, torch.stack([jx, jy], dim=1), kin


def longitudinal_part_2d(J: torch.Tensor, lengths) -> torch.Tensor:
    """Irrotational (Helmholtz) part of a periodic vector field."""
    nx, ny = J.shape[-2], J.shape[-1]
    kx = wavenumbers(nx, lengths[0], J.device, J.dtype).view(nx, 1)
    ky = wavenumbers(ny, lengths[1], J.device, J.dtype).view(1, ny)
    k2 = kx ** 2 + ky ** 2
    k2_safe = torch.where(k2 == 0, torch.ones_like(k2), k2)
    j_hat = torch.fft.fft2(J, dim=(-2, -1))
    k_dot_j = kx * j_hat[..., 0, :, :] + ky * j_hat[..., 1, :, :]
    coef = k_dot_j / k2_safe
    out = torch.stack([kx * coef, ky * coef], dim=-3)
    out = torch.where((k2 == 0).unsqueeze(-3), j_hat, out)   # keep the mean
    return torch.fft.ifft2(out, dim=(-2, -1)).real


### flow-map models

def _periodic_features(x: torch.Tensor, half_length: float) -> torch.Tensor:
    w = torch.pi / half_length
    return torch.stack([torch.sin(w * x), torch.cos(w * x)], dim=-1)


class CoordEmbed(nn.Module):
    """Embeds a coordinate as its scaled raw value, or (sin, cos) if periodic.
    `scale` normalises the non-periodic coordinate to O(1); the periodic
    features are already bounded.
    """

    def __init__(self, periodic_half_length, scale=1.0):
        super().__init__()
        self.half_length = periodic_half_length
        self.scale = scale

    @property
    def out_dim(self) -> int:
        return 2 if self.half_length is not None else 1

    def forward(self, x):
        if self.half_length is not None:
            return _periodic_features(x, self.half_length)
        return (x / self.scale).unsqueeze(-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width, depth, zero_init_last=False):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        last = nn.Linear(d, out_dim)
        if zero_init_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class FlowMap(nn.Module):
    """Base class: batched inverse map Psi(t [N], z [N, d]) -> z0 [N, d]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, z):
        raise NotImplementedError

    def spatial_jacobian(self, t, z):
        """d(Psi)/dz, shape [N, d, d], via vmapped reverse-mode autodiff."""
        def single(t_i, z_i):
            return self.forward(t_i.unsqueeze(0), z_i.unsqueeze(0)).squeeze(0)
        return torch.func.vmap(torch.func.jacrev(single, argnums=1))(t, z)

    def logdet(self, t, z):
        jac = self.spatial_jacobian(t, z)
        return torch.linalg.slogdet(jac).logabsdet

    def inverse(self, t, z0, newton_iters=30, tol=1e-10):
        """Solve Psi(t, z) = z0 for z (forward-in-time characteristics)."""
        z = z0.detach().clone()
        with torch.enable_grad():
            for _ in range(newton_iters):
                r = self.forward(t, z) - z0
                if r.abs().max() < tol:
                    break
                jac = self.spatial_jacobian(t, z)
                step = torch.linalg.solve(jac, r.unsqueeze(-1)).squeeze(-1)
                z = z - step
        return z.detach()


class _CouplingLayer(nn.Module):
    """Additive or affine coupling on the (x-block, v-block) split."""

    def __init__(self, target, space_dim, cond_periodic_half_length,
                 width, depth, affine, max_log_scale=1.0, t_scale=1.0,
                 cond_scale=1.0, gate_tau=1.0, gating="split"):
        super().__init__()
        self.target = target
        self.s = space_dim
        self.affine = affine
        self.max_log_scale = max_log_scale
        self.t_scale = t_scale
        self.gate_tau = gate_tau
        self.gating = gating
        self.embed = CoordEmbed(cond_periodic_half_length, cond_scale)
        out = 2 * self.s if affine else self.s
        self.net = MLP(self.s * self.embed.out_dim + 1, out, width, depth,
                       zero_init_last=True)
        self.shear = (MLP(self.s * self.embed.out_dim, self.s, width,
                          max(1, depth - 1), zero_init_last=True)
                      if gating == "split" else None)

    def _gate(self, t):
        if self.gating == "split":
            return -torch.expm1(-t / self.gate_tau)
        if self.gating == "saturating":
            return -torch.expm1(-t)
        return t                                    # "linear"

    def _params(self, t, other):
        """Total displacement (already including its t-gate) and log-scale."""
        feats = torch.cat([self.embed(other[:, j]) for j in range(self.s)],
                          dim=-1)
        h = torch.cat([feats, (t / self.t_scale).unsqueeze(-1)], dim=-1)
        out = self.net(h)
        disp = self._gate(t).unsqueeze(-1) * out[:, :self.s]
        if self.shear is not None:
            disp = disp + t.unsqueeze(-1) * self.shear(feats)
        if not self.affine:
            return disp, None
        log_scale = (self.max_log_scale * torch.tanh(out[:, self.s:])
                     * torch.tanh(t / self.t_scale).unsqueeze(-1))
        return disp, log_scale

    def forward_with_logdet(self, t, z):
        s = self.s
        x_block, v_block = z[:, :s], z[:, s:]
        other = v_block if self.target == 0 else x_block
        disp, log_scale = self._params(t, other)
        upd = x_block if self.target == 0 else v_block
        if self.affine:
            upd = upd * torch.exp(log_scale) + disp
            ld = log_scale.sum(dim=-1)
        else:
            upd = upd + disp
            ld = torch.zeros(len(z), device=z.device, dtype=z.dtype)
        if self.target == 0:
            return torch.cat([upd, v_block], dim=-1), ld
        return torch.cat([x_block, upd], dim=-1), ld

    def forward(self, t, z):
        return self.forward_with_logdet(t, z)[0]

    def inverse(self, t, z):
        s = self.s
        x_block, v_block = z[:, :s], z[:, s:]
        other = v_block if self.target == 0 else x_block
        disp, log_scale = self._params(t, other)
        upd = x_block if self.target == 0 else v_block
        if self.affine:
            upd = (upd - disp) * torch.exp(-log_scale)
        else:
            upd = upd - disp
        if self.target == 0:
            return torch.cat([upd, v_block], dim=-1)
        return torch.cat([x_block, upd], dim=-1)


class CouplingFlowMap(FlowMap):
    """Composition of alternating coupling layers.

    affine=False -> volume-preserving (det = 1 exactly).
    affine=True  -> affine, with an exact log-determinant.
    """

    def __init__(self, dim, space_dim, n_layers=8, width=64, depth=2,
                 periodic_half_length=None, affine=False, t_scale=1.0,
                 v_scale=1.0, gate_tau=1.0, gating="split"):
        super().__init__(dim)
        layers = []
        for i in range(n_layers):
            target = i % 2
            cond_periodic = periodic_half_length if target == 1 else None
            layers.append(_CouplingLayer(target, space_dim, cond_periodic,
                                         width, depth, affine,
                                         t_scale=t_scale, cond_scale=v_scale,
                                         gate_tau=gate_tau, gating=gating))
        self.layers = nn.ModuleList(layers)

    def forward_with_logdet(self, t, z):
        logdet = torch.zeros(len(z), device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, ld = layer.forward_with_logdet(t, z)
            logdet = logdet + ld
        return z, logdet

    def forward(self, t, z):
        return self.forward_with_logdet(t, z)[0]

    def logdet(self, t, z):
        return self.forward_with_logdet(t, z)[1]

    def inverse(self, t, z0, **_):
        z = z0
        for layer in reversed(self.layers):
            z = layer.inverse(t, z)
        return z


class MLPFlowMap(FlowMap):
    """Residual map Psi = z + t*NN(t, z); Jacobian by autodiff.
    This is the paper's unconstrained ansatz and is kept as-is for the "soft"
    ablation; only the network *inputs* are normalised (t/t_final, v/v_scale),
    which leaves the function class unchanged.
    """

    def __init__(self, dim, periodic_half_length=None, width=128, depth=5,
                 t_scale=1.0, v_scale=1.0):
        super().__init__(dim)
        self.embed = CoordEmbed(periodic_half_length)
        # periodic embedding on the spatial coordinates, raw on velocities
        self.space_dim = max(1, dim // 2)
        self.t_scale = t_scale
        self.v_scale = v_scale
        in_dim = self.space_dim * self.embed.out_dim + (dim - self.space_dim) + 1
        self.net = MLP(in_dim, dim, width, depth, zero_init_last=True)

    def _features(self, t, z):
        s = self.space_dim
        feats = [self.embed(z[:, j]) for j in range(s)]
        feats.append(z[:, s:] / self.v_scale)
        feats.append((t / self.t_scale).unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def forward(self, t, z):
        return z + t.unsqueeze(-1) * self.net(self._features(t, z))


def build_model(cfg, problem, device, dtype):
    d, s = problem.dim, problem.space_dim
    hl = problem.periodic_half_length
    if cfg.normalize_inputs:
        t_scale = max(problem.t_final, 1e-12)
        v_scale = max(abs(problem.v_lo), abs(problem.v_hi), 1e-12)
    else:
        t_scale = v_scale = 1.0
    tau = cfg.gate_tau if cfg.gate_tau is not None else problem.t_final / 5.0
    kw = dict(t_scale=t_scale, v_scale=v_scale, gate_tau=max(tau, 1e-12),
              gating=cfg.gating)
    if cfg.mode == "symplectic":
        model = CouplingFlowMap(d, s, cfg.n_layers, cfg.width, cfg.depth, hl,
                                affine=False, **kw)
    elif cfg.mode == "normalizing":
        model = CouplingFlowMap(d, s, cfg.n_layers, cfg.width, cfg.depth, hl,
                                affine=True, **kw)
    elif cfg.mode == "soft":
        model = MLPFlowMap(d, hl, cfg.width, cfg.depth,
                           t_scale=t_scale, v_scale=v_scale)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    return model.to(device=device, dtype=dtype)


### reconstruction and residuals

def reconstruct_f(model, f0, change_of_variables=False):
    """Return f(t, z) = f0(Psi(t, z)) [* |det dPsi/dz| if requested]."""
    if change_of_variables:
        def f_fn(t, z):
            z0, logdet = model.forward_with_logdet(t, z)
            return f0(z0) * torch.exp(logdet)
    else:
        def f_fn(t, z):
            return f0(model(t, z))
    return f_fn


def _jvp_along_flow(fn, t, z, velocity):
    """Directional derivative of fn(t, z) along (dt, dz) = (1, velocity)."""
    _, tangent = torch.func.jvp(fn, (t, z), (torch.ones_like(t), velocity))
    return tangent


def vector_residual(model, t, z, velocity):
    """Characteristic residual d_t Psi + u . grad_z Psi, shape [N, d]."""
    return _jvp_along_flow(lambda t_, z_: model(t_, z_), t, z, velocity)


def jacobian_penalty(model, t, z):
    """(det grad_z Psi - 1)^2 for the soft-penalty formulation."""
    jac = model.spatial_jacobian(t, z)
    return (torch.linalg.det(jac) - 1.0) ** 2


### collocation sampling

@dataclass
class Batch:
    t: torch.Tensor
    z: torch.Tensor
    t_slices: torch.Tensor
    slice_index: torch.Tensor


def sample_collocation(problem, n_slices, n_per_slice, device, dtype,
                       generator=None, weight_fn=None, uniform_frac=0.25,
                       oversample=4, weight_alpha=1.0, weight_floor=1e-3):
    """Stratified time slices with phase-space collocation points."""
    u = torch.rand(n_slices, device=device, dtype=dtype, generator=generator)
    t_slices = (torch.arange(n_slices, device=device, dtype=dtype) + u) \
        * (problem.t_final / n_slices)

    lo = torch.tensor(problem.domain_lo(), device=device, dtype=dtype)
    hi = torch.tensor(problem.domain_hi(), device=device, dtype=dtype)
    d = problem.dim
    slice_index = torch.arange(n_slices, device=device) \
        .repeat_interleave(n_per_slice)

    n_w = n_per_slice - int(round(uniform_frac * n_per_slice)) \
        if weight_fn is not None else 0
    if n_w <= 0:
        n = n_slices * n_per_slice
        z = lo + (hi - lo) * torch.rand(n, d, device=device, dtype=dtype,
                                        generator=generator)
        return Batch(t_slices[slice_index], z, t_slices, slice_index)

    n_u = n_per_slice - n_w
    z_u = lo + (hi - lo) * torch.rand(n_slices, n_u, d, device=device,
                                      dtype=dtype, generator=generator)
    m = oversample * n_w
    cand = lo + (hi - lo) * torch.rand(n_slices, m, d, device=device,
                                       dtype=dtype, generator=generator)
    t_cand = t_slices.repeat_interleave(m)
    with torch.no_grad():
        wf = weight_fn(t_cand, cand.reshape(-1, d))
    w = torch.nan_to_num(wf.detach()).clamp_min(0).pow(weight_alpha).view(n_slices, m)
    w = w + weight_floor * w.mean(dim=1, keepdim=True)
    ok = w.sum(dim=1, keepdim=True) > 0
    w = torch.where(ok, w, torch.ones_like(w))
    g = torch.rand(n_slices, m, device=device, dtype=dtype, generator=generator)
    keys = torch.log(w) - torch.log(-torch.log(g.clamp(1e-12, 1 - 1e-6)))
    idx = keys.topk(n_w, dim=1).indices
    z_w = cand.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
    z = torch.cat([z_u, z_w], dim=1).reshape(n_slices * n_per_slice, d)
    return Batch(t_slices[slice_index], z, t_slices, slice_index)


### training


@dataclass
class TrainConfig:
    problem: str = "gaussian_vp_2d"
    mode: str = "symplectic"          # symplectic | normalizing | soft
    iters: int = 4000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_slices: int = 8
    n_per_slice: int = 512
    field_nx: int = 16                # per-axis Eulerian grid for rho/E
    field_nv: int = 16
    lambda_jac: float = 1e-3          # soft mode only
    # --- field self-consistency (see Trainer.step) -------------------------
    detach_field: bool = True         # Picard: no "shrink my own E" gradient
    lambda_energy: float = 1e-3       # penalty on total-energy drift
    lambda_ampere: float = 1.0        # penalty on d_t E + P_L[J] - P_0/A = 0
    # velocity quadrature: "grid" = tensor-product trapezoid (Nv^s nodes per
    #   spatial point, spectrally accurate); "qmc"/"mc" = importance-sampled
    #   point cloud of `n_velocity_nodes` nodes (see velocity_nodes for the
    #   measured accuracy trade).  The node count is what decouples cost from
    #   the velocity dimension, so the saving grows as s grows.
    velocity_quadrature: str = "grid"   # grid | qmc | mc
    n_velocity_nodes: int = 256
    resample_velocity: bool = True      # redraw nodes each step (unbiased)
    proposal_widen: float = 1.25        # widen the fitted proposal std
    field_checkpoint: bool = True     # recompute the field solve in backward
    ampere_slices: int | None = 4     # time slices used by the Ampere jvp;
    #   None = all.  This is the knob that bounds the one path checkpointing
    #   cannot cover (see Trainer._fields).
    field_chunk: int = 1 << 16        # points per chunk in the field solve.
    #   Measured on the 1D case: this does NOT lower peak training memory,
    #   because every chunk's activations stay in the autograd graph until
    #   backward.  Lowering field_nx/field_nv or n_slices is what helps -- and
    #   in 4D the field grid is field_nx^2 * field_nv^2 points per time slice,
    #   so those knobs bite hard.
    # --- flow-map parameterisation (see _CouplingLayer) --------------------
    gating: str = "split"             # split | linear | saturating
    gate_tau: float | None = None     # transient gate scale; None -> t_final/5
    normalize_inputs: bool = True     # feed t/t_final and v/v_max to the MLPs
    sampling: str = "uniform"         # uniform | support
    uniform_frac: float = 0.25
    oversample: int = 8
    weight_alpha: float = 1.0
    n_layers: int = 8
    width: int = 64
    depth: int = 2
    seed: int = 0
    device: str | None = None
    log_every: int = 200

    @property
    def change_of_variables(self) -> bool:
        return self.mode == "normalizing"


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.dtype = default_dtype(self.device)
        torch.manual_seed(cfg.seed)

        self.problem = make_problem(cfg.problem)
        self.model = build_model(cfg, self.problem, self.device, self.dtype)
        self.f_fn = reconstruct_f(self.model, self.problem.f0,
                                  cfg.change_of_variables)
        self._weight_fn = self.f_fn if cfg.sampling == "support" else None

        if self.problem.has_field:
            p = self.problem
            xg = uniform_grid(p.x_lo, p.x_hi, cfg.field_nx, self.device, self.dtype)
            vg = torch.linspace(p.v_lo, p.v_hi, cfg.field_nv,
                                device=self.device, dtype=self.dtype)
            self.x_grids = [xg, xg.clone()]
            self.v_grids = [vg, vg.clone()]
            self.v_weights = [trapezoid_weights(g) for g in self.v_grids]
            self.dx = (p.length / cfg.field_nx) ** 2      # area element
            # Importance proposal fitted to the velocity marginal of f0 by a
            # one-off grid quadrature.  Getting this wrong is expensive: a
            # proposal centred at 0 against a blob centred at v=(0.5, 0) costs
            # an order of magnitude on rho and more on J, because a mismatched
            # mean biases the first moment hardest.  Widened by
            # `proposal_widen` because the blob spreads under the flow and
            # heavier proposal tails are the cheap insurance against a few
            # samples landing where f/q is large.
            self._v_mean, self._v_sigma = self._fit_velocity_proposal()
            self._v_gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
            self._v_fixed = (None if cfg.velocity_quadrature == "grid"
                             else self._draw_velocity_nodes())
            # invariants at t = 0, on the same quadrature the penalties use
            with torch.no_grad():
                t0 = torch.zeros(1, device=self.device, dtype=self.dtype)
                # Same estimator as the penalties use: with a stochastic
                # quadrature, exact t=0 constants would leave a constant bias
                # in eps_w and in the Ampere residual.
                if cfg.velocity_quadrature == "grid":
                    rho0, j0, k0 = moments_on_grid_2d(
                        lambda t, z: p.f0(z), t0, self.x_grids, self.v_grids,
                        self.v_weights, cfg.field_chunk)
                else:
                    v0, w0 = (self._v_fixed if self._v_fixed is not None
                              else self._draw_velocity_nodes())
                    rho0, j0, k0 = moments_on_nodes_2d(
                        lambda t, z: p.f0(z), t0, self.x_grids, v0, w0,
                        cfg.field_chunk)
                e0 = electric_field_2d(rho0, (p.length, p.length))
                self.w_field0 = (0.5 * (e0 ** 2).sum() * self.dx).clamp_min(1e-12)
                self.w_total0 = k0.sum() * self.dx + self.w_field0
                # <J> = P_0 / A, the constant of integration in Ampere's law.
                # Momentum is conserved exactly under a self-consistent field
                # (dP/dt = int rho E = 0 by the Maxwell stress on a torus), so
                # this is fixed once from f0 rather than recomputed from the
                # model -- a running mean would make the zero mode vacuous.
                self.j_bar0 = j0.mean(dim=(-2, -1)).reshape(2, 1, 1)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.history = []

    def _fit_velocity_proposal(self):
        """Mean and per-axis std of the velocity marginal of f0."""
        p, cfg = self.problem, self.cfg
        t0 = torch.zeros(1, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            f = eval_on_mesh(lambda t, z: p.f0(z), t0,
                             self.x_grids + self.v_grids, cfg.field_chunk)
        wu, ww = self.v_weights
        tot = torch.einsum("txyuw,u,w->", f, wu, ww).clamp_min(1e-30)
        gu, gw = self.v_grids
        mx = torch.einsum("txyuw,u,u,w->", f, gu, wu, ww) / tot
        my = torch.einsum("txyuw,u,w,w->", f, wu, gw, ww) / tot
        sx = (torch.einsum("txyuw,u,u,w->", f, (gu - mx) ** 2, wu, ww) / tot).sqrt()
        sy = (torch.einsum("txyuw,u,w,w->", f, wu, (gw - my) ** 2, ww) / tot).sqrt()
        mean = torch.stack([mx, my])
        sigma = torch.stack([sx, sy]).clamp_min(1e-3) * cfg.proposal_widen
        return mean, sigma

    def _draw_velocity_nodes(self):
        cfg, p = self.cfg, self.problem
        return velocity_nodes(cfg.velocity_quadrature, cfg.n_velocity_nodes,
                              p.space_dim, self._v_mean, self._v_sigma,
                              self.device, self.dtype, self._v_gen)

    def _moments(self, t_slices, checkpoint, nodes=None):
        """Velocity moments, by tensor grid or by importance-sampled nodes."""
        cfg = self.cfg
        if cfg.velocity_quadrature == "grid":
            return moments_on_grid_2d(self.f_fn, t_slices, self.x_grids,
                                      self.v_grids, self.v_weights,
                                      cfg.field_chunk, checkpoint)
        v, w = nodes
        return moments_on_nodes_2d(self.f_fn, t_slices, self.x_grids, v, w,
                                   cfg.field_chunk, checkpoint)

    def _fields(self, t_slices):
        """E and the moments per time slice, plus (dE/dt, J) on the subset of
        slices the Ampere penalty uses."""
        cfg, p = self.cfg, self.problem
        lengths = (p.length, p.length)

        nodes = None
        if cfg.velocity_quadrature != "grid":
            nodes = (self._draw_velocity_nodes() if cfg.resample_velocity
                     else self._v_fixed)

        def moments(ts, checkpoint):
            return self._moments(ts, checkpoint, nodes)

        rho, J, K = moments(t_slices, cfg.field_checkpoint)
        e_grid = electric_field_2d(rho, lengths)
        if cfg.lambda_ampere <= 0:
            return e_grid, J, K, None, None

        n = len(t_slices)
        k = n if cfg.ampere_slices is None else min(cfg.ampere_slices, n)
        idx = (torch.randperm(n, device=t_slices.device)[:k] if k < n
               else torch.arange(n, device=t_slices.device))
        ts_a = t_slices[idx]
        # checkpoint=False: forward-mode AD cannot go through checkpoint()
        (rho_a, J_a, _), (drho_a, _, _) = torch.func.jvp(
            lambda s: moments(s, False), (ts_a,), (torch.ones_like(ts_a),))
        # The Poisson solve is linear in rho, so d_t E = E[d_t rho]; taking
        # d/dt before the solve rather than through it keeps the FFT out of the
        # forward-mode trace (backward through a dual-number FFT is not
        # supported by the CPU/MKL backend).
        return e_grid, J, K, electric_field_2d(drho_a, lengths), J_a

    def step(self) -> dict:
        cfg, p = self.cfg, self.problem
        lengths = (p.length, p.length)
        batch = sample_collocation(self.problem, cfg.n_slices, cfg.n_per_slice,
                                   self.device, self.dtype,
                                   weight_fn=self._weight_fn,
                                   uniform_frac=cfg.uniform_frac,
                                   oversample=cfg.oversample,
                                   weight_alpha=cfg.weight_alpha)
        rec = {}
        e = None
        if p.has_field:
            e_grid, J, K, de_dt, J_a = self._fields(batch.t_slices)
            # The characteristic residual sees the field as a *fixed* target
            # (Picard).  Left attached, the cheapest way to lower that residual
            # is for the map to flatten its own rho and drive E -> 0: E enters
            # the residual additively as E * d_v Psi, so for any map that is
            # not already the exact response, |E| > 0 only costs loss.
            # Detaching removes that degenerate descent direction; the
            # conservation terms below then supply the *correct* gradient path
            # to the field, and they are deliberately left attached.
            e_used = e_grid.detach() if cfg.detach_field else e_grid
            e = interp_periodic_2d(e_used, (p.x_lo, p.x_lo), lengths,
                                   batch.z[:, :2], batch.slice_index)
        velocity = self.problem.phase_velocity(batch.t, batch.z, e)
        res = vector_residual(self.model, batch.t, batch.z, velocity)
        loss_char = res.pow(2).sum(dim=-1).mean()

        loss = loss_char
        rec["loss_char"] = loss_char.item()
        if cfg.mode == "soft" and cfg.lambda_jac > 0:
            n_pen = min(2048, len(batch.t))
            loss_jac = jacobian_penalty(self.model, batch.t[:n_pen],
                                        batch.z[:n_pen]).mean()
            loss = loss + cfg.lambda_jac * loss_jac
            rec["loss_jac"] = loss_jac.item()

        if p.has_field and cfg.lambda_energy > 0:
            # Total energy is a global invariant no local residual implies, and
            # det = 1 does not protect it.  Measured relative to the initial
            # field energy, so eps_W = -1 is the scale of losing all of it.
            w = (K.sum(dim=(1, 2)) + 0.5 * (e_grid ** 2).sum(dim=(1, 2, 3))) \
                * self.dx
            eps_w = (w - self.w_total0) / self.w_field0
            loss_energy = torch.nn.functional.smooth_l1_loss(
                eps_w, torch.zeros_like(eps_w), beta=1.0)
            loss = loss + cfg.lambda_energy * loss_energy
            rec["loss_energy"] = loss_energy.item()
            rec["eps_w"] = eps_w.abs().max().item()

        if p.has_field and cfg.lambda_ampere > 0:
            # Ampere in 2D: d_t E + P_L[J] - P_0/A = 0, where P_L is the
            # longitudinal projector (see longitudinal_part_2d -- the 1D
            # identity holds only for the irrotational part of J once the
            # domain has more than one spatial dimension).
            #
            # The constant of integration is fixed at t = 0, not recomputed
            # from the model: subtracting a running mean would force the x-mean
            # of the residual to vanish algebraically for any map, projecting
            # the zero mode -- the momentum -- out of the loss.  Against the
            # fixed j_bar0 the mean of res_a is (P(t) - P_0)/A, i.e. the
            # momentum drift itself.
            res_a = de_dt + longitudinal_part_2d(J_a, lengths) - self.j_bar0
            loss_ampere = res_a.pow(2).mean()
            loss = loss + cfg.lambda_ampere * loss_ampere
            rec["loss_ampere"] = loss_ampere.item()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        rec["loss"] = loss.item()
        return rec

    def train(self) -> None:
        t0 = time.time()
        for it in range(1, self.cfg.iters + 1):
            rec = self.step()
            rec["iter"] = it
            self.history.append(rec)
            if it % self.cfg.log_every == 0 or it == 1:
                rate = it / (time.time() - t0)
                extra = "".join(f"  {k[5:]}={rec[k]:.2e}" for k in
                                ("loss_char", "loss_energy", "loss_ampere")
                                if k in rec)
                print(f"[{self.cfg.mode}] iter {it:6d}  loss={rec['loss']:.3e}"
                      f"{extra}  ({rate:.1f} it/s)", flush=True)
        self.train_time = time.time() - t0

    def save(self, path) -> None:
        """Checkpoint the map plus enough provenance to skip retraining."""
        torch.save({"model_state": self.model.state_dict(),
                    "config": asdict(self.cfg),
                    "history": self.history,
                    "train_time": getattr(self, "train_time", None)}, path)


# Values these fields had before they existed, so checkpoints written by an
# earlier version reload with exactly the architecture and loss they trained
# under rather than picking up the new defaults.
_LEGACY_CONFIG = {"gating": "linear", "gate_tau": None, "normalize_inputs": False,
                  "lambda_energy": 0.0, "lambda_ampere": 0.0,
                  "field_checkpoint": False, "ampere_slices": None,
                  "velocity_quadrature": "grid", "n_velocity_nodes": 256,
                  "resample_velocity": True,
                  "proposal_widen": 1.25}


def load_history(path):
    """(history, train_time) from a checkpoint; (None, None) if not stored."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return blob.get("history"), blob.get("train_time")


def load_model(path, device=None):
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    stored = dict(blob["config"])
    for key, val in _LEGACY_CONFIG.items():
        stored.setdefault(key, val)
    cfg = TrainConfig(**stored)
    device = pick_device(cfg.device if device is None else device)
    dtype = default_dtype(device)
    problem = make_problem(cfg.problem)
    model = build_model(cfg, problem, device, dtype)
    model.load_state_dict(blob["model_state"])
    model.eval()
    f_fn = reconstruct_f(model, problem.f0, cfg.change_of_variables)
    return model, problem, f_fn, cfg


### reference solver: spectral semi-Lagrangian (4D)

@dataclass
class ReferenceSolution:
    times: torch.Tensor
    f: torch.Tensor
    x_grid: torch.Tensor
    v_grid: torch.Tensor
    e_grid: torch.Tensor


def _slab_width(elems_per_index, chunk_elems) -> int:
    return max(1, chunk_elems // max(1, elems_per_index))


def _shift_axis(f, dim, phase, chunk_dim, chunk_elems):
    """In-place exact FFT phase shift of f along `dim`, applied in slabs."""
    n = f.shape[chunk_dim]
    step = _slab_width(f.numel() // n, chunk_elems)
    sliced = phase.shape[chunk_dim] == n
    for s in range(0, n, step):
        w = min(step, n - s)
        blk = f.narrow(chunk_dim, s, w)
        ph = phase.narrow(chunk_dim, s, w) if sliced else phase
        blk.copy_(torch.fft.ifft(torch.fft.fft(blk, dim=dim) * ph, dim=dim).real)
    return f


def solve_reference(problem, nx=48, nv=48, dt=1.0 / 64.0, save_times=None,
                    device="cpu", dtype=torch.float64, chunk_mib=64.0
                    ) -> ReferenceSolution:
    """4D spectral semi-Lagrangian solver (Strang splitting, exact FFT shifts)."""
    device = torch.device(device)
    length = problem.length
    x = uniform_grid(problem.x_lo, problem.x_hi, nx, device, dtype)
    v = uniform_grid(problem.v_lo, problem.v_hi, nv, device, dtype)
    v_len = problem.v_hi - problem.v_lo

    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=length / nx, device=device, dtype=dtype)
    kv = 2.0 * torch.pi * torch.fft.fftfreq(nv, d=v_len / nv, device=device, dtype=dtype)

    bytes_per_elem = 4 * (torch.finfo(dtype).bits // 8)
    chunk_elems = max(1, int(chunk_mib * 2 ** 20 / bytes_per_elem))

    f = torch.empty((nx, nx, nv, nv), device=device, dtype=dtype)
    step_x = _slab_width(5 * nx * nv * nv, chunk_elems)
    for s in range(0, nx, step_x):
        w = min(step_x, nx - s)
        mesh = torch.meshgrid(x[s:s + w], x, v, v, indexing="ij")
        z = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        f[s:s + w] = problem.f0(z).reshape(w, nx, nv, nv)
        del mesh, z

    dv2 = (v_len / nv) ** 2
    x_mesh = torch.stack([m.reshape(-1) for m in
                          torch.meshgrid(x, x, indexing="ij")], dim=-1)

    def drift(f_now, step):
        ph_x = torch.exp(-1j * kx.view(nx, 1, 1, 1) * v.view(1, 1, nv, 1) * step)
        _shift_axis(f_now, 0, ph_x, chunk_dim=1, chunk_elems=chunk_elems)
        ph_y = torch.exp(-1j * kx.view(1, nx, 1, 1) * v.view(1, 1, 1, nv) * step)
        _shift_axis(f_now, 1, ph_y, chunk_dim=0, chunk_elems=chunk_elems)
        return f_now

    def field(f_now, t_now):
        if problem.has_field:
            rho = f_now.sum(dim=(2, 3)) * dv2
            return electric_field_2d(rho, (length, length))
        if problem.external_field is not None:
            tt = torch.full((nx * nx,), t_now, device=device, dtype=dtype)
            e = problem.external_field(tt, x_mesh)
            return torch.stack([e[:, 0].view(nx, nx), e[:, 1].view(nx, nx)])
        return torch.zeros(2, nx, nx, device=device, dtype=dtype)

    def kick(f_now, e, step):
        ph_vx = torch.exp(-1j * e[0].view(nx, nx, 1, 1) * kv.view(1, 1, nv, 1) * step)
        _shift_axis(f_now, 2, ph_vx, chunk_dim=0, chunk_elems=chunk_elems)
        del ph_vx
        ph_vy = torch.exp(-1j * e[1].view(nx, nx, 1, 1) * kv.view(1, 1, 1, nv) * step)
        _shift_axis(f_now, 3, ph_vy, chunk_dim=0, chunk_elems=chunk_elems)
        return f_now

    if save_times is None:
        save_times = problem.plot_times or [problem.t_final]
    save_times = sorted(set([0.0] + [float(s) for s in save_times]))
    n_steps = max(1, round(problem.t_final / dt))
    dt = problem.t_final / n_steps
    save_steps = {round(s / dt): s for s in save_times}
    steps_to_save = sorted(save_steps)

    out_f = torch.empty((len(steps_to_save), nx, nx, nv, nv),
                        device=device, dtype=dtype)
    saved_e, saved_t = [], []

    def record(slot, t_now):
        out_f[slot].copy_(f)
        saved_e.append(field(f, t_now))
        saved_t.append(t_now)

    with torch.no_grad():
        if steps_to_save and steps_to_save[0] == 0:
            record(0, 0.0)
        for step in range(1, n_steps + 1):
            f = drift(f, 0.5 * dt)
            e = field(f, (step - 0.5) * dt)
            f = kick(f, e, dt)
            f = drift(f, 0.5 * dt)
            if step in save_steps:
                record(len(saved_t), step * dt)

    return ReferenceSolution(
        times=torch.tensor(saved_t, device=device, dtype=dtype),
        f=out_f, x_grid=x, v_grid=v, e_grid=torch.stack(saved_e))


def _reference_ok(f: torch.Tensor) -> bool:
    if not torch.isfinite(f).all():
        return False
    mass = f.reshape(f.shape[0], -1).sum(-1)
    return bool((mass > 1e-6 * mass.abs().max().clamp_min(1e-30)).all()
                and mass.max() > 0)


def _reference_fingerprint(problem, times, nx, nv, dt) -> str:
    """Hash of everything a cached reference solve depends on."""
    lo = torch.tensor(problem.domain_lo(), dtype=torch.float64)
    hi = torch.tensor(problem.domain_hi(), dtype=torch.float64)
    d = len(lo)
    frac = torch.tensor([[((i + 1) * (k + 2) % 17) / 17.0 for k in range(d)]
                         for i in range(16)], dtype=torch.float64)
    with torch.no_grad():
        f0_probe = problem.f0(lo + (hi - lo) * frac).reshape(-1).tolist()
    payload = {
        "name": problem.name, "dim": problem.dim,
        "space_dim": getattr(problem, "space_dim", 1),
        "t_final": problem.t_final, "has_field": problem.has_field,
        "external_field": problem.external_field is not None,
        "box": [float(q) for q in problem.domain_lo() + problem.domain_hi()],
        "nx": nx, "nv": nv, "dt": float(dt),
        "times": [round(float(t), 9) for t in times],
        "f0": [round(float(q), 10) for q in f0_probe],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def reference_on_grid(problem, times, nx, nv, cache_dir, device, dtype,
                      dt=1.0 / 64.0, solver_device="auto", verbose=False):
    """Reference f on the (nx, nx, nv, nv) grid, cached to disk."""
    cache_dir = Path(cache_dir)
    x = uniform_grid(problem.x_lo, problem.x_hi, nx, device, dtype)
    v = uniform_grid(problem.v_lo, problem.v_hi, nv, device, dtype)
    if problem.f_exact is not None:
        f = eval_on_mesh(lambda t, z: problem.f_exact(t, z), times, [x, x, v, v])
        return f, x, v

    key = _reference_fingerprint(problem, times, nx, nv, dt)
    cache = cache_dir / f"reference2d_{nx}x{nv}_n{len(times)}_{key}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        # The times are stored and re-checked too, so a hash collision or a
        # hand-copied file is caught rather than trusted.
        cached_t = blob.get("times")
        matches = (cached_t is not None and len(cached_t) == len(times)
                   and all(abs(float(a) - float(b)) < 1e-9
                           for a, b in zip(cached_t, times)))
        if matches and _reference_ok(blob["f"]):
            return (blob["f"].to(dtype).to(device),
                    blob["x"].to(dtype).to(device),
                    blob["v"].to(dtype).to(device))
        cache.unlink()

    save_times = [float(t) for t in times]
    # "auto": solve on the same CUDA device the eval tensors live on (falling
    # back to the default cuda:0, which may belong to another user, is what
    # produced a silent zero reference in earlier runs).  Pass "cpu" explicitly
    # to trade speed for the far larger host RAM -- the N^4 state is what caps
    # the reachable resolution, and only the numerical solver has that problem.
    # Results are cast back to `device`/`dtype` on return, so callers never see
    # a CPU or float64 tensor either way.
    if solver_device == "auto":
        dev = str(device) if torch.device(device).type == "cuda" else "cpu"
    else:
        dev = solver_device
    sdtype = solver_dtype(torch.device(dev))
    try:
        if verbose:
            print(f"solving reference {nx}x{nv} on {dev} ...", flush=True)
        ref = solve_reference(problem, nx=nx, nv=nv, dt=dt,
                              save_times=save_times, device=dev, dtype=sdtype)
        if not _reference_ok(ref.f):
            raise RuntimeError("degenerate solve")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        # OutOfMemoryError subclasses RuntimeError, so this also catches real
        # bugs; re-raise anything that is not plainly a memory problem rather
        # than silently re-running the whole solve on the CPU and failing again
        # with a confusing traceback.
        if not isinstance(exc, torch.cuda.OutOfMemoryError) \
                and "degenerate solve" not in str(exc) \
                and "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print(f"  reference {nx}x{nv} failed on {dev} ({type(exc).__name__})"
                  f" -> retrying on cpu", flush=True)
        ref = solve_reference(problem, nx=nx, nv=nv, dt=dt,
                              save_times=save_times, device="cpu",
                              dtype=solver_dtype(torch.device("cpu")))
        if not _reference_ok(ref.f):
            raise RuntimeError(
                f"reference solve for {problem.name} is degenerate on cpu too;"
                f" not caching") from exc
    # solve_reference always saves the t=0 snapshot; select the frames matching
    # the requested times (in order) so the result aligns with `times`.
    sel = torch.stack([(ref.times - float(t)).abs().argmin() for t in times])
    f_sel = ref.f.index_select(0, sel)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"f": f_sel.cpu(), "x": ref.x_grid.cpu(), "v": ref.v_grid.cpu(),
                "times": [float(t) for t in times]}, cache)
    return (f_sel.to(device=device, dtype=dtype),
            ref.x_grid.to(device=device, dtype=dtype),
            ref.v_grid.to(device=device, dtype=dtype))


### cost scaling: grid solver (N^4) vs flow map (resolution-free)

def solver_footprint_bytes(nx, nv, n_saves=1, dtype_bytes=8) -> dict:
    """Bytes the 4D grid solver must hold, split by contributor."""
    cells = nx * nx * nv * nv
    return {"cells": cells,
            "state": cells * dtype_bytes,
            "saved": n_saves * cells * dtype_bytes,
            "total": (1 + n_saves) * cells * dtype_bytes}


def benchmark_scaling(problem, grid_sizes, n_steps=8, device=None,
                      dtype=torch.float64, verbose=True) -> list:
    """Wall-clock and memory of the grid solver as the resolution grows."""
    rows = []
    for n in grid_sizes:
        fp = solver_footprint_bytes(n, n, n_saves=1,
                                    dtype_bytes=torch.finfo(dtype).bits // 8)
        dev = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")
        used, err = dev, None
        t0 = time.time()
        try:
            if torch.device(dev).type == "cuda":
                torch.cuda.synchronize()
            ran = _bench_steps(problem, n, n_steps, dev, dtype)
            if torch.device(dev).type == "cuda":
                torch.cuda.synchronize()
        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as exc:
            # GPU cannot hold the N^4 state -> retry on CPU, which has more RAM
            # but is far slower.  This is the wall the grid method hits and the
            # flow map does not.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            err = type(exc).__name__
            used = "cpu (gpu OOM)"
            t0 = time.time()
            ran = _bench_steps(problem, n, n_steps, "cpu", dtype)
        dt_s = time.time() - t0
        rows.append({"n": n, "cells": fp["cells"], "gib": fp["total"] / 2 ** 30,
                     "device": used, "sec_per_step": dt_s / ran, "steps": ran,
                     "fell_back": err is not None})
        if verbose:
            print(f"  N={n:4d}  cells={fp['cells']:>13,}  "
                  f"state+save={fp['total'] / 2 ** 30:8.3f} GiB  "
                  f"{used:>14}  {dt_s / n_steps:8.3f} s/step", flush=True)
    return rows


def _bench_steps(problem, n, n_steps, device, dtype) -> int:
    """Run the solver at resolution n and return how many steps it ACTUALLY did."""
    dt = problem.t_final / max(1, round(problem.t_final * 64))
    solve_reference(problem, nx=n, nv=n, dt=dt,
                    save_times=[float(problem.t_final)], device=device,
                    dtype=dtype, chunk_mib=64.0)
    return max(1, round(problem.t_final / dt))


### diagnostics

@dataclass
class Conservation:
    times: torch.Tensor
    mass: torch.Tensor
    momentum: torch.Tensor
    energy: torch.Tensor
    e_rms: torch.Tensor


def conservation_from_f_grid_2d(f, times, x_grids, v_grids, lengths,
                                has_field=True) -> Conservation:
    """Invariants on the [Nt, Nx, Ny, Nvx, Nvy] tensor grid."""
    dx = (lengths[0] / f.shape[1]) * (lengths[1] / f.shape[2])
    wu = trapezoid_weights(v_grids[0])
    ww = trapezoid_weights(v_grids[1])
    rho = torch.einsum("txyuw,u,w->txy", f, wu, ww)
    if has_field:
        e = electric_field_2d(rho, lengths)
    else:
        e = torch.zeros(rho.shape[0], 2, *rho.shape[1:],
                        device=rho.device, dtype=rho.dtype)
    mass = rho.sum(dim=(1, 2)) * dx
    px = torch.einsum("txyuw,u,u,w->t", f, v_grids[0], wu, ww) * dx
    py = torch.einsum("txyuw,u,w,w->t", f, wu, v_grids[1], ww) * dx
    momentum = torch.sqrt(px ** 2 + py ** 2)
    vsq = v_grids[0].view(-1, 1) ** 2 + v_grids[1].view(1, -1) ** 2
    kinetic = 0.5 * torch.einsum("txyuw,uw,u,w->t", f, vsq, wu, ww) * dx
    field_e = 0.5 * (e ** 2).sum(dim=(1, 2, 3)) * dx
    e_rms = ((e ** 2).sum(dim=1).mean(dim=(1, 2))).sqrt()
    return Conservation(times, mass, momentum, kinetic + field_e, e_rms)


def evaluate_f_on_grids(f_fn, times, grids, chunk_size=1 << 18):
    with torch.no_grad():
        return eval_on_mesh(f_fn, times, grids, chunk_size)


def error_table(f_model, f_ref, times) -> list:
    rows = []
    for i, t in enumerate(times.tolist()):
        diff = f_model[i] - f_ref[i]
        rel = (diff.pow(2).sum().sqrt()
               / f_ref[i].pow(2).sum().sqrt().clamp_min(1e-30)).item()
        rows.append({"t": t, "l2": diff.pow(2).mean().sqrt().item(),
                     "linf": diff.abs().max().item(), "rel_l2": rel})
    return rows


def det_statistics(model, problem, times, n_points=512, seed=0, chunk=256):
    """Distribution of |det grad_z Psi - 1| over phase space"""
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lo = torch.tensor(problem.domain_lo())
    hi = torch.tensor(problem.domain_hi())
    z = (lo + (hi - lo) * torch.rand(n_points, problem.dim, generator=gen)) \
        .to(device=device, dtype=torch.float64)
    m64 = model.double()
    rows = []
    for t in times.tolist():
        errs = []
        for i in range(0, n_points, chunk):
            z_i = z[i:i + chunk]
            t_i = torch.full((len(z_i),), t, device=device, dtype=torch.float64)
            jac = m64.spatial_jacobian(t_i, z_i)
            errs.append((torch.linalg.det(jac) - 1.0).abs())
        e = torch.cat(errs)
        rows.append({"t": t, "median": e.median().item(),
                     "max": e.max().item()})
    model.float()
    return rows

# characteristics

def characteristics_reference(problem, z0, dt=1.0 / 64.0, nx=48, nv=48,
                              device="cpu", dtype=torch.float64):
    """Tracer trajectories z(t) advanced with the same leapfrog and field as
    the reference solver. Returns (times [Ns], traj [Ns, n_traj, 4])."""
    device = torch.device(device)
    length = problem.length
    x = uniform_grid(problem.x_lo, problem.x_hi, nx, device, dtype)
    v = uniform_grid(problem.v_lo, problem.v_hi, nv, device, dtype)
    v_len = problem.v_hi - problem.v_lo
    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=length / nx, device=device, dtype=dtype)
    kv = 2.0 * torch.pi * torch.fft.fftfreq(nv, d=v_len / nv, device=device, dtype=dtype)

    mesh = torch.meshgrid(x, x, v, v, indexing="ij")
    zz = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    f = problem.f0(zz).reshape(nx, nx, nv, nv)
    dv2 = (v_len / nv) ** 2
    x_mesh = torch.stack([m.reshape(-1) for m in
                          torch.meshgrid(x, x, indexing="ij")], dim=-1)

    def drift(fn, step):
        ph_x = torch.exp(-1j * kx.view(nx, 1, 1, 1) * v.view(1, 1, nv, 1) * step)
        fn = torch.fft.ifftn(torch.fft.fftn(fn, dim=(0,)) * ph_x, dim=(0,)).real
        ph_y = torch.exp(-1j * kx.view(1, nx, 1, 1) * v.view(1, 1, 1, nv) * step)
        return torch.fft.ifftn(torch.fft.fftn(fn, dim=(1,)) * ph_y, dim=(1,)).real

    def field(fn, t_now):
        if problem.has_field:
            rho = fn.sum(dim=(2, 3)) * dv2
            return electric_field_2d(rho, (length, length))
        if problem.external_field is not None:
            tt = torch.full((nx * nx,), t_now, device=device, dtype=dtype)
            e = problem.external_field(tt, x_mesh)
            return torch.stack([e[:, 0].view(nx, nx), e[:, 1].view(nx, nx)])
        return torch.zeros(2, nx, nx, device=device, dtype=dtype)

    def kick(fn, e, step):
        ph_vx = torch.exp(-1j * e[0].view(nx, nx, 1, 1) * kv.view(1, 1, nv, 1) * step)
        fn = torch.fft.ifftn(torch.fft.fftn(fn, dim=(2,)) * ph_vx, dim=(2,)).real
        ph_vy = torch.exp(-1j * e[1].view(nx, nx, 1, 1) * kv.view(1, 1, 1, nv) * step)
        return torch.fft.ifftn(torch.fft.fftn(fn, dim=(3,)) * ph_vy, dim=(3,)).real

    def e_at(e_grid, xy):
        # bilinear-interpolate the two field components at particle positions
        vals = e_grid.unsqueeze(0)                      # [1, 2, Nx, Ny]
        idx = torch.zeros(len(xy), dtype=torch.long, device=device)
        return interp_periodic_2d(vals, (problem.x_lo, problem.x_lo),
                                  (length, length), xy, idx)

    n_steps = max(1, round(problem.t_final / dt))
    dt = problem.t_final / n_steps
    p = z0.to(device=device, dtype=dtype).clone()
    times = [0.0]
    traj = [p.clone()]
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            f = drift(f, 0.5 * dt)
            e = field(f, (step - 0.5) * dt)
            f = kick(f, e, dt)
            f = drift(f, 0.5 * dt)
            # advance tracers with the same half-drift / kick / half-drift
            p = torch.cat([p[:, 0:2] + 0.5 * dt * p[:, 2:4], p[:, 2:4]], dim=-1)
            ep = e_at(e, p[:, 0:2])
            p = torch.cat([p[:, 0:2], p[:, 2:4] + dt * ep], dim=-1)
            p = torch.cat([p[:, 0:2] + 0.5 * dt * p[:, 2:4], p[:, 2:4]], dim=-1)
            times.append(step * dt)
            traj.append(p.clone())
    return torch.tensor(times, device=device, dtype=dtype), torch.stack(traj)


def characteristics_map(model, z0, times):
    """Forward trajectories z(t) with Psi(t, z(t)) = z0, from the learned map."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    z0 = z0.to(device=device, dtype=dtype)
    out = []
    with torch.no_grad():
        for t in times.tolist():
            tt = torch.full((len(z0),), t, device=device, dtype=dtype)
            out.append(model.inverse(tt, z0))
    return torch.stack(out)


### small plotting / reporting helpers

def use_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "image.cmap": "viridis",
        "figure.max_open_warning": 0,
    })


def markdown_table(rows, columns=None, float_fmt="{:.4g}") -> str:
    if not rows:
        return ""
    cols = columns or list(dict.fromkeys(k for r in rows for k in r))

    def fmt(v):
        return float_fmt.format(v) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
