"""Semi-Lagrangian PINN for the 1D-1V Vlasov-Poisson system (Landau damping).

We solve the Vlasov-Poisson equations in two-dimensional phase space z = (x, v)
by learning the inverse characteristic flow map Psi(t, z) -> z0 and
reconstructing the distribution as f = f0(Psi). The electric field is obtained
from a spectral Poisson solve on an Eulerian grid at each iteration.

Three parameterisations of the map are provided:

    "symplectic"  area-preserving coupling flow, det(dPsi/dz) = 1 exactly;
                  mass and all Casimir invariants are conserved by construction.
    "normalizing" affine coupling flow with analytic log-determinant; mass is
                  conserved through the change-of-variables reconstruction
                  f = f0(Psi) |det dPsi/dz|.
    "soft"        unconstrained network Psi = z + t*NN(t, z) with a soft penalty
                  (det - 1)^2, as in the original formulation.

The reference solution is a spectral semi-Lagrangian scheme (Cheng-Knorr,
Strang splitting, exact FFT sub-steps, float64).

Requires only torch, numpy and matplotlib.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    return torch.float32 if device.type == "mps" else torch.float64


### problem definition

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _wrap(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
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
    space_dim: int = 1
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
        return [self.x_lo, self.v_lo]

    def domain_hi(self) -> list:
        return [self.x_hi, self.v_hi]


def _vlasov_velocity(t, z, E):
    return torch.stack([z[:, 1], E], dim=-1)


def _landau(alpha=0.1, k=1.0) -> Problem:
    def f0(z):
        return (_INV_SQRT_2PI * torch.exp(-0.5 * z[:, 1] ** 2)
                * (1.0 + alpha * torch.cos(k * z[:, 0])))

    return Problem(
        name="landau", dim=2, t_final=15.0,
        x_lo=-math.pi, x_hi=math.pi, v_lo=-6.0, v_hi=6.0,
        f0=f0, phase_velocity=_vlasov_velocity, has_field=True,
        plot_times=[3.75, 7.5, 11.25, 15.0],
    )


def _free_streaming() -> Problem:
    """E = 0; exact map (x - v t, v). Used as a check."""
    u = 0.5

    def f0(z):
        x = _wrap(z[:, 0], -math.pi, math.pi)
        return torch.exp(-0.5 * (x ** 2 + (z[:, 1] - u) ** 2))

    def velocity(t, z, E=None):
        v = z[:, 1]
        return torch.stack([v, torch.zeros_like(v)], dim=-1)

    def map_exact(t, z):
        return torch.stack([z[:, 0] - z[:, 1] * t, z[:, 1]], dim=-1)

    def f_exact(t, z):
        return f0(map_exact(t, z))

    return Problem(
        name="free_streaming", dim=2, t_final=2.0,
        x_lo=-math.pi, x_hi=math.pi, v_lo=-6.0, v_hi=6.0,
        f0=f0, phase_velocity=velocity, f_exact=f_exact, map_exact=map_exact,
        plot_times=[0.0, 1.0, 2.0],
    )


_PROBLEMS = {"landau": _landau, "free_streaming": _free_streaming}


def make_problem(name: str) -> Problem:
    if name not in _PROBLEMS:
        raise ValueError(f"unknown problem {name!r}; choose from "
                         f"{sorted(_PROBLEMS)}")
    return _PROBLEMS[name]()


### spectral Poisson (periodic 1D grid)

def wavenumbers(n: int, length: float, device, dtype) -> torch.Tensor:
    return 2.0 * torch.pi * torch.fft.fftfreq(n, d=length / n,
                                              device=device, dtype=dtype)


def electric_field(rho: torch.Tensor, length: float) -> torch.Tensor:
    """E on the same grid as rho (last dim), from E' = rho - mean(rho)."""
    n = rho.shape[-1]
    k = wavenumbers(n, length, rho.device, rho.dtype)
    rho_hat = torch.fft.fft(rho, dim=-1)
    ik = 1j * k
    ik_safe = torch.where(k == 0, torch.ones_like(ik), ik)
    e_hat = torch.where(k == 0, torch.zeros_like(rho_hat), rho_hat / ik_safe)
    return torch.fft.ifft(e_hat, dim=-1).real


def interp_periodic(values, x0, length, x_query, row_index=None):
    """Linear interpolation on a uniform periodic grid."""
    n = values.shape[-1]
    u = (x_query - x0) / (length / n)
    j0 = torch.floor(u).long()
    w = u - j0.to(u.dtype)
    j0 = j0 % n
    j1 = (j0 + 1) % n
    if values.dim() == 1:
        v0, v1 = values[j0], values[j1]
    else:
        v0, v1 = values[row_index, j0], values[row_index, j1]
    return v0 * (1.0 - w) + v1 * w


### velocity-space quadrature and charge density

def uniform_grid(lo, hi, n, device, dtype) -> torch.Tensor:
    step = (hi - lo) / n
    return lo + step * torch.arange(n, device=device, dtype=dtype)


def trapezoid_weights(v: torch.Tensor) -> torch.Tensor:
    dv = v[1] - v[0]
    w = torch.full_like(v, dv)
    w[0] = 0.5 * dv
    w[-1] = 0.5 * dv
    return w


def _f_on_slices(f_fn, t_slices, x_grid, v_grid, chunk_size=1 << 18):
    """f[Nt, Nx, Nv] on the tensor grid (t_slices x x_grid x v_grid)."""
    nt, nx, nv = len(t_slices), len(x_grid), len(v_grid)
    tt = t_slices.view(nt, 1, 1).expand(nt, nx, nv).reshape(-1)
    xx = x_grid.view(1, nx, 1).expand(nt, nx, nv).reshape(-1)
    vv = v_grid.view(1, 1, nv).expand(nt, nx, nv).reshape(-1)
    z = torch.stack([xx, vv], dim=-1)
    outs = []
    for i in range(0, len(tt), chunk_size):
        outs.append(f_fn(tt[i:i + chunk_size], z[i:i + chunk_size]))
    return torch.cat(outs).view(nt, nx, nv)


def rho_on_grid(f_fn, t_slices, x_grid, v_grid, v_weights, chunk_size=1 << 18):
    """rho[Nt, Nx] = sum_v f(t, x, v) w_v."""
    f = _f_on_slices(f_fn, t_slices, x_grid, v_grid, chunk_size)
    return torch.einsum("txv,v->tx", f, v_weights)


def moments_on_grid(f_fn, t_slices, x_grid, v_grid, v_weights,
                    chunk_size=1 << 18):
    """First three velocity moments on the grid, each [Nt, Nx].
    rho = int f dv,  J = int v f dv,  K = int v^2/2 f dv.
    """
    f = _f_on_slices(f_fn, t_slices, x_grid, v_grid, chunk_size)
    return (torch.einsum("txv,v->tx", f, v_weights),
            torch.einsum("txv,v,v->tx", f, v_grid, v_weights),
            0.5 * torch.einsum("txv,v,v->tx", f, v_grid ** 2, v_weights))


def evaluate_f_on_grid(f_fn, times, x_grid, v_grid, chunk_size=1 << 18):
    """Sample f_fn on (times x x_grid x v_grid) -> [Nt, Nx, Nv]."""
    nt, nx, nv = len(times), len(x_grid), len(v_grid)
    tt = times.view(nt, 1, 1).expand(nt, nx, nv).reshape(-1)
    xx = x_grid.view(1, nx, 1).expand(nt, nx, nv).reshape(-1)
    vv = v_grid.view(1, 1, nv).expand(nt, nx, nv).reshape(-1)
    z = torch.stack([xx, vv], dim=-1)
    outs = []
    with torch.no_grad():
        for i in range(0, len(tt), chunk_size):
            outs.append(f_fn(tt[i:i + chunk_size], z[i:i + chunk_size]))
    return torch.cat(outs).view(nt, nx, nv)


### flow-map models

def _periodic_features(x: torch.Tensor, half_length: float) -> torch.Tensor:
    w = torch.pi / half_length
    return torch.stack([torch.sin(w * x), torch.cos(w * x)], dim=-1)


class CoordEmbed(nn.Module):
    """(sin, cos) features for a periodic coordinate, else the scaled raw value."""

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
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, z):
        raise NotImplementedError

    def spatial_jacobian(self, t, z):
        def single(t_i, z_i):
            return self.forward(t_i.unsqueeze(0), z_i.unsqueeze(0)).squeeze(0)
        return torch.func.vmap(torch.func.jacrev(single, argnums=1))(t, z)

    def logdet(self, t, z):
        jac = self.spatial_jacobian(t, z)
        return torch.linalg.slogdet(jac).logabsdet

    def inverse(self, t, z0, newton_iters=30, tol=1e-10):
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
    """Additive or affine coupling on the (x, v) split."""

    def __init__(self, target, cond_periodic_half_length, width, depth,
                 affine, max_log_scale=1.0, t_scale=1.0, cond_scale=1.0,
                 gate_tau=1.0, gating="split"):
        super().__init__()
        self.target = target
        self.affine = affine
        self.max_log_scale = max_log_scale
        self.t_scale = t_scale
        self.gate_tau = gate_tau
        self.gating = gating
        self.embed = CoordEmbed(cond_periodic_half_length, cond_scale)
        out = 2 if affine else 1
        self.net = MLP(self.embed.out_dim + 1, out, width, depth,
                       zero_init_last=True)
        self.shear = (MLP(self.embed.out_dim, 1, width, max(1, depth - 1),
                          zero_init_last=True) if gating == "split" else None)

    def _gate(self, t):
        if self.gating == "split":
            return -torch.expm1(-t / self.gate_tau)
        if self.gating == "saturating":
            return -torch.expm1(-t)
        return t                                    # "linear"

    def _params(self, t, other):
        """Total displacement (already including its t-gate) and log-scale."""
        h_other = self.embed(other)
        h = torch.cat([h_other, (t / self.t_scale).unsqueeze(-1)], dim=-1)
        out = self.net(h)
        disp = self._gate(t) * out[:, 0]
        if self.shear is not None:
            disp = disp + t * self.shear(h_other)[:, 0]
        if not self.affine:
            return disp, None
        log_scale = (self.max_log_scale * torch.tanh(out[:, 1])
                     * torch.tanh(t / self.t_scale))
        return disp, log_scale

    def forward_with_logdet(self, t, z):
        x, v = z[:, 0], z[:, 1]
        other = v if self.target == 0 else x
        disp, log_scale = self._params(t, other)
        upd = x if self.target == 0 else v
        if self.affine:
            upd = upd * torch.exp(log_scale) + disp
            ld = log_scale
        else:
            upd = upd + disp
            ld = torch.zeros_like(t)
        if self.target == 0:
            return torch.stack([upd, v], dim=-1), ld
        return torch.stack([x, upd], dim=-1), ld

    def forward(self, t, z):
        return self.forward_with_logdet(t, z)[0]

    def inverse(self, t, z):
        x, v = z[:, 0], z[:, 1]
        other = v if self.target == 0 else x
        disp, log_scale = self._params(t, other)
        upd = x if self.target == 0 else v
        if self.affine:
            upd = (upd - disp) * torch.exp(-log_scale)
        else:
            upd = upd - disp
        if self.target == 0:
            return torch.stack([upd, v], dim=-1)
        return torch.stack([x, upd], dim=-1)


class CouplingFlowMap(FlowMap):
    """Composition of alternating coupling layers.
    affine=False -> area-preserving (det = 1 exactly).
    affine=True  -> affine, with an exact log-determinant.
    """

    def __init__(self, dim, n_layers=8, width=96, depth=2,
                 periodic_half_length=None, affine=False, t_scale=1.0,
                 v_scale=1.0, gate_tau=1.0, gating="split"):
        super().__init__(dim)
        layers = []
        for i in range(n_layers):
            target = i % 2
            cond_periodic = periodic_half_length if target == 1 else None
            layers.append(_CouplingLayer(target, cond_periodic, width, depth,
                                         affine, t_scale=t_scale,
                                         cond_scale=v_scale, gate_tau=gate_tau,
                                         gating=gating))
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
        self.t_scale = t_scale
        self.v_scale = v_scale
        in_dim = self.embed.out_dim + 1 + 1     # (sin,cos)x + v + t
        self.net = MLP(in_dim, dim, width, depth, zero_init_last=True)

    def forward(self, t, z):
        h = torch.cat([self.embed(z[:, 0]), z[:, 1:] / self.v_scale,
                       (t / self.t_scale).unsqueeze(-1)], dim=-1)
        return z + t.unsqueeze(-1) * self.net(h)


def build_model(cfg, problem, device, dtype):
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
        model = CouplingFlowMap(problem.dim, cfg.n_layers, cfg.width, cfg.depth,
                                hl, affine=False, **kw)
    elif cfg.mode == "normalizing":
        model = CouplingFlowMap(problem.dim, cfg.n_layers, cfg.width, cfg.depth,
                                hl, affine=True, **kw)
    elif cfg.mode == "soft":
        model = MLPFlowMap(problem.dim, hl, cfg.width, cfg.depth,
                           t_scale=t_scale, v_scale=v_scale)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    return model.to(device=device, dtype=dtype)


### reconstruction and residuals

def reconstruct_f(model, f0, change_of_variables=False):
    if change_of_variables:
        def f_fn(t, z):
            z0, logdet = model.forward_with_logdet(t, z)
            return f0(z0) * torch.exp(logdet)
    else:
        def f_fn(t, z):
            return f0(model(t, z))
    return f_fn


def _jvp_along_flow(fn, t, z, velocity):
    _, tangent = torch.func.jvp(fn, (t, z), (torch.ones_like(t), velocity))
    return tangent


def vector_residual(model, t, z, velocity):
    return _jvp_along_flow(lambda t_, z_: model(t_, z_), t, z, velocity)


def jacobian_penalty(model, t, z):
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
    problem: str = "landau"
    mode: str = "symplectic"          # symplectic | normalizing | soft
    iters: int = 6000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_slices: int = 16
    n_per_slice: int = 1280
    field_nx: int = 128
    field_nv: int = 128
    lambda_jac: float = 1e-3          # soft mode only
    # --- field self-consistency (see Trainer.step) -------------------------
    detach_field: bool = True         # Picard: no "shrink my own E" gradient
    lambda_energy: float = 1e-3       # penalty on total-energy drift
    lambda_ampere: float = 1.0        # penalty on d_t E + J - P_0/L = 0
    field_chunk: int = 1 << 16        # points per chunk in the field solve.
    #   Measured: this does NOT lower peak training memory (19.00 GiB at 256k
    #   vs 18.99 GiB at 64k for field_nx=field_nv=128, n_slices=16).  Every
    #   chunk's activations stay in the autograd graph until backward, so
    #   chunking only shrinks the transient forward working set.  The peak is
    #   set by the field solve's retained graph, which the Ampere jvp roughly
    #   doubles; lowering field_nx/field_nv or n_slices is what actually helps.
    # --- flow-map parameterisation (see _CouplingLayer) --------------------
    gating: str = "split"             # split | linear | saturating
    gate_tau: float | None = None     # transient gate scale; None -> t_final/5
    #   t_final/5 is only sensible when the forcing lives across the whole
    #   horizon.  Here it does not: for k = 1 the field is dead by t ~ 5 of 15,
    #   so the default tau = 3.0 leaves the gate still ramping through the one
    #   window that matters.  Measured E_rms at the first Landau bounce
    #   (t = 2), as a fraction of the reference, 3000 iters:
    #       tau = 3.0 (default)   0.17 / 0.27 / 0.26   (3 seeds)
    #       tau = 0.5             0.46 / 0.43          (2 seeds)
    #   at equal late-time drift (|drift(t=15)| <= 0.017 either way), so pass
    #   gate_tau=0.5 for this problem.  Set tau from the timescale over which
    #   the forcing actually acts, not from the horizon.
    normalize_inputs: bool = True     # feed t/t_final and v/v_max to the MLPs
    sampling: str = "uniform"         # uniform | support
    uniform_frac: float = 0.25
    oversample: int = 4
    weight_alpha: float = 0.5
    n_layers: int = 8
    width: int = 96
    depth: int = 2
    seed: int = 0
    device: str | None = None
    log_every: int = 500

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
            self.x_grid = uniform_grid(p.x_lo, p.x_hi, cfg.field_nx,
                                       self.device, self.dtype)
            self.v_grid = torch.linspace(p.v_lo, p.v_hi, cfg.field_nv,
                                         device=self.device, dtype=self.dtype)
            self.v_weights = trapezoid_weights(self.v_grid)
            self.dx = p.length / cfg.field_nx
            # invariants at t = 0, on the same quadrature the penalties use
            with torch.no_grad():
                t0 = torch.zeros(1, device=self.device, dtype=self.dtype)
                rho0, j0, k0 = moments_on_grid(lambda t, z: p.f0(z), t0,
                                               self.x_grid, self.v_grid,
                                               self.v_weights)
                e0 = electric_field(rho0, p.length)
                self.w_field0 = (0.5 * (e0 ** 2).sum() * self.dx).clamp_min(1e-12)
                self.w_total0 = k0.sum() * self.dx + self.w_field0
                # <J> = P_0 / L, the constant of integration in Ampere's law
                # (see Trainer.step).  Under a self-consistent field momentum
                # is conserved exactly, so this is fixed once from f0 rather
                # than recomputed from the model -- a running mean would make
                # the zero mode of the penalty vacuous.
                self.j_bar0 = j0.mean()

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.history = []

    def _fields(self, t_slices):
        """E on the Eulerian grid per time slice, with the moments (and dE/dt
        when the Ampere penalty is on) needed by the conservation terms."""
        p = self.problem

        def moments(ts):
            return moments_on_grid(self.f_fn, ts, self.x_grid, self.v_grid,
                                   self.v_weights, self.cfg.field_chunk)

        if self.cfg.lambda_ampere > 0:
            (rho, J, K), (drho_dt, _, _) = torch.func.jvp(
                moments, (t_slices,), (torch.ones_like(t_slices),))
            # The Poisson solve is linear in rho, so d_t E = E[d_t rho]; taking
            # d/dt before the solve rather than through it keeps the FFT out of
            # the forward-mode trace (backward through a dual-number FFT is not
            # supported by the CPU/MKL backend).
            return (electric_field(rho, p.length), J, K,
                    electric_field(drho_dt, p.length))
        rho, J, K = moments(t_slices)
        return electric_field(rho, p.length), J, K, None

    def step(self) -> dict:
        cfg, p = self.cfg, self.problem
        batch = sample_collocation(self.problem, cfg.n_slices, cfg.n_per_slice,
                                   self.device, self.dtype,
                                   weight_fn=self._weight_fn,
                                   uniform_frac=cfg.uniform_frac,
                                   oversample=cfg.oversample,
                                   weight_alpha=cfg.weight_alpha)
        rec = {}
        e = None
        if p.has_field:
            e_grid, J, K, de_dt = self._fields(batch.t_slices)
            # The characteristic residual sees the field as a *fixed* target
            # (Picard).  Left attached, the cheapest way to lower that residual
            # is for the map to flatten its own rho and drive E -> 0: E enters
            # the residual additively as E * d_v Psi = E * (-t, 1), so for any
            # map that is not already the exact response, |E| > 0 only costs
            # loss.  Detaching removes that degenerate descent direction; the
            # conservation terms below then supply the *correct* gradient path
            # to the field, and they are deliberately left attached.
            e_used = e_grid.detach() if cfg.detach_field else e_grid
            e = interp_periodic(e_used, p.x_lo, p.length, batch.z[:, 0],
                                batch.slice_index)
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
            # Total energy is the one global invariant the E -> 0 collapse
            # actually violates, and no local residual implies it: a map that
            # phase-mixes rho flat conserves its kinetic energy exactly, so the
            # whole initial field energy simply disappears.  Measured relative
            # to that field energy, so eps_W = -1 is a full collapse.
            w = (K.sum(dim=1) + 0.5 * (e_grid ** 2).sum(dim=1)) * self.dx
            eps_w = (w - self.w_total0) / self.w_field0
            # Huber with the knee at eps_w = 1, i.e. exactly the scale of a
            # full collapse: quadratic over the range that matters, linear
            # beyond, so the large transient of the first few steps (eps_w can
            # reach ~1e2 before the map settles) cannot swamp the residual.
            loss_energy = torch.nn.functional.smooth_l1_loss(
                eps_w, torch.zeros_like(eps_w), beta=1.0)
            loss = loss + cfg.lambda_energy * loss_energy
            rec["loss_energy"] = loss_energy.item()
            rec["eps_w"] = eps_w.abs().max().item()

        if p.has_field and cfg.lambda_ampere > 0:
            # Ampere: d_t E + J - P_0/L = 0, the x-integrated form of Gauss +
            # continuity.  Exact solutions satisfy it identically, so it adds
            # no new physics -- but for an *approximate* map it is an
            # independent, mass-weighted projection of the residual onto the
            # moments that actually set E, with no weight in the empty |v|
            # tails and none of the growing (-t, 1) amplification.
            #
            # The constant of integration is <J> = P/L, and it is *fixed at
            # t = 0*, not recomputed from the model each step.  Subtracting the
            # running mean J.mean(dim=1) would force the x-mean of the residual
            # to vanish algebraically for any map, projecting the zero mode --
            # which is exactly the momentum -- out of the loss.  Against the
            # fixed j_bar0 the x-mean of res_a is (P(t) - P_0)/L, i.e. the
            # momentum drift itself, so the penalty constrains the one
            # invariant neither det = 1 (mass) nor lambda_energy protects.
            res_a = de_dt + J - self.j_bar0
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
        torch.save({"model_state": self.model.state_dict(),
                    "config": asdict(self.cfg)}, path)


# Values these fields had before they existed, so checkpoints written by an
# earlier version reload with exactly the architecture and loss they trained
# under rather than picking up the new defaults.
_LEGACY_CONFIG = {"gating": "linear", "gate_tau": None, "normalize_inputs": False,
                  "lambda_energy": 0.0, "lambda_ampere": 0.0}


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


### reference solver: spectral semi-Lagrangian (1D-1V)

@dataclass
class ReferenceSolution:
    times: torch.Tensor
    f: torch.Tensor
    x_grid: torch.Tensor
    v_grid: torch.Tensor
    e_grid: torch.Tensor


def _phase_shift_x(f, kx, v, dt):
    """f[Nx, Nv] <- f(x - v dt, v), exact spectral shift along x."""
    f_hat = torch.fft.fft(f, dim=0)
    phase = torch.exp(-1j * kx[:, None] * v[None, :] * dt)
    return torch.fft.ifft(f_hat * phase, dim=0).real


def _phase_shift_v(f, kv, e, dt):
    """f[Nx, Nv] <- f(x, v - E(x) dt), exact spectral shift along v."""
    f_hat = torch.fft.fft(f, dim=1)
    phase = torch.exp(-1j * e[:, None] * kv[None, :] * dt)
    return torch.fft.ifft(f_hat * phase, dim=1).real


def solve_reference(problem, nx=256, nv=256, dt=1.0 / 64.0, save_times=None,
                    device="cpu", dtype=torch.float64) -> ReferenceSolution:
    device = torch.device(device)
    length = problem.length
    x = uniform_grid(problem.x_lo, problem.x_hi, nx, device, dtype)
    v = uniform_grid(problem.v_lo, problem.v_hi, nv, device, dtype)
    v_len = problem.v_hi - problem.v_lo

    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=length / nx, device=device, dtype=dtype)
    kv = 2.0 * torch.pi * torch.fft.fftfreq(nv, d=v_len / nv, device=device, dtype=dtype)

    xx, vv = torch.meshgrid(x, v, indexing="ij")
    z = torch.stack([xx.reshape(-1), vv.reshape(-1)], dim=-1)
    f = problem.f0(z).reshape(nx, nv)
    dv = v_len / nv

    def field(f_now, t_now):
        if problem.has_field:
            rho = f_now.sum(dim=1) * dv
            return electric_field(rho, length)
        if problem.external_field is not None:
            tt = torch.full((nx,), t_now, device=device, dtype=dtype)
            return problem.external_field(tt, x.unsqueeze(-1)).reshape(nx)
        return torch.zeros(nx, device=device, dtype=dtype)

    if save_times is None:
        save_times = problem.plot_times or [problem.t_final]
    save_times = sorted(set([0.0] + [float(s) for s in save_times]))
    n_steps = max(1, round(problem.t_final / dt))
    dt = problem.t_final / n_steps
    save_steps = {round(s / dt): s for s in save_times}

    saved_f, saved_e, saved_t = [], [], []
    if 0 in save_steps:
        saved_f.append(f.clone()); saved_e.append(field(f, 0.0)); saved_t.append(0.0)

    with torch.no_grad():
        for step in range(1, n_steps + 1):
            f = _phase_shift_x(f, kx, v, 0.5 * dt)
            e = field(f, (step - 0.5) * dt)
            f = _phase_shift_v(f, kv, e, dt)
            f = _phase_shift_x(f, kx, v, 0.5 * dt)
            if step in save_steps:
                saved_f.append(f.clone())
                saved_e.append(field(f, step * dt))
                saved_t.append(step * dt)

    return ReferenceSolution(
        times=torch.tensor(saved_t, device=device, dtype=dtype),
        f=torch.stack(saved_f), x_grid=x, v_grid=v,
        e_grid=torch.stack(saved_e))


def _reference_ok(f: torch.Tensor) -> bool:
    if not torch.isfinite(f).all():
        return False
    mass = f.reshape(f.shape[0], -1).sum(-1)
    return bool((mass > 1e-6 * mass.abs().max().clamp_min(1e-30)).all()
                and mass.max() > 0)


def _reference_fingerprint(problem, times, nx, nv, dt) -> str:
    """Hash of everything a cached reference solve depends on."""
    span_x, span_v = problem.x_hi - problem.x_lo, problem.v_hi - problem.v_lo
    probe = torch.tensor(
        [[problem.x_lo + span_x * a, problem.v_lo + span_v * b]
         for a in (0.11, 0.37, 0.62, 0.89) for b in (0.13, 0.41, 0.68, 0.94)],
        dtype=torch.float64)
    with torch.no_grad():
        f0_probe = problem.f0(probe).reshape(-1).tolist()
    payload = {
        "name": problem.name, "dim": problem.dim,
        "t_final": problem.t_final, "has_field": problem.has_field,
        "external_field": problem.external_field is not None,
        "box": [problem.x_lo, problem.x_hi, problem.v_lo, problem.v_hi],
        "nx": nx, "nv": nv, "dt": float(dt),
        "times": [round(float(t), 9) for t in times],
        "f0": [round(float(q), 10) for q in f0_probe],
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def reference_on_grid(problem, times, nx, nv, cache_dir, device, dtype,
                      dt=1.0 / 64.0, verbose=False):
    """Reference f on the (nx, nv) grid, cached to disk."""
    cache_dir = Path(cache_dir)
    x = uniform_grid(problem.x_lo, problem.x_hi, nx, device, dtype)
    v = uniform_grid(problem.v_lo, problem.v_hi, nv, device, dtype)
    if problem.f_exact is not None:
        f = evaluate_f_on_grid(lambda t, z: problem.f_exact(t, z), times, x, v)
        return f, x, v

    key = _reference_fingerprint(problem, times, nx, nv, dt)
    cache = cache_dir / f"reference_{nx}x{nv}_n{len(times)}_{key}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        # The times are also stored and re-checked, so a hash collision or a
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
    dev = str(device) if torch.device(device).type == "cuda" else "cpu"
    sdtype = solver_dtype(torch.device(dev))
    try:
        if verbose:
            print(f"solving reference {nx}x{nv} on {dev} ...", flush=True)
        ref = solve_reference(problem, nx=nx, nv=nv, dt=dt,
                              save_times=save_times, device=dev, dtype=sdtype)
        if not _reference_ok(ref.f):
            raise RuntimeError("degenerate solve")
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ref = solve_reference(problem, nx=nx, nv=nv, dt=dt,
                              save_times=save_times, device="cpu")
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


### diagnostics

@dataclass
class Conservation:
    times: torch.Tensor
    mass: torch.Tensor
    momentum: torch.Tensor
    energy: torch.Tensor
    e_rms: torch.Tensor


def conservation_from_f_grid(f, times, x_grid, v_grid, length,
                             has_field=True) -> Conservation:
    """Invariants on the [Nt, Nx, Nv] tensor grid."""
    dx = length / f.shape[1]
    wv = trapezoid_weights(v_grid)
    rho = torch.einsum("txv,v->tx", f, wv)
    e = electric_field(rho, length) if has_field else torch.zeros_like(rho)
    mass = rho.sum(dim=1) * dx
    momentum = torch.einsum("txv,v,v->t", f, v_grid, wv) * dx
    kinetic = 0.5 * torch.einsum("txv,v,v->t", f, v_grid ** 2, wv) * dx
    field_e = 0.5 * (e ** 2).sum(dim=1) * dx
    e_rms = ((e ** 2).mean(dim=1)).sqrt()
    return Conservation(times, mass, momentum, kinetic + field_e, e_rms)


def error_table(f_model, f_ref, times) -> list:
    rows = []
    for i, t in enumerate(times.tolist()):
        diff = f_model[i] - f_ref[i]
        rel = (diff.pow(2).sum().sqrt()
               / f_ref[i].pow(2).sum().sqrt().clamp_min(1e-30)).item()
        rows.append({"t": t, "l2": diff.pow(2).mean().sqrt().item(),
                     "linf": diff.abs().max().item(), "rel_l2": rel})
    return rows


def det_statistics(model, problem, times, n_points=1024, seed=0, chunk=512):
    """Distribution of |det grad_z Psi - 1|, measured in double precision."""
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
        rows.append({"t": t, "median": e.median().item(), "max": e.max().item()})
    model.float()
    return rows


def fit_damping_rate(times, e_rms):
    """Slope gamma of the E_rms local maxima on a log scale."""
    t, e = times.cpu(), e_rms.cpu().clamp_min(1e-14)
    peaks = [(t[i].item(), e[i].item()) for i in range(1, len(t) - 1)
             if e[i] >= e[i - 1] and e[i] >= e[i + 1]]
    if len(peaks) < 2:
        return None
    tp = torch.tensor([p[0] for p in peaks])
    lp = torch.tensor([p[1] for p in peaks]).log()
    A = torch.stack([tp, torch.ones_like(tp)], dim=1)
    return torch.linalg.lstsq(A, lp.unsqueeze(1)).solution[0].item()


### characteristics

def characteristics_reference(problem, z0, dt=1.0 / 64.0, nx=256, nv=256,
                              device="cpu", dtype=torch.float64):
    """Tracer trajectories advanced with the same leapfrog and field as the
    reference solver. Returns (times [Ns], traj [Ns, n_traj, 2])."""
    device = torch.device(device)
    length = problem.length
    x = uniform_grid(problem.x_lo, problem.x_hi, nx, device, dtype)
    v = uniform_grid(problem.v_lo, problem.v_hi, nv, device, dtype)
    v_len = problem.v_hi - problem.v_lo
    kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=length / nx, device=device, dtype=dtype)
    kv = 2.0 * torch.pi * torch.fft.fftfreq(nv, d=v_len / nv, device=device, dtype=dtype)

    xx, vv = torch.meshgrid(x, v, indexing="ij")
    f = problem.f0(torch.stack([xx.reshape(-1), vv.reshape(-1)], dim=-1)).reshape(nx, nv)
    dv = v_len / nv

    def field(f_now):
        if problem.has_field:
            return electric_field(f_now.sum(dim=1) * dv, length)
        return torch.zeros(nx, device=device, dtype=dtype)

    n_steps = max(1, round(problem.t_final / dt))
    dt = problem.t_final / n_steps
    p = z0.to(device=device, dtype=dtype).clone()
    times, traj = [0.0], [p.clone()]
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            f = _phase_shift_x(f, kx, v, 0.5 * dt)
            e = field(f)
            f = _phase_shift_v(f, kv, e, dt)
            f = _phase_shift_x(f, kx, v, 0.5 * dt)
            xp = p[:, 0] + 0.5 * dt * p[:, 1]
            ep = interp_periodic(e, problem.x_lo, length, xp)
            vp = p[:, 1] + dt * ep
            xp = xp + 0.5 * dt * vp
            p = torch.stack([xp, vp], dim=-1)
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
