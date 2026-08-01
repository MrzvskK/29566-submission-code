"""JAX/Equinox port of the 2D-2V Vlasov-Poisson SL-PINN pipeline.

Ports the *headline* configuration only: the symplectic (det ∇Psi = 1 exactly)
flow map, trained with the Ampere + energy-drift conservation penalties and
support-weighted collocation sampling, plus the classical Strang-splitting
spectral semi-Lagrangian reference solver. Normalizing/soft flow-map modes and
the tensor-grid velocity quadrature ablation are intentionally out of scope.

This file exists to make the *training loop* JIT-compiled end to end (model
forward, forward-mode jvp residual, spectral Poisson solve, conservation
penalties, backward, optimizer update all fuse into one XLA program per step)
and to give the classical solver a real GPU->CPU fallback as the grid
resolution N grows -- the point of both is a "wall-clock vs N" comparison
where the classical solver's O(N^4) cost eventually crosses the (resolution
independent) SL-PINN training+inference cost. See the companion notebook
`slpinn_2d2v_jax.ipynb` for the full writeup and the crossover experiment.

Design invariant that must never be violated: `field_nx`/`field_nv` (the
quadrature grid used to solve the self-consistent field *during training*)
are fixed constants chosen once for field accuracy, and must never be a
function of the classical solver's comparison resolution `N` used in the
scaling sweep below -- that decoupling is the entire point of the argument.
"""

from __future__ import annotations

import functools
import json
import math
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import jax
jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

F32 = jnp.float32
F64 = jnp.float64


### device helpers

def gpu_device():
    gpus = jax.devices("gpu")
    return gpus[0] if gpus else None


def cpu_device():
    return jax.devices("cpu")[0]


def pick_device(preferred: str | None = None):
    if preferred == "cpu":
        return cpu_device()
    if preferred == "gpu":
        g = gpu_device()
        if g is None:
            raise RuntimeError("no GPU visible to JAX")
        return g
    return gpu_device() or cpu_device()


def is_oom(exc: Exception) -> bool:
    return isinstance(exc, jax.errors.JaxRuntimeError) and (
        "RESOURCE_EXHAUSTED" in str(exc) or "out of memory" in str(exc).lower())


def device_memory_stats(dev) -> dict:
    try:
        return dev.memory_stats() or {}
    except Exception:
        return {}


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    
### problem definitions

def _wrap(x, lo: float, hi: float):
    return jnp.mod(x - lo, hi - lo) + lo


@dataclass
class Problem:
    name: str
    dim: int
    space_dim: int
    t_final: float
    x_lo: float
    x_hi: float
    v_lo: float
    v_hi: float
    f0: Callable
    phase_velocity: Callable
    has_field: bool = False
    f_exact: Optional[Callable] = None
    map_exact: Optional[Callable] = None
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


_INV_2PI = 1.0 / (2.0 * math.pi)


def _blob_2d(u=(0.5, 0.0)):
    def f0(z):
        x = _wrap(z[..., 0], -6.0, 6.0)
        y = _wrap(z[..., 1], -6.0, 6.0)
        return _INV_2PI * jnp.exp(-0.5 * (x ** 2 + y ** 2
                                          + (z[..., 2] - u[0]) ** 2
                                          + (z[..., 3] - u[1]) ** 2))
    return f0


def _vlasov_velocity_2d(t, z, e):
    return jnp.concatenate([z[..., 2:4], e], axis=-1)


def gaussian_vp_2d() -> Problem:
    return Problem(name="gaussian_vp_2d", dim=4, space_dim=2, t_final=2.0,
                   x_lo=-6.0, x_hi=6.0, v_lo=-6.0, v_hi=6.0,
                   f0=_blob_2d((0.5, 0.0)), phase_velocity=_vlasov_velocity_2d,
                   has_field=True, plot_times=[0.0, 1.0, 2.0])


def free_streaming_2d() -> Problem:
    u = (0.5, 0.0)

    def f0(z):
        x = _wrap(z[..., 0], -6.0, 6.0)
        y = _wrap(z[..., 1], -6.0, 6.0)
        return jnp.exp(-0.5 * (x ** 2 + y ** 2
                               + (z[..., 2] - u[0]) ** 2 + (z[..., 3] - u[1]) ** 2))

    def velocity(t, z, e=None):
        v = z[..., 2:4]
        return jnp.concatenate([v, jnp.zeros_like(v)], axis=-1)

    def map_exact(t, z):
        ts = t[..., None]
        return jnp.concatenate([z[..., 0:2] - z[..., 2:4] * ts, z[..., 2:4]], axis=-1)

    def f_exact(t, z):
        return f0(map_exact(t, z))

    return Problem(name="free_streaming_2d", dim=4, space_dim=2, t_final=1.0,
                   x_lo=-6.0, x_hi=6.0, v_lo=-6.0, v_hi=6.0,
                   f0=f0, phase_velocity=velocity, f_exact=f_exact,
                   map_exact=map_exact, plot_times=[0.0, 0.5, 1.0])


_PROBLEMS = {"gaussian_vp_2d": gaussian_vp_2d, "free_streaming_2d": free_streaming_2d}


def make_problem(name: str) -> Problem:
    if name not in _PROBLEMS:
        raise ValueError(f"unknown problem {name!r}; choose from {sorted(_PROBLEMS)}")
    return _PROBLEMS[name]()


### spectral Poisson (periodic 2D grid) 
### -- always a FIXED small field_nx grid, never a function of the classical solver's comparison resolution N.


def wavenumbers(n: int, length: float, dtype=F32):
    return (2.0 * jnp.pi * jnp.fft.fftfreq(n, d=length / n)).astype(dtype)


def electric_field_2d(rho, lengths):
    nx, ny = rho.shape[-2], rho.shape[-1]
    kx = wavenumbers(nx, lengths[0], rho.dtype).reshape(nx, 1)
    ky = wavenumbers(ny, lengths[1], rho.dtype).reshape(1, ny)
    rho_hat = jnp.fft.fft2(rho, axes=(-2, -1))
    k2 = kx ** 2 + ky ** 2
    k2_safe = jnp.where(k2 == 0, jnp.ones_like(k2), k2)
    phi_hat = jnp.where(k2 == 0, jnp.zeros_like(rho_hat), rho_hat / k2_safe)
    ex = jnp.fft.ifft2(-1j * kx * phi_hat, axes=(-2, -1)).real
    ey = jnp.fft.ifft2(-1j * ky * phi_hat, axes=(-2, -1)).real
    return jnp.stack([ex, ey], axis=-3)


def interp_periodic_2d(values, x0s, lengths, xq, row_index):
    nx, ny = values.shape[-2], values.shape[-1]
    ux = (xq[:, 0] - x0s[0]) / (lengths[0] / nx)
    uy = (xq[:, 1] - x0s[1]) / (lengths[1] / ny)
    ix0 = jnp.floor(ux).astype(jnp.int32)
    iy0 = jnp.floor(uy).astype(jnp.int32)
    wx = (ux - ix0.astype(ux.dtype))[..., None]
    wy = (uy - iy0.astype(uy.dtype))[..., None]
    ix0, iy0 = ix0 % nx, iy0 % ny
    ix1, iy1 = (ix0 + 1) % nx, (iy0 + 1) % ny
    r = row_index
    v00 = values[r, :, ix0, iy0]
    v10 = values[r, :, ix1, iy0]
    v01 = values[r, :, ix0, iy1]
    v11 = values[r, :, ix1, iy1]
    return v00 * (1 - wx) * (1 - wy) + v10 * wx * (1 - wy) + v01 * (1 - wx) * wy + v11 * wx * wy


def longitudinal_part_2d(J, lengths):
    nx, ny = J.shape[-2], J.shape[-1]
    kx = wavenumbers(nx, lengths[0], J.dtype).reshape(nx, 1)
    ky = wavenumbers(ny, lengths[1], J.dtype).reshape(1, ny)
    k2 = kx ** 2 + ky ** 2
    k2_safe = jnp.where(k2 == 0, jnp.ones_like(k2), k2)
    j_hat = jnp.fft.fft2(J, axes=(-2, -1))
    k_dot_j = kx * j_hat[..., 0, :, :] + ky * j_hat[..., 1, :, :]
    coef = k_dot_j / k2_safe
    out = jnp.stack([kx * coef, ky * coef], axis=-3)
    out = jnp.where((k2 == 0)[..., None, :, :], j_hat, out)
    return jnp.fft.ifft2(out, axes=(-2, -1)).real


def uniform_grid(lo: float, hi: float, n: int, dtype=F32):
    step = (hi - lo) / n
    return lo + step * jnp.arange(n, dtype=dtype)


def trapezoid_weights(v):
    dv = v[1] - v[0]
    w = jnp.full_like(v, dv)
    w = w.at[0].set(0.5 * dv)
    w = w.at[-1].set(0.5 * dv)
    return w


# ---------------------------------------------------------------------------
# velocity quadrature: Cranley-Patterson rotated Sobol nodes.
#
# The PyTorch version redraws a fresh CPU-only torch.quasirandom.SobolEngine
# every training step, forcing a host round-trip each iteration. Here the
# Sobol base point set is built ONCE on the host (scipy) and transferred to
# device once; each step applies a toroidal shift (Cranley & Patterson, 1976)
# -- u_i <- frac(base_i + U), fresh U ~ Uniform each step -- which is a
# standard randomized-QMC technique: translation mod 1 of a low-discrepancy
# set is still low-discrepancy, and a fresh U gives an unbiased replication.
# Entirely on-device, no host round-trip per step.
# ---------------------------------------------------------------------------

def next_pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def sobol_base_points(space_dim: int, n_nodes: int, seed: int) -> jnp.ndarray:
    from scipy.stats import qmc
    m = max(1, math.ceil(math.log2(max(n_nodes, 1))))
    pts = qmc.Sobol(d=space_dim, scramble=True, seed=seed).random_base2(m=m)
    return jnp.asarray(pts, dtype=F32)


def rotate_qmc_nodes(base_u, key, mean, sigma):
    s = base_u.shape[-1]
    u_shift = jr.uniform(key, (s,), dtype=F32)
    u01 = jnp.mod(base_u + u_shift, 1.0)
    u01 = jnp.clip(u01, 1e-7, 1.0 - 1e-7)
    eps = jax.scipy.special.erfinv(2.0 * u01 - 1.0) * jnp.sqrt(jnp.array(2.0, dtype=F32))
    v = mean + sigma * eps
    log_q = (-0.5 * jnp.sum(eps ** 2, axis=-1) - jnp.sum(jnp.log(sigma))
             - 0.5 * s * jnp.log(jnp.array(2.0 * jnp.pi, dtype=F32)))
    m = base_u.shape[0]
    w = jnp.exp(-log_q) / m
    return v, w


# field evaluation on a fixed small (field_nx, field_nx) spatial grid crossed
# with an importance-sampled velocity point cloud -- 2D, not 4D, and constant
# cost regardless of the classical solver's N (see module docstring).

def eval_on_nodes_2d(f_fn, t_slices, x_grids, v_nodes):
    nt, nx, ny, m = t_slices.shape[0], x_grids[0].shape[0], x_grids[1].shape[0], v_nodes.shape[0]
    s = v_nodes.shape[-1]
    tt = jnp.broadcast_to(t_slices.reshape(nt, 1, 1, 1), (nt, nx, ny, m)).reshape(-1)
    xx = jnp.broadcast_to(x_grids[0].reshape(1, nx, 1, 1), (nt, nx, ny, m)).reshape(-1)
    yy = jnp.broadcast_to(x_grids[1].reshape(1, 1, ny, 1), (nt, nx, ny, m)).reshape(-1)
    vv = jnp.broadcast_to(v_nodes.reshape(1, 1, 1, m, s), (nt, nx, ny, m, s)).reshape(-1, s)
    z = jnp.concatenate([xx[:, None], yy[:, None], vv], axis=-1)
    f = f_fn(tt, z)
    return f.reshape(nt, nx, ny, m)


def moments_on_nodes_2d(f_fn, t_slices, x_grids, v_nodes, v_w):
    f = eval_on_nodes_2d(f_fn, t_slices, x_grids, v_nodes)
    rho = jnp.einsum("txym,m->txy", f, v_w)
    jvec = jnp.einsum("txym,m,mj->tjxy", f, v_w, v_nodes)
    vsq = jnp.sum(v_nodes ** 2, axis=-1)
    kin = 0.5 * jnp.einsum("txym,m,m->txy", f, v_w, vsq)
    return rho, jvec, kin


def eval_on_mesh(f_fn, t_slices, grids):
    shape = [len(t_slices)] + [len(g) for g in grids]
    mesh = jnp.meshgrid(t_slices, *grids, indexing="ij")
    tt = mesh[0].reshape(-1)
    z = jnp.stack([m.reshape(-1) for m in mesh[1:]], axis=-1)
    return f_fn(tt, z).reshape(*shape)


def moments_on_grid_2d(f_fn, t_slices, x_grids, v_grids, v_weights):
    f = eval_on_mesh(f_fn, t_slices, x_grids + v_grids)
    wu, ww = v_weights
    rho = jnp.einsum("txyuw,u,w->txy", f, wu, ww)
    jx = jnp.einsum("txyuw,u,u,w->txy", f, v_grids[0], wu, ww)
    jy = jnp.einsum("txyuw,u,w,w->txy", f, wu, v_grids[1], ww)
    vsq = v_grids[0].reshape(-1, 1) ** 2 + v_grids[1].reshape(1, -1) ** 2
    kin = 0.5 * jnp.einsum("txyuw,uw,u,w->txy", f, vsq, wu, ww)
    return rho, jnp.stack([jx, jy], axis=1), kin



### flow-map model (Equinox) -- symplectic (det = 1) coupling flow only.
#
# Written UNBATCHED (single t: scalar, z: [d]); batching is done by jax.vmap
# at call sites (the same idiom the torch version already uses for its
# Jacobian via vmap(jacrev(...))).

class CoordEmbed(eqx.Module):
    half_length: Optional[float] = eqx.field(static=True)
    scale: float = eqx.field(static=True, default=1.0)

    @property
    def out_dim(self) -> int:
        return 2 if self.half_length is not None else 1

    def __call__(self, x):
        if self.half_length is not None:
            w = jnp.pi / self.half_length
            return jnp.stack([jnp.sin(w * x), jnp.cos(w * x)])
        return jnp.stack([x / self.scale])


class MLP(eqx.Module):
    layers: tuple

    def __call__(self, h):
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        return self.layers[-1](h)


def make_mlp(key, in_dim, out_dim, width, depth, zero_init_last=False) -> MLP:
    keys = jr.split(key, depth + 1)
    dims = [in_dim] + [width] * depth + [out_dim]
    layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], dtype=F32)
              for i in range(depth + 1)]
    if zero_init_last:
        last = layers[-1]
        layers[-1] = eqx.tree_at(lambda l: (l.weight, l.bias), last,
                                 (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))
    return MLP(layers=tuple(layers))


def _gate_split(t, tau):
    return -jnp.expm1(-t / tau)


class CouplingLayer(eqx.Module):
    target: int = eqx.field(static=True)
    s: int = eqx.field(static=True)
    t_scale: float = eqx.field(static=True)
    gate_tau: float = eqx.field(static=True)
    embed: CoordEmbed
    net: MLP
    shear: MLP

    def _disp(self, t, other):
        feats = jnp.concatenate([self.embed(other[j]) for j in range(self.s)])
        h = jnp.concatenate([feats, jnp.stack([t / self.t_scale])])
        out = self.net(h)
        gate = _gate_split(t, self.gate_tau)
        return gate * out + t * self.shear(feats)

    def __call__(self, t, z):
        s = self.s
        x_block, v_block = z[:s], z[s:]
        other = v_block if self.target == 0 else x_block
        disp = self._disp(t, other)
        upd = (x_block if self.target == 0 else v_block) + disp
        return (jnp.concatenate([upd, v_block]) if self.target == 0
                else jnp.concatenate([x_block, upd]))

    def inverse(self, t, z):
        s = self.s
        x_block, v_block = z[:s], z[s:]
        other = v_block if self.target == 0 else x_block
        disp = self._disp(t, other)
        upd = (x_block if self.target == 0 else v_block) - disp
        return (jnp.concatenate([upd, v_block]) if self.target == 0
                else jnp.concatenate([x_block, upd]))


class CouplingFlowMap(eqx.Module):
    layers: tuple
    dim: int = eqx.field(static=True)

    def __call__(self, t, z):
        for layer in self.layers:
            z = layer(t, z)
        return z

    def spatial_jacobian(self, t, z):
        return jax.jacfwd(lambda zz: self(t, zz))(z)

    def inverse(self, t, z0, newton_iters=30, tol=1e-10):
        def cond(carry):
            z, i = carry
            r = self(t, z) - z0
            return jnp.logical_and(i < newton_iters, jnp.max(jnp.abs(r)) >= tol)

        def body(carry):
            z, i = carry
            r = self(t, z) - z0
            jac = self.spatial_jacobian(t, z)
            step = jnp.linalg.solve(jac, r)
            return (z - step, i + 1)

        z, _ = jax.lax.while_loop(cond, body, (z0, 0))
        return z


def build_model(key, dim, space_dim, n_layers, width, depth,
                periodic_half_length, t_scale, v_scale, gate_tau) -> CouplingFlowMap:
    keys = jr.split(key, 2 * n_layers)
    layers = []
    for i in range(n_layers):
        target = i % 2
        cond_periodic = periodic_half_length if target == 1 else None
        embed = CoordEmbed(half_length=cond_periodic, scale=v_scale)
        in_dim = space_dim * embed.out_dim + 1
        net = make_mlp(keys[2 * i], in_dim, space_dim, width, depth, zero_init_last=True)
        shear = make_mlp(keys[2 * i + 1], space_dim * embed.out_dim, space_dim,
                         width, max(1, depth - 1), zero_init_last=True)
        layers.append(CouplingLayer(target=target, s=space_dim, t_scale=t_scale,
                                    gate_tau=gate_tau, embed=embed, net=net, shear=shear))
    return CouplingFlowMap(layers=tuple(layers), dim=dim)


def flow_forward(model, t, z):
    return jax.vmap(lambda tt, zz: model(tt, zz))(t, z)


def flow_inverse(model, t, z0, newton_iters=30, tol=1e-10):
    return jax.vmap(lambda tt, zz: model.inverse(tt, zz, newton_iters, tol))(t, z0)


def flow_jacobian(model, t, z):
    return jax.vmap(lambda tt, zz: model.spatial_jacobian(tt, zz))(t, z)


def make_f_fn(model, f0):
    def f_fn(t, z):
        return f0(flow_forward(model, t, z))
    return f_fn


def vector_residual(model, t, z, velocity):
    def single(t_, z_, v_):
        _, tangent = jax.jvp(lambda tt, zz: model(tt, zz), (t_, z_), (jnp.ones_like(t_), v_))
        return tangent
    return jax.vmap(single)(t, z, velocity)


def smooth_l1(x, beta: float = 1.0):
    ax = jnp.abs(x)
    return jnp.where(ax < beta, 0.5 * x ** 2 / beta, ax - 0.5 * beta)


### training

@dataclass
class TrainConfig:
    problem: str = "gaussian_vp_2d"
    iters: int = 2500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_slices: int = 16
    n_per_slice: int = 512
    field_nx: int = 24
    field_nv: int = 24
    detach_field: bool = True
    lambda_energy: float = 1e-3
    lambda_ampere: float = 1.0
    n_velocity_nodes: int = 128
    proposal_widen: float = 1.25
    gate_tau: Optional[float] = None
    normalize_inputs: bool = True
    uniform_frac: float = 0.25
    oversample: int = 8
    weight_alpha: float = 1.0
    weight_floor: float = 1e-3
    n_layers: int = 8
    width: int = 64
    depth: int = 2
    seed: int = 0
    log_every: int = 200


class Trainer:
    def __init__(self, cfg: TrainConfig, device=None):
        self.cfg = cfg
        self.device = device or pick_device()
        with jax.default_device(self.device):
            self.problem = p = make_problem(cfg.problem)
            key = jr.PRNGKey(cfg.seed)
            key, mkey = jr.split(key)

            t_scale = p.t_final if cfg.normalize_inputs else 1.0
            v_scale = max(abs(p.v_lo), abs(p.v_hi), 1e-12) if cfg.normalize_inputs else 1.0
            tau = max(cfg.gate_tau if cfg.gate_tau is not None else p.t_final / 5.0, 1e-12)
            model = build_model(mkey, p.dim, p.space_dim, cfg.n_layers, cfg.width,
                                cfg.depth, p.periodic_half_length, t_scale, v_scale, tau)
            self.model = model

            x_grid = uniform_grid(p.x_lo, p.x_hi, cfg.field_nx)
            self.x_grids = [x_grid, x_grid]

            self.v_mean, self.v_sigma = self._fit_velocity_proposal()
            m_nodes = next_pow2(cfg.n_velocity_nodes)
            self.base_u = sobol_base_points(p.space_dim, m_nodes, cfg.seed)

            self.dx = (p.length / cfg.field_nx) ** 2
            self.lengths = (p.length, p.length)
            t0 = jnp.zeros(1, dtype=F32)
            v0, w0 = rotate_qmc_nodes(self.base_u, jr.PRNGKey(cfg.seed), self.v_mean, self.v_sigma)
            rho0, j0, k0 = moments_on_nodes_2d(lambda t, z: p.f0(z), t0, self.x_grids, v0, w0)
            e0 = electric_field_2d(rho0, self.lengths)
            self.w_field0 = jnp.clip(0.5 * jnp.sum(e0 ** 2) * self.dx, 1e-12, None)
            self.w_total0 = jnp.sum(k0) * self.dx + self.w_field0
            self.j_bar0 = jnp.mean(j0, axis=(-2, -1)).reshape(2, 1, 1)

            self.optimizer = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
            self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
            self.train_step = self._make_train_step()
            self.key = jr.PRNGKey(cfg.seed + 1)
        self.history = []
        self.train_time = None

    def _fit_velocity_proposal(self):
        p, cfg = self.problem, self.cfg
        fit_n = min(cfg.field_nx, cfg.field_nv, 24)
        x_grid = uniform_grid(p.x_lo, p.x_hi, fit_n)
        x_grids = [x_grid, x_grid]
        v_grid = jnp.linspace(p.v_lo, p.v_hi, fit_n, dtype=F32)
        v_grids = [v_grid, v_grid]
        v_weights = [trapezoid_weights(g) for g in v_grids]
        t0 = jnp.zeros(1, dtype=F32)
        f = eval_on_mesh(lambda t, z: p.f0(z), t0, x_grids + v_grids)
        wu, ww = v_weights
        tot = jnp.clip(jnp.einsum("txyuw,u,w->", f, wu, ww), 1e-30, None)
        gu, gw = v_grids
        mx = jnp.einsum("txyuw,u,u,w->", f, gu, wu, ww) / tot
        my = jnp.einsum("txyuw,u,w,w->", f, wu, gw, ww) / tot
        sx = jnp.sqrt(jnp.einsum("txyuw,u,u,w->", f, (gu - mx) ** 2, wu, ww) / tot)
        sy = jnp.sqrt(jnp.einsum("txyuw,u,w,w->", f, wu, (gw - my) ** 2, ww) / tot)
        mean = jnp.stack([mx, my])
        sigma = jnp.clip(jnp.stack([sx, sy]), 1e-3, None) * cfg.proposal_widen
        return mean, sigma

    def _make_train_step(self):
        p, cfg = self.problem, self.cfg
        x_grids, lengths, dx = self.x_grids, self.lengths, self.dx
        base_u, v_mean, v_sigma = self.base_u, self.v_mean, self.v_sigma
        w_field0, w_total0, j_bar0 = self.w_field0, self.w_total0, self.j_bar0
        lo = jnp.array(p.domain_lo(), dtype=F32)
        hi = jnp.array(p.domain_hi(), dtype=F32)
        d = p.dim
        n_slices, n_per_slice = cfg.n_slices, cfg.n_per_slice
        n_w = n_per_slice - round(cfg.uniform_frac * n_per_slice)
        n_u = n_per_slice - n_w
        m_cand = cfg.oversample * max(n_w, 1)
        optimizer = self.optimizer

        def sample_collocation(model, key):
            k1, k2, k3, k4 = jr.split(key, 4)
            u = jr.uniform(k1, (n_slices,), dtype=F32)
            t_slices = (jnp.arange(n_slices, dtype=F32) + u) * (p.t_final / n_slices)
            if n_w <= 0:
                z = lo + (hi - lo) * jr.uniform(k2, (n_slices, n_per_slice, d), dtype=F32)
            else:
                z_u = lo + (hi - lo) * jr.uniform(k2, (n_slices, n_u, d), dtype=F32)
                cand = lo + (hi - lo) * jr.uniform(k3, (n_slices, m_cand, d), dtype=F32)
                t_cand = jnp.repeat(t_slices, m_cand)
                f_fn = make_f_fn(model, p.f0)
                wf = jax.lax.stop_gradient(f_fn(t_cand, cand.reshape(-1, d))).reshape(n_slices, m_cand)
                w = jnp.clip(jnp.nan_to_num(wf), 0.0, None) ** cfg.weight_alpha
                w = w + cfg.weight_floor * jnp.mean(w, axis=1, keepdims=True)
                ok = jnp.sum(w, axis=1, keepdims=True) > 0
                w = jnp.where(ok, w, jnp.ones_like(w))
                g = jr.uniform(k4, (n_slices, m_cand), dtype=F32)
                gk = jnp.log(w) - jnp.log(-jnp.log(jnp.clip(g, 1e-12, 1 - 1e-6)))
                _, idx = jax.lax.top_k(gk, n_w)
                idx_full = jnp.broadcast_to(idx[..., None], (n_slices, n_w, d))
                z_w = jnp.take_along_axis(cand, idx_full, axis=1)
                z = jnp.concatenate([z_u, z_w], axis=1)
            z = z.reshape(n_slices * n_per_slice, d)
            slice_index = jnp.repeat(jnp.arange(n_slices), n_per_slice)
            t = t_slices[slice_index]
            return t, z, t_slices, slice_index

        def loss_fn(model, key):
            k_sample, k_qmc = jr.split(key)
            t, z, t_slices, slice_index = sample_collocation(model, k_sample)
            f_fn = make_f_fn(model, p.f0)
            v_nodes, v_w = rotate_qmc_nodes(base_u, k_qmc, v_mean, v_sigma)

            def moments(ts):
                return moments_on_nodes_2d(f_fn, ts, x_grids, v_nodes, v_w)

            moments_ckpt = jax.checkpoint(moments)
            rho, jvec, kin = moments_ckpt(t_slices)
            e_grid = electric_field_2d(rho, lengths)

            # The Ampere tangent pass is the single most expensive term in the
            # step, so skip it outright rather than scaling it by zero: `cfg` is
            # static under filter_jit, so this branch is resolved at trace time.
            if cfg.lambda_ampere > 0:
                (rho_a, j_a, _), (drho_a, _, _) = jax.jvp(
                    moments_ckpt, (t_slices,), (jnp.ones_like(t_slices),))
                de_dt = electric_field_2d(drho_a, lengths)
                res_a = de_dt + longitudinal_part_2d(j_a, lengths) - j_bar0
                loss_ampere = jnp.mean(res_a ** 2)
            else:
                loss_ampere = jnp.zeros((), F32)

            e_used = jax.lax.stop_gradient(e_grid) if cfg.detach_field else e_grid
            e = interp_periodic_2d(e_used, (p.x_lo, p.x_lo), lengths, z[:, :2], slice_index)
            velocity = p.phase_velocity(t, z, e)
            res = vector_residual(model, t, z, velocity)
            loss_char = jnp.mean(jnp.sum(res ** 2, axis=-1))

            w_now = (jnp.sum(kin, axis=(1, 2)) + 0.5 * jnp.sum(e_grid ** 2, axis=(1, 2, 3))) * dx
            eps_w = (w_now - w_total0) / w_field0
            loss_energy = jnp.mean(smooth_l1(eps_w))

            loss = loss_char + cfg.lambda_energy * loss_energy + cfg.lambda_ampere * loss_ampere
            metrics = {"loss": loss, "loss_char": loss_char, "loss_energy": loss_energy,
                      "loss_ampere": loss_ampere, "eps_w": jnp.max(jnp.abs(eps_w))}
            return loss, metrics

        @eqx.filter_jit
        def train_step(model, opt_state, key):
            key, subkey = jr.split(key)
            (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, subkey)
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, key, metrics

        return train_step

    def train(self, iters: Optional[int] = None) -> None:
        iters = iters or self.cfg.iters
        raw_history = []
        t0 = time.time()
        for it in range(1, iters + 1):
            self.model, self.opt_state, self.key, rec = self.train_step(
                self.model, self.opt_state, self.key)
            raw_history.append(rec)
            if it % self.cfg.log_every == 0 or it == 1:
                rate = it / (time.time() - t0)
                print(f"[jax] iter {it:6d}  loss={float(rec['loss']):.3e}  "
                      f"char={float(rec['loss_char']):.2e}  "
                      f"energy={float(rec['loss_energy']):.2e}  "
                      f"ampere={float(rec['loss_ampere']):.2e}  ({rate:.1f} it/s)", flush=True)
        jax.block_until_ready(raw_history[-1])
        self.train_time = time.time() - t0
        self.history = [{**{k: float(v) for k, v in r.items()}, "iter": i + 1}
                        for i, r in enumerate(raw_history)]

    def f_fn(self):
        return make_f_fn(self.model, self.problem.f0)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, self.model)
        meta = {"config": asdict(self.cfg), "history": self.history,
               "train_time": self.train_time}
        Path(str(path) + ".json").write_text(json.dumps(meta))


def load_model(path, device=None):
    path = Path(path)
    meta = json.loads(Path(str(path) + ".json").read_text())
    cfg = TrainConfig(**meta["config"])
    trainer = Trainer(cfg, device=device)
    trainer.model = eqx.tree_deserialise_leaves(path, trainer.model)
    trainer.history = meta["history"]
    trainer.train_time = meta["train_time"]
    return trainer


def n_params(model) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(int(np.prod(l.shape)) for l in leaves)



### classical reference solver: 4D Strang-splitting spectral semi-Lagrangian.
#
# jax.lax.scan over static-length segments between save points (segment
# boundaries are known ahead of time from n_steps/save_times, all Python
# ints). No manual memory "slabbing" -- XLA fuses the FFT/elementwise chain
# and plans its own buffer reuse; the un-chunked array at large N is exactly
# what should hit the real GPU memory wall (see GPU->CPU fallback below).

@dataclass
class ReferenceSolution:
    times: jnp.ndarray
    f: jnp.ndarray
    x_grid: jnp.ndarray
    v_grid: jnp.ndarray
    e_grid: jnp.ndarray


def _drift(f, kx, v, step):
    nx, nv = kx.shape[0], v.shape[0]
    ph_x = jnp.exp(-1j * kx.reshape(nx, 1, 1, 1) * v.reshape(1, 1, nv, 1) * step)
    f = jnp.fft.ifft(jnp.fft.fft(f, axis=0) * ph_x, axis=0).real
    ph_y = jnp.exp(-1j * kx.reshape(1, nx, 1, 1) * v.reshape(1, 1, 1, nv) * step)
    f = jnp.fft.ifft(jnp.fft.fft(f, axis=1) * ph_y, axis=1).real
    return f


def _kick(f, e, kv, step):
    nx, nv = e.shape[-1], kv.shape[0]
    ph_vx = jnp.exp(-1j * e[0].reshape(nx, nx, 1, 1) * kv.reshape(1, 1, nv, 1) * step)
    f = jnp.fft.ifft(jnp.fft.fft(f, axis=2) * ph_vx, axis=2).real
    ph_vy = jnp.exp(-1j * e[1].reshape(nx, nx, 1, 1) * kv.reshape(1, 1, 1, nv) * step)
    f = jnp.fft.ifft(jnp.fft.fft(f, axis=3) * ph_vy, axis=3).real
    return f


def _field(f, dv2, length, has_field):
    if not has_field:
        nx = f.shape[0]
        return jnp.zeros((2, nx, nx), dtype=f.dtype)
    rho = jnp.sum(f, axis=(2, 3)) * dv2
    return electric_field_2d(rho, (length, length))


@functools.partial(jax.jit, static_argnames=("length", "dv2", "has_field", "n_steps"))
def _march_segment(f, kx, kv, v, length, dv2, has_field, dt, n_steps):
    def step_fn(f, _):
        f = _drift(f, kx, v, 0.5 * dt)
        e = _field(f, dv2, length, has_field)
        f = _kick(f, e, kv, dt)
        f = _drift(f, kx, v, 0.5 * dt)
        return f, None
    f, _ = jax.lax.scan(step_fn, f, xs=None, length=n_steps)
    return f


def _solve_reference_impl(problem: Problem, nx: int, nv: int, dt: float,
                          save_times, dtype, dev):
    with jax.default_device(dev):
        length = problem.length
        x = uniform_grid(problem.x_lo, problem.x_hi, nx, dtype=dtype)
        v = uniform_grid(problem.v_lo, problem.v_hi, nv, dtype=dtype)
        v_len = problem.v_hi - problem.v_lo
        kx = (2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=length / nx)).astype(dtype)
        kv = (2.0 * jnp.pi * jnp.fft.fftfreq(nv, d=v_len / nv)).astype(dtype)
        dv2 = (v_len / nv) ** 2

        xx, yy, vvx, vvy = jnp.meshgrid(x, x, v, v, indexing="ij")
        z = jnp.stack([xx, yy, vvx, vvy], axis=-1).reshape(-1, 4)
        f = problem.f0(z).reshape(nx, nx, nv, nv).astype(dtype)
        del xx, yy, vvx, vvy, z

        save_times = sorted(set([0.0] + [float(s) for s in save_times]))
        n_steps_total = max(1, round(problem.t_final / dt))
        dt = problem.t_final / n_steps_total
        save_steps = sorted({round(s / dt) for s in save_times})

        out_f, saved_e, saved_t = [], [], []
        if save_steps and save_steps[0] == 0:
            out_f.append(f)
            saved_e.append(_field(f, dv2, length, problem.has_field))
            saved_t.append(0.0)
            save_steps = save_steps[1:]

        step = 0
        for target in save_steps:
            seg = target - step
            f = _march_segment(f, kx, kv, v, length, dv2, problem.has_field, dt, seg)
            step = target
            out_f.append(f)
            saved_e.append(_field(f, dv2, length, problem.has_field))
            saved_t.append(step * dt)

        return ReferenceSolution(times=jnp.array(saved_t, dtype=dtype),
                                 f=jnp.stack(out_f), x_grid=x, v_grid=v,
                                 e_grid=jnp.stack(saved_e))


def solve_reference(problem: Problem, nx=48, nv=48, dt=1.0 / 64.0, save_times=None,
                    device: str = "auto", dtype=F64) -> ReferenceSolution:
    save_times = save_times if save_times is not None else (problem.plot_times or [problem.t_final])
    dev = pick_device(None if device == "auto" else device)
    try:
        return _solve_reference_impl(problem, nx, nv, dt, save_times, dtype, dev)
    except Exception as exc:
        if not is_oom(exc) or dev.platform == "cpu":
            raise
        jax.clear_caches()
        return _solve_reference_impl(problem, nx, nv, dt, save_times, dtype, cpu_device())


def solver_footprint_bytes(nx: int, nv: int, n_saves: int = 1, dtype_bytes: int = 8) -> dict:
    cells = nx * nx * nv * nv
    return {"cells": cells, "state": cells * dtype_bytes,
           "saved": n_saves * cells * dtype_bytes,
           "total": (1 + n_saves) * cells * dtype_bytes}


# ---------------------------------------------------------------------------
# resolution-scaling benchmark: wall-clock, peak memory, and GPU->CPU
# fallback vs N, run one resolution per subprocess (JAX exposes no reset for
# its peak-memory counters in this version, so an in-process loop over N
# would produce meaningless per-row peaks).
# ---------------------------------------------------------------------------

def bench_one_resolution(problem_name: str, n: int, n_steps: int, device: str) -> dict:
    problem = make_problem(problem_name)
    dt = problem.t_final / max(1, n_steps)
    dev = pick_device(None if device == "auto" else device)
    t0 = time.perf_counter()
    fell_back = False
    used = dev.platform
    try:
        ref = _solve_reference_impl(problem, n, n, dt, [problem.t_final], F64, dev)
        jax.block_until_ready(ref.f)
    except Exception as exc:
        if not is_oom(exc) or dev.platform == "cpu":
            raise
        jax.clear_caches()
        fell_back = True
        used = "cpu"
        t0 = time.perf_counter()
        ref = _solve_reference_impl(problem, n, n, dt, [problem.t_final], F64, cpu_device())
        jax.block_until_ready(ref.f)
    wall = time.perf_counter() - t0
    mem = device_memory_stats(dev if not fell_back else cpu_device())
    return {"method": "classical_sl", "n": n, "device_used": used, "fell_back": fell_back,
           "wall_time_solve_s": wall, "wall_time_train_s": 0.0, "wall_time_infer_s": 0.0,
           "wall_time_total_s": wall,
           "peak_gpu_bytes": mem.get("peak_bytes_in_use", 0) if not fell_back else 0,
           "peak_host_rss_bytes": peak_rss_bytes(),
           "n_steps": n_steps, "problem": problem_name, "dtype": "float64"}


def run_resolution_sweep(problem_name: str, grid_sizes, n_steps: int = 8,
                         device: str = "auto") -> list:
    """Each N in its own subprocess -- see module docstring / bench_one_resolution."""
    rows = []
    for n in grid_sizes:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--bench-one-n", str(n),
             "--n-steps", str(n_steps), "--device", device, "--problem", problem_name],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[sweep] n={n} failed:\n{proc.stderr[-4000:]}", flush=True)
            rows.append({"method": "classical_sl", "n": n, "device_used": "error",
                        "fell_back": None, "wall_time_total_s": None,
                        "problem": problem_name, "notes": "subprocess failed"})
            continue
        row = json.loads(proc.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(f"[sweep] n={n:4d}  {row['wall_time_solve_s']:8.3f}s  "
              f"device={row['device_used']}  fell_back={row['fell_back']}", flush=True)
    return rows


### diagnostics

@dataclass
class Conservation:
    times: jnp.ndarray
    mass: jnp.ndarray
    momentum: jnp.ndarray
    energy: jnp.ndarray
    e_rms: jnp.ndarray


def conservation_from_f_grid_2d(f, times, x_grids, v_grids, lengths, has_field=True) -> Conservation:
    dx = (lengths[0] / f.shape[1]) * (lengths[1] / f.shape[2])
    wu = trapezoid_weights(v_grids[0])
    ww = trapezoid_weights(v_grids[1])
    rho = jnp.einsum("txyuw,u,w->txy", f, wu, ww)
    if has_field:
        e = electric_field_2d(rho, lengths)
    else:
        e = jnp.zeros((rho.shape[0], 2) + rho.shape[1:], dtype=rho.dtype)
    mass = jnp.sum(rho, axis=(1, 2)) * dx
    px = jnp.einsum("txyuw,u,u,w->t", f, v_grids[0], wu, ww) * dx
    py = jnp.einsum("txyuw,u,w,w->t", f, wu, v_grids[1], ww) * dx
    momentum = jnp.sqrt(px ** 2 + py ** 2)
    vsq = v_grids[0].reshape(-1, 1) ** 2 + v_grids[1].reshape(1, -1) ** 2
    kinetic = 0.5 * jnp.einsum("txyuw,uw,u,w->t", f, vsq, wu, ww) * dx
    field_e = 0.5 * jnp.sum(e ** 2, axis=(1, 2, 3)) * dx
    e_rms = jnp.sqrt(jnp.mean(jnp.sum(e ** 2, axis=1), axis=(1, 2)))
    return Conservation(times, mass, momentum, kinetic + field_e, e_rms)


def evaluate_f_on_grids(f_fn, times, grids):
    return eval_on_mesh(f_fn, times, grids)


def error_table(f_model, f_ref, times) -> list:
    rows = []
    for i, t in enumerate(np.asarray(times).tolist()):
        diff = f_model[i] - f_ref[i]
        rel = float(jnp.sqrt(jnp.sum(diff ** 2))
                    / jnp.clip(jnp.sqrt(jnp.sum(f_ref[i] ** 2)), 1e-30, None))
        rows.append({"t": t, "l2": float(jnp.sqrt(jnp.mean(diff ** 2))),
                    "linf": float(jnp.max(jnp.abs(diff))), "rel_l2": rel})
    return rows


def rel_l2(a, b) -> float:
    a64, b64 = a.astype(F64), b.astype(F64)
    return float(jnp.sqrt(jnp.sum((a64 - b64) ** 2))
                / jnp.clip(jnp.sqrt(jnp.sum(b64 ** 2)), 1e-30, None))


def det_statistics(model, problem: Problem, times, n_points=512, seed=0) -> list:
    key = jr.PRNGKey(seed)
    lo = jnp.array(problem.domain_lo(), dtype=F64)
    hi = jnp.array(problem.domain_hi(), dtype=F64)
    z = lo + (hi - lo) * jr.uniform(key, (n_points, problem.dim), dtype=F64)
    model64 = jax.tree_util.tree_map(lambda a: a.astype(F64) if eqx.is_array(a) else a, model)
    rows = []
    for t in np.asarray(times).tolist():
        t_arr = jnp.full((n_points,), t, dtype=F64)
        jac = flow_jacobian(model64, t_arr, z)
        det_err = jnp.abs(jnp.linalg.det(jac) - 1.0)
        rows.append({"t": t, "median": float(jnp.median(det_err)), "max": float(jnp.max(det_err))})
    return rows


def characteristics_map(model, z0, times):
    out = []
    for t in np.asarray(times).tolist():
        tt = jnp.full((z0.shape[0],), t, dtype=z0.dtype)
        out.append(flow_inverse(model, tt, z0))
    return jnp.stack(out)


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


def use_paper_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "image.cmap": "viridis",
        "figure.max_open_warning": 0,
    })


# meshless Poisson ablation (stretch, not wired into the headline Trainer):
# truncated Fourier-mode Monte Carlo estimate of rho_hat(k), reconstructing E
# by explicit truncated Fourier synthesis -- no spatial grid at all, not even
# the small fixed one above. See the notebook for the validation-vs-grid cell.

def fourier_modes(k_max: int, length: float, dtype=F32):
    ks = jnp.arange(-k_max, k_max + 1, dtype=dtype) * (2.0 * jnp.pi / length)
    kx, ky = jnp.meshgrid(ks, ks, indexing="ij")
    return kx.reshape(-1), ky.reshape(-1)


def rho_hat_monte_carlo(f_fn, t, z, w, kx, ky):
    """rho_hat(k) ~= mean_m[ f(z_m) exp(-i k.x_m) w_m ] over an importance-
    sampled 4D phase-space point cloud (t,z,w already drawn by the caller)."""
    f = f_fn(t, z)
    phase = jnp.exp(-1j * (kx[None, :] * z[:, 0:1] + ky[None, :] * z[:, 1:2]))
    return jnp.sum((f * w)[:, None] * phase, axis=0)


def e_from_fourier_modes(rho_hat, kx, ky, xq):
    k2 = kx ** 2 + ky ** 2
    k2_safe = jnp.where(k2 == 0, jnp.ones_like(k2), k2)
    coef = jnp.where(k2 == 0, jnp.zeros_like(rho_hat), rho_hat / k2_safe)
    phase = jnp.exp(1j * (kx[None, :] * xq[:, 0:1] + ky[None, :] * xq[:, 1:2]))
    n_modes = kx.shape[0]
    ex = jnp.sum((-1j * kx[None, :] * coef[None, :] * phase).real, axis=1) / n_modes
    ey = jnp.sum((-1j * ky[None, :] * coef[None, :] * phase).real, axis=1) / n_modes
    return jnp.stack([ex, ey], axis=-1)


# CLI: `python slpinn_2d2v_jax.py --bench-one-n N ...` runs one resolution of 
# the classical solver in its own process and prints one JSON row to stdout
# (used by run_resolution_sweep's subprocess-per-N driver).

def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-one-n", type=int, required=True)
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--problem", default="gaussian_vp_2d")
    args = ap.parse_args()
    row = bench_one_resolution(args.problem, args.bench_one_n, args.n_steps, args.device)
    print(json.dumps(row))


if __name__ == "__main__":
    _main()
