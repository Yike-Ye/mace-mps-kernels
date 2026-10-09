"""
MACE SymmetricContraction as a precomputed polynomial, evaluated in one Metal kernel.

At inference the product-basis weights and the U matrices are constants, so for every element e, channel c and output
component M the contraction is a polynomial in the node features x = x[b, c, :] (n_ell = (l_max+1)^2 components):
    out[b, c, M] = Σ_{a≤b'≤c'} T3 x_a x_b' x_c' + Σ_{a≤b'} T2 x_a x_b' + Σ_a T1 x_a          (correlation 3)
The coefficients follow MACE's nested contraction  A = Σ_i U_ν[..., i, p] W_ν[p] x_i ; B = Σ (A + U_{ν-1}·W_{ν-1}) x ;
…  and are computed once in float64 on the CPU.  Monomials are symmetrised and those that vanish for every (e, c) are
dropped (parity and the coupling structure remove most of them).  Works for any correlation order, any l_max and the
full or reduced CG basis, because only U and the weights are used.

Coefficient layout T[element][monomial × M][channel] → coalesced reads across the channel threads of one node.
One thread per (node, channel).  Backward: ∂/∂x of the polynomial, first order only (double backward raises).
"""
import itertools
import numpy as np
import torch


def _poly_coeffs(ct):
    corr = ct.correlation
    U = {nu: getattr(ct, f'U_matrix_{nu}').detach().cpu().double() for nu in range(1, corr + 1)}
    W = {corr: ct.weights_max.detach().cpu().double()}
    for j, w in enumerate(ct.weights):                     # weights[0] ↔ ν = corr-1, weights[1] ↔ corr-2, ...
        W[corr - 1 - j] = w.detach().cpu().double()
    lead = U[1].dim() - 2                                   # 1 if there is an M axis (L > 0), else 0
    M = U[1].shape[0] if lead else 1
    n_ell = U[1].shape[-2]
    res = {}
    for nu in range(1, corr + 1):
        u = U[nu] if lead else U[nu].unsqueeze(0)           # [M, ell × ν, p]
        T = torch.einsum(u.reshape(M, -1, u.shape[-1]), [0, 1, 2], W[nu], [3, 2, 4], [3, 4, 0, 1])
        T = T.reshape(W[nu].shape[0], W[nu].shape[2], M, *([n_ell] * nu)).numpy()
        acc = {}
        for idx in itertools.product(range(n_ell), repeat=nu):
            key = tuple(sorted(idx))
            acc[key] = acc.get(key, 0) + T[(slice(None),) * 3 + idx]
        res[nu] = {k: v.transpose(0, 2, 1) for k, v in acc.items()}       # [e, M, c]
    return res, M, n_ell


def sc_supported(sc):
    try:
        cts = list(sc.contractions)
        for ct in cts:
            _ = ct.correlation, ct.weights_max, ct.U_matrix_1
    except AttributeError as e:
        return False, f'not a MACE SymmetricContraction ({e})'
    return True, ''


class SCSpec:
    def __init__(self, sc, tol=1e-12):
        ok, why = sc_supported(sc)
        if not ok:
            raise ValueError(why)
        self.blocks, rows, out_off = [], [], 0
        self.n_ell = None
        for ct in sc.contractions:
            coeffs, M, n_ell = _poly_coeffs(ct)
            assert self.n_ell in (None, n_ell)
            self.n_ell = n_ell
            terms = []
            for nu in sorted(coeffs, reverse=True):
                for idx, v in sorted(coeffs[nu].items()):
                    if np.abs(v).max() <= tol:
                        continue
                    terms.append((idx, len(rows)))
                    rows += [v[:, m, :] for m in range(M)]
            self.C = next(iter(coeffs[1].values())).shape[2]
            self.blocks.append(dict(M=M, terms=terms, out_off=out_off))
            out_off += self.C * M
        self.out_dim, self.nrows = out_off, len(rows)
        self.T = torch.tensor(np.stack(rows, axis=1), dtype=torch.float32)   # [e, rows, c]
        self.n_terms = sum(len(b['terms']) for b in self.blocks)


def _mono(idx):
    return ' * '.join(f'x{i}' for i in idx)


def source(spec):
    C, R, n = spec.C, spec.nrows, spec.n_ell
    f = ['#include <metal_stdlib>', 'using namespace metal;',
         'kernel void sc_fwd(device const float* X [[buffer(0)]], device const int* elem [[buffer(1)]],',
         '    device const float* T [[buffer(2)]], device float* out [[buffer(3)]], constant uint& N [[buffer(4)]],',
         '    uint gid [[thread_position_in_grid]]) {',
         f'  uint b = gid / {C}; uint c = gid % {C}; if (b >= N) return;',
         f'  const device float* xp = X + (b * {C} + c) * {n};']
    f += [f'  float x{i} = xp[{i}];' for i in range(n)]
    f.append(f'  const device float* t = T + elem[b] * {R * C} + c;')
    for bl in spec.blocks:
        M = bl['M']
        f.append('  {')
        f += [f'    float a{m} = 0.0f;' for m in range(M)]
        for idx, r0 in bl['terms']:
            if M == 1:
                f.append(f'    a0 += t[{r0 * C}] * {_mono(idx)};')
            else:
                f.append(f'    {{ float p = {_mono(idx)};')
                f += [f'      a{m} += t[{(r0 + m) * C}] * p;' for m in range(M)]
                f.append('    }')
        f += [f'    out[b * {spec.out_dim} + {bl["out_off"]} + c * {M} + {m}] = a{m};' for m in range(M)]
        f.append('  }')
    f.append('}')
    g = ['kernel void sc_bwd(device const float* X [[buffer(0)]], device const int* elem [[buffer(1)]],',
         '    device const float* T [[buffer(2)]], device const float* G [[buffer(3)]], device float* GX [[buffer(4)]],',
         '    constant uint& N [[buffer(5)]], uint gid [[thread_position_in_grid]]) {',
         f'  uint b = gid / {C}; uint c = gid % {C}; if (b >= N) return;',
         f'  const device float* xp = X + (b * {C} + c) * {n};']
    g += [f'  float x{i} = xp[{i}]; float d{i} = 0.0f;' for i in range(n)]
    g.append(f'  const device float* t = T + elem[b] * {R * C} + c;')
    for bl in spec.blocks:
        M = bl['M']
        g.append('  {')
        g += [f'    float g{m} = G[b * {spec.out_dim} + {bl["out_off"]} + c * {M} + {m}];' for m in range(M)]
        for idx, r0 in bl['terms']:
            s = ' + '.join(f't[{(r0 + m) * C}] * g{m}' for m in range(M))
            g.append(f'    {{ float s = {s};')
            for i in sorted(set(idx)):
                rest = list(idx)
                rest.remove(i)
                k = idx.count(i)
                fac = f'{float(k)}f * ' if k > 1 else ''
                g.append(f'      d{i} += s * {fac}{_mono(rest) if rest else "1.0f"};')
            g.append('    }')
        g.append('  }')
    g.append(f'  device float* gp = GX + (b * {C} + c) * {n};')
    g += [f'  gp[{i}] = d{i};' for i in range(n)]
    g.append('}')
    return '\n'.join(f + g)


class _SCFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, elem, op):
        x = x.contiguous()
        N = x.shape[0]
        out = torch.empty(N, op.spec.out_dim, device=x.device, dtype=torch.float32)
        op.lib.sc_fwd(x, elem, op.T, out, N, threads=N * op.spec.C)
        ctx.save_for_backward(x, elem)
        ctx.op = op
        return out

    @staticmethod
    def backward(ctx, g):
        if torch.is_grad_enabled():      # only true under create_graph=True
            raise RuntimeError('mace-mps-kernels: double backward (create_graph=True, e.g. training on forces/stress or an autograd Hessian) '
            'is not supported by the Metal kernels; use the unpatched e3nn model for that')
        x, elem = ctx.saved_tensors
        op = ctx.op
        gx = torch.empty_like(x)
        N = x.shape[0]
        op.lib.sc_bwd(x, elem, op.T, g.contiguous(), gx, N, threads=N * op.spec.C)
        return gx, None, None


class MetalSymmetricContraction(torch.nn.Module):
    """Drop-in for mace ``SymmetricContraction`` at inference: forward(x[N, C, n_ell], node_attrs[N, n_elem])."""

    def __init__(self, sc, device='mps'):
        super().__init__()
        self.spec = SCSpec(sc)
        self.register_buffer('T', self.spec.T.to(device).contiguous())
        self.lib = torch.mps.compile_shader(source(self.spec))
        self.irreps_in, self.irreps_out = sc.irreps_in, sc.irreps_out

    def forward(self, x, y):
        elem = torch.argmax(y, dim=1).to(torch.int32).contiguous()
        return _SCFn.apply(x, elem, self)
