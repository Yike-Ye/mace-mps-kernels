"""
Fused MACE convolution for Apple GPUs: gather → channel-wise (uvu) tensor product → scatter-sum in one Metal kernel.

The kernel is generated per e3nn ``TensorProduct`` (one per interaction layer).  The effective coefficient of every
path (Clebsch–Gordan × normalisation × path weight, exactly as e3nn applies them) is measured by probing the
TensorProduct with unit inputs, and written into the Metal source as a constant, so the generated code reproduces
e3nn's conventions without re-deriving them.

Layouts (e3nn "mul_ir", per node / edge):
  x   [N, X_DIM]        input block b (l_b, mul) at x_off[b] + u*(2l_b+1) + m
  Y   [E, Y_DIM]        edge attributes (spherical harmonics), block b at y_off[b] + m
  W   [E, NINS*MUL]     per-edge weights, instruction k / channel u at k*MUL + u
  out [N, OUT_DIM]      instruction k output block at o_off[k] + u*(2l_k+1) + m
  message[n] = Σ_{e: receiver(e)=n} TP(x[sender(e)], Y[e], W[e])

Forward: one thread per (receiver, channel), edges visited in receiver-sorted (CSR) order → deterministic, no atomics.
Backward: d/dx one thread per (sender, channel) over sender-sorted edges; d/dW and d/dY one threadgroup of MUL threads
per edge with a simdgroup reduction for d/dY.  First-order only: double backward raises.
"""
import torch


def _blocks(irreps):
    res, off = [], 0
    for mul, ir in irreps:
        res.append((ir.l, mul, off))
        off += mul * (2 * ir.l + 1)
    return res, off


def conv_supported(tp):
    """(ok, reason) — whether an e3nn TensorProduct can be replaced by the fused Metal convolution."""
    try:
        ins = tp.instructions
        if any(i.connection_mode != 'uvu' or not i.has_weight for i in ins):
            return False, 'only weighted uvu instructions are supported'
        if getattr(tp, 'shared_weights', True) or getattr(tp, 'internal_weights', True):
            return False, 'per-edge (external, unshared) weights required'
        muls = {m for m, _ in tp.irreps_in1} | {m for m, _ in tp.irreps_out}
        if len(muls) != 1:
            return False, 'all node/out blocks must have the same multiplicity'
        mul = muls.pop()
        if mul % 32:
            return False, f'multiplicity {mul} is not a multiple of 32'
        if any(m != 1 for m, _ in tp.irreps_in2):
            return False, 'edge attributes must have multiplicity 1 per l'
        if tp.weight_numel != len(ins) * mul:
            return False, 'unexpected weight layout'
    except AttributeError as e:
        return False, f'not an e3nn TensorProduct ({e})'
    return True, ''


class ConvSpec:
    """Instruction table + effective coefficients of one uvu TensorProduct."""

    def __init__(self, tp, tol=1e-7):
        ok, why = conv_supported(tp)
        if not ok:
            raise ValueError(why)
        self.in1, self.x_dim = _blocks(tp.irreps_in1)
        self.in2, self.y_dim = _blocks(tp.irreps_in2)
        self.outb, self.out_dim = _blocks(tp.irreps_out)
        self.mul = self.in1[0][1]
        self.ins = [(i.i_in1, i.i_in2, i.i_out) for i in tp.instructions]
        self.nins = len(self.ins)
        dev = next(iter(tp.buffers()), torch.zeros(1)).device
        self.coef = []
        for k, (i1, i2, io) in enumerate(self.ins):
            l1, _, o1 = self.in1[i1]
            l2, _, o2 = self.in2[i2]
            l3, _, o3 = self.outb[io]
            d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
            x1 = torch.zeros(d1 * d2, self.x_dim, device=dev)
            x2 = torch.zeros(d1 * d2, self.y_dim, device=dev)
            w = torch.zeros(d1 * d2, tp.weight_numel, device=dev)
            for a in range(d1):
                for b in range(d2):
                    r = a * d2 + b
                    x1[r, o1 + a] = 1.0                   # channel u = 0
                    x2[r, o2 + b] = 1.0
                    w[r, k * self.mul] = 1.0
            with torch.no_grad():
                y = tp(x1, x2, w)[:, o3:o3 + d3].cpu().double().reshape(d1, d2, d3)   # MPS has no float64
            terms = [(a, b, c, float(y[a, b, c])) for a in range(d1) for b in range(d2) for c in range(d3)
                     if abs(float(y[a, b, c])) > tol]
            self.coef.append(dict(k=k, l1=l1, l2=l2, l3=l3, o1=o1, o2=o2, o3=o3, terms=terms))
        self.nnz = sum(len(c['terms']) for c in self.coef)

    def xreg(self):
        r, idx = 0, {}
        for b, (l, _, _) in enumerate(self.in1):
            idx[b] = r
            r += 2 * l + 1
        return idx, r

    def oreg(self):
        r, idx = 0, {}
        for c in self.coef:
            idx[c['k']] = r
            r += 2 * c['l3'] + 1
        return idx, r


def _f(v):
    return f'{v:.9e}f'


def _fwd(spec):
    xi, nx = spec.xreg()
    oi, no = spec.oreg()
    L = ['kernel void conv_fwd(device const float* x [[buffer(0)]], device const float* Y [[buffer(1)]],',
         '    device const float* W [[buffer(2)]], device const int* rowptr [[buffer(3)]],',
         '    device const int* eids [[buffer(4)]], device const int* senders [[buffer(5)]],',
         '    device float* out [[buffer(6)]], constant uint& N [[buffer(7)]],',
         '    uint gid [[thread_position_in_grid]]) {',
         f'  const uint MUL = {spec.mul}; uint n = gid / MUL; uint u = gid % MUL; if (n >= N) return;',
         f'  float acc[{no}];',
         f'  for (int i = 0; i < {no}; ++i) acc[i] = 0.0f;',
         '  for (int p = rowptr[n]; p < rowptr[n + 1]; ++p) {',
         '    int e = eids[p]; int s = senders[e];',
         f'    float xs[{nx}]; float y[{spec.y_dim}];']
    for b, (l, _, off) in enumerate(spec.in1):
        d = 2 * l + 1
        L += [f'    xs[{xi[b] + m}] = x[s * {spec.x_dim} + {off} + u * {d} + {m}];' for m in range(d)]
    L.append(f'    for (int j = 0; j < {spec.y_dim}; ++j) y[j] = Y[e * {spec.y_dim} + j];')
    L.append(f'    const device float* we = W + e * {spec.nins * spec.mul} + u;')
    for c in spec.coef:
        k, i1, d3 = c['k'], spec.ins[c['k']][0], 2 * c['l3'] + 1
        L.append(f'    {{ float w = we[{k * spec.mul}];')
        for m3 in range(d3):
            ts = [f'{_f(v)} * xs[{xi[i1] + a}] * y[{c["o2"] + b}]' for a, b, m, v in c['terms'] if m == m3]
            if ts:
                L.append(f'      acc[{oi[k] + m3}] += w * (' + ' + '.join(ts) + ');')
        L.append('    }')
    L.append('  }')
    for c in spec.coef:
        d3 = 2 * c['l3'] + 1
        L += [f'  out[n * {spec.out_dim} + {c["o3"]} + u * {d3} + {m3}] = acc[{oi[c["k"]] + m3}];' for m3 in range(d3)]
    L.append('}')
    return L


def _bwd_x(spec):
    xi, nx = spec.xreg()
    oi, no = spec.oreg()
    L = ['kernel void conv_bwd_x(device const float* g [[buffer(0)]], device const float* Y [[buffer(1)]],',
         '    device const float* W [[buffer(2)]], device const int* rowptr [[buffer(3)]],',
         '    device const int* eids [[buffer(4)]], device const int* receivers [[buffer(5)]],',
         '    device float* gx [[buffer(6)]], constant uint& N [[buffer(7)]],',
         '    uint gid [[thread_position_in_grid]]) {',
         f'  const uint MUL = {spec.mul}; uint s = gid / MUL; uint u = gid % MUL; if (s >= N) return;',
         f'  float acc[{nx}];',
         f'  for (int i = 0; i < {nx}; ++i) acc[i] = 0.0f;',
         '  for (int p = rowptr[s]; p < rowptr[s + 1]; ++p) {',
         '    int e = eids[p]; int r = receivers[e];',
         f'    float go[{no}]; float y[{spec.y_dim}];']
    for c in spec.coef:
        d3 = 2 * c['l3'] + 1
        L += [f'    go[{oi[c["k"]] + m3}] = g[r * {spec.out_dim} + {c["o3"]} + u * {d3} + {m3}];' for m3 in range(d3)]
    L.append(f'    for (int j = 0; j < {spec.y_dim}; ++j) y[j] = Y[e * {spec.y_dim} + j];')
    L.append(f'    const device float* we = W + e * {spec.nins * spec.mul} + u;')
    for c in spec.coef:
        k, i1, d1 = c['k'], spec.ins[c['k']][0], 2 * c['l1'] + 1
        L.append(f'    {{ float w = we[{k * spec.mul}];')
        for a in range(d1):
            ts = [f'{_f(v)} * y[{c["o2"] + b}] * go[{oi[k] + m}]' for aa, b, m, v in c['terms'] if aa == a]
            if ts:
                L.append(f'      acc[{xi[i1] + a}] += w * (' + ' + '.join(ts) + ');')
        L.append('    }')
    L.append('  }')
    for b, (l, _, off) in enumerate(spec.in1):
        d = 2 * l + 1
        L += [f'  gx[s * {spec.x_dim} + {off} + u * {d} + {m}] = acc[{xi[b] + m}];' for m in range(d)]
    L.append('}')
    return L


def _bwd_wy(spec):
    xi, nx = spec.xreg()
    oi, no = spec.oreg()
    nsg = spec.mul // 32
    L = ['kernel void conv_bwd_wy(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]],',
         '    device const float* Y [[buffer(2)]], device const float* W [[buffer(3)]],',
         '    device const int* senders [[buffer(4)]], device const int* receivers [[buffer(5)]],',
         '    device float* gW [[buffer(6)]], device float* gY [[buffer(7)]], constant uint& E [[buffer(8)]],',
         '    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],',
         '    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {',
         f'  const uint MUL = {spec.mul}; uint e = gid / MUL; uint u = tid;',
         f'  threadgroup float part[{nsg}][{spec.y_dim}];',
         f'  float gy[{spec.y_dim}];',
         f'  for (int j = 0; j < {spec.y_dim}; ++j) gy[j] = 0.0f;',
         '  if (e < E) {',
         '    int s = senders[e]; int r = receivers[e];',
         f'    float xs[{nx}]; float go[{no}]; float y[{spec.y_dim}];']
    for b, (l, _, off) in enumerate(spec.in1):
        d = 2 * l + 1
        L += [f'    xs[{xi[b] + m}] = x[s * {spec.x_dim} + {off} + u * {d} + {m}];' for m in range(d)]
    for c in spec.coef:
        d3 = 2 * c['l3'] + 1
        L += [f'    go[{oi[c["k"]] + m3}] = g[r * {spec.out_dim} + {c["o3"]} + u * {d3} + {m3}];' for m3 in range(d3)]
    L.append(f'    for (int j = 0; j < {spec.y_dim}; ++j) y[j] = Y[e * {spec.y_dim} + j];')
    L.append(f'    const device float* we = W + e * {spec.nins * spec.mul} + u;')
    L.append(f'    device float* gwe = gW + e * {spec.nins * spec.mul} + u;')
    for c in spec.coef:
        k, i1 = c['k'], spec.ins[c['k']][0]
        L.append(f'    {{ float w = we[{k * spec.mul}];')
        ts = [f'{_f(v)} * xs[{xi[i1] + a}] * y[{c["o2"] + b}] * go[{oi[k] + m}]' for a, b, m, v in c['terms']]
        L.append(f'      gwe[{k * spec.mul}] = ' + (' + '.join(ts) if ts else '0.0f') + ';')
        for b in range(2 * c['l2'] + 1):
            tb = [f'{_f(v)} * xs[{xi[i1] + a}] * go[{oi[k] + m}]' for a, bb, m, v in c['terms'] if bb == b]
            if tb:
                L.append(f'      gy[{c["o2"] + b}] += w * (' + ' + '.join(tb) + ');')
        L.append('    }')
    L.append('  }')
    L.append(f'  for (int j = 0; j < {spec.y_dim}; ++j) {{ float v = simd_sum(gy[j]); if (lane == 0) part[sg][j] = v; }}')
    L.append('  threadgroup_barrier(mem_flags::mem_threadgroup);')
    L.append(f'  if (e < E && tid < {spec.y_dim}) {{ float v = 0.0f;')
    L.append(f'    for (int q = 0; q < {nsg}; ++q) v += part[q][tid];')
    L.append(f'    gY[e * {spec.y_dim} + tid] = v; }}')
    L.append('}')
    return L


def source(spec):
    return '\n'.join(['#include <metal_stdlib>', 'using namespace metal;'] + _fwd(spec) + _bwd_x(spec) + _bwd_wy(spec))


def csr_by(index, n):
    """edges sorted by `index` → (rowptr[n+1], edge ids) as int32 tensors."""
    order = torch.argsort(index, stable=True)
    counts = torch.bincount(index, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=torch.int64, device=index.device)
    rowptr[1:] = torch.cumsum(counts, 0)
    return rowptr.to(torch.int32), order.to(torch.int32)


class _ConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Y, W, edge_index, op):
        x, Y, W = x.contiguous(), Y.contiguous(), W.contiguous()
        n = x.shape[0]
        recv = edge_index[1].to(torch.int32).contiguous()
        send = edge_index[0].to(torch.int32).contiguous()
        rowptr, eids = csr_by(edge_index[1], n)
        out = torch.empty(n, op.spec.out_dim, device=x.device, dtype=torch.float32)
        op.lib.conv_fwd(x, Y, W, rowptr, eids, send, out, n, threads=n * op.spec.mul)
        ctx.save_for_backward(x, Y, W, send, recv)
        ctx.op, ctx.n = op, n
        return out

    @staticmethod
    def backward(ctx, g):
        if torch.is_grad_enabled():      # only true under create_graph=True
            raise RuntimeError('mace-mps-kernels: double backward (create_graph=True, e.g. training on forces/stress or an autograd Hessian) '
            'is not supported by the Metal kernels; use the unpatched e3nn model for that')
        x, Y, W, send, recv = ctx.saved_tensors
        op, n = ctx.op, ctx.n
        g = g.contiguous()
        gx = gY = gW = None
        if ctx.needs_input_grad[0]:
            rowptr_s, eids_s = csr_by(send.long(), n)
            gx = torch.empty_like(x)
            op.lib.conv_bwd_x(g, Y, W, rowptr_s, eids_s, recv, gx, n, threads=n * op.spec.mul)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            gW = torch.empty_like(W)
            gY = torch.empty_like(Y)
            E = Y.shape[0]
            op.lib.conv_bwd_wy(x, g, Y, W, send, recv, gW, gY, E, threads=E * op.spec.mul, group_size=op.spec.mul)
        return gx, gY, gW, None, None


class MetalConvTP(torch.nn.Module):
    """Replacement for an interaction block's ``conv_tp`` under MACE's ``conv_fusion`` path:
    forward(node_feats, edge_attrs, tp_weights, edge_index) → summed messages [N, OUT_DIM]."""

    def __init__(self, tp):
        super().__init__()
        self.spec = ConvSpec(tp)
        self.lib = torch.mps.compile_shader(source(self.spec))
        self.irreps_in1, self.irreps_in2, self.irreps_out = tp.irreps_in1, tp.irreps_in2, tp.irreps_out
        self.weight_numel = tp.weight_numel

    def forward(self, node_feats, edge_attrs, tp_weights, edge_index):
        return _ConvFn.apply(node_feats, edge_attrs, tp_weights, edge_index, self)
