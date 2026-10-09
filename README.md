# mace-mps-kernels

Fused Metal kernels for [MACE](https://github.com/ACEsuit/mace) **inference** on Apple-silicon GPUs (PyTorch `mps`).

On Apple GPUs MACE runs on the generic e3nn path, where one equivariant tensor product is split into hundreds of small
GPU dispatches and most of an energy + force call is spent in the backward pass. This package replaces the two
operations that dominate that time with generated, fused Metal kernels (forward and first-order backward):

| op | what the kernel does |
|---|---|
| interaction convolution (`conv_tp`, channel-wise *uvu* tensor product) | gather → tensor product → scatter-sum in one kernel; per-layer code with the e3nn path coefficients baked in as constants |
| symmetric contraction (product basis) | evaluated as a precomputed polynomial in the node features (weights and U matrices are constants at inference); symmetrised monomials, vanishing ones dropped |

Everything else (embeddings, radial MLP, linears, readouts) stays on the original PyTorch path. The model file is not
modified; the patch is applied in memory.

## Usage

```python
from mace.calculators import MACECalculator
from mace_mps_kernels import patch_model

calc = MACECalculator(model_paths="my.model", device="mps", default_dtype="float32")
report = patch_model(calc.models[0])      # {'patched': [...], 'skipped': [(name, reason), ...]}
atoms.calc = calc                          # energies, forces, stress as usual
```

`patch_model` replaces only what it supports and leaves the rest on e3nn (with a warning naming the reason).

## Performance

Apple M4 Pro (20-core GPU), macOS 27, torch 2.14, mace-torch 0.3.16, e3nn 0.4.4, float32.
Model: 2 interactions, 128 channels, hidden `128x0e+128x1o`, l_max = 3, correlation 3, 4 elements, r_max 5 Å.
Structures: Pt(100) slabs with adsorbates, 140–181 atoms (~5 000 edges).

| | e3nn | Metal kernels | speed-up |
|---|---:|---:|---:|
| conv layer 1, fwd + bwd | 22.7 ms | 1.2 ms | 19× |
| conv layer 2, fwd + bwd | 83.2 ms | 2.2 ms | 37× |
| symmetric contraction 1 / 2, fwd + bwd | 20.4 / 7.1 ms | 0.42 / 0.07 ms | 49× / 102× |
| **one energy + force call, 1 structure** | 161–194 ms | 13–17 ms | **11–12×** |
| energy + force, MACE batch of 4 / 16 structures (`benchmarks/bench.py`) | 121 / 125 ms per structure | 7.1 / 7.4 ms per structure | **17×** |

A small random model (32 channels, 79-atom slab; `python benchmarks/bench.py` without arguments) gives 6–9×.

MACE-MP-0 medium (89 elements; `benchmarks/foundation_demo.py`): Au(111)+CO/O, Cu(100)+H/OH, Ni(111)+N/NH and Au13 on
CeO2(111), 72–157 atoms: 6–9× per energy + force call; LBFGS relaxation of Au13/CeO2(111): 33 vs 269 ms per step.

Agreement with e3nn: max |ΔF| ≤ 4 × 10⁻⁶ eV/Å (relative ≤ 10⁻⁶), |ΔE| ≤ 1 meV per structure (fp32 summation order).
Relaxations of adsorbate layers follow e3nn step for step (same steps, same final energy). For systems with many
nearby minima (e.g. a reconstructing Au13 cluster on ceria) the trajectories agree for ~100 steps and can then end in
different local minima — the usual consequence of fp32 round-off at a bifurcation, not a kernel error.
Run `benchmarks/bench.py` on your own model.

Note: on MPS a plain `MACECalculator` sees a new graph shape whenever the number of edges changes (e.g. during a
relaxation), which is slow and makes memory grow on the e3nn path; long runs should use fixed-size (padded) graphs.

## Scope and limitations

* **Inference only.** The backward pass is first order: forces, stress, MD, relaxations, NEB and finite-difference
  Hessians work. Double backward (training on forces/stress, autograd Hessians) **raises an error** instead of
  silently returning incomplete gradients — use the unpatched model for training.
* float32 only (Metal has no float64); `mps` device only.
* Convolution: weighted *uvu* instructions with per-edge weights, equal multiplicity across node/output irreps, a
  multiple of 32 (128 is typical); the interaction block must expose MACE's `conv_fusion` path
  (`RealAgnosticInteractionBlock`, `RealAgnosticResidualInteractionBlock`).
* Symmetric contraction: any correlation order and l_max, full or reduced CG basis, element-dependent weights.
* Tested on the model above and on random models (`tests/`): l_max 2–3, correlation 2–3, scalar-only and
  `0e+1o+2e` hidden irreps. Other MACE variants (e.g. density/attention interaction blocks, multi-head, embeddings)
  are left on e3nn where unsupported, but have not been tested.

## Tests

```bash
python tests/test_kernels.py          # or: pytest tests/
```

Random MACE models + synthetic Pt(100)/adsorbate structures: agreement with e3nn for four architectures, rotation
equivariance, the double-backward guard, and fallback for unsupported blocks. Needs an Apple-silicon GPU.

## How it works

* **Convolution.** For each e3nn `TensorProduct`, the effective coefficient of every path is measured by probing it
  with unit inputs, so the generated Metal source reproduces e3nn's normalisation exactly. Forward: one thread per
  (receiving atom, channel), edges in receiver-sorted CSR order (deterministic, no atomics). Backward: d/dx per
  (sending atom, channel); d/dW and d/dY per edge with a simdgroup reduction over channels.
* **Symmetric contraction.** Coefficients `T[element][monomial × M][channel]` are assembled once in float64 from
  MACE's U matrices and weights; one thread per (atom, channel) evaluates the polynomial and its gradient.
* Kernels are compiled at patch time with `torch.mps.compile_shader`.

## Related

MACE is developing a kernel-backend protocol (ACEsuit/mace #1560, #1567). This package currently patches models
in place; it is intended to become a backend once that interface is available.

## License

MIT
