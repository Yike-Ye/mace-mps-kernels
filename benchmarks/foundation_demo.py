"""
foundation_demo.py — mace-mps-kernels with the public MACE-MP-0 (medium) foundation model on other chemistries.

  python benchmarks/foundation_demo.py            (downloads MACE-MP-0 medium, 44 MB, to ~/.cache/mace on first use)

1. single points (energy + forces): Au(111)+CO/O, Cu(100)+H/OH, Ni(111)+N/NH, CeO2(111)+Au13 — e3nn vs Metal
2. relaxation of Au13 on CeO2(111) (bottom O–Ce–O trilayer fixed), LBFGS fmax 0.05, e3nn vs Metal:
   steps, final energy, wall time
"""
import copy
import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import fcc100, fcc111, molecule, surface  # noqa: E402
from ase.cluster import Icosahedron  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402
from ase.optimize import LBFGS  # noqa: E402
from ase.spacegroup import crystal  # noqa: E402
from mace.calculators import mace_mp, MACECalculator  # noqa: E402
from mace_mps_kernels import patch_model  # noqa: E402


def decorate(slab, frags, n, seed, hmin=1.3, hmax=2.0):
    rng = np.random.default_rng(seed)
    ztop = slab.positions[:, 2].max()
    atoms = slab.copy()
    added = tries = 0
    while added < n and tries < 2000:
        tries += 1
        f = frags[rng.integers(len(frags))].copy()
        f.rotate(rng.uniform(0, 360), 'z', center=(0, 0, 0))
        xy = rng.uniform(0, 1, 2) @ slab.cell[:2, :2]
        f.positions += np.array([xy[0], xy[1], ztop + rng.uniform(hmin, hmax)]) - f.positions[0]
        trial = atoms + f
        if trial.get_all_distances(mic=True)[len(atoms):, :len(atoms)].min() > 1.7:
            atoms, added = trial, added + 1
    return atoms


def ceo2_au13():
    ceo2 = crystal(['Ce', 'O'], basis=[(0, 0, 0), (0.25, 0.25, 0.25)], spacegroup=225, cellpar=[5.411] * 3 + [90] * 3)
    slab = surface(ceo2, (1, 1, 1), layers=3, vacuum=0.0)     # 3 O–Ce–O trilayers in a 7.65 Å (2×2 primitive) cell
    slab = slab.repeat((2, 2, 1))                              # 15.3 Å × 15.3 Å, 144 atoms
    slab.center(vacuum=8.0, axis=2)
    z = slab.positions[:, 2]
    slab.set_constraint(FixAtoms(indices=np.where(z < z.min() + 2.0)[0]))   # bottom trilayer
    au = Icosahedron('Au', noshells=2)                       # 13 atoms
    c = slab.cell[0] / 2 + slab.cell[1] / 2                  # centre of the surface cell
    au.positions += np.array([c[0], c[1], z.max() + 2.4 - au.positions[:, 2].min()]) - \
        np.array([au.positions[:, 0].mean(), au.positions[:, 1].mean(), 0])
    s = slab + au
    s.pbc = True
    return s


def systems():
    co = molecule('CO')
    oh = Atoms('OH', [[0, 0, 0], [0, 0, 0.98]])
    nh = Atoms('NH', [[0, 0, 0], [0, 0, 1.03]])
    au = fcc111('Au', size=(4, 4, 4), vacuum=8.0)
    cu = fcc100('Cu', size=(5, 5, 4), vacuum=8.0)
    ni = fcc111('Ni', size=(5, 5, 4), vacuum=8.0)
    return {
        'Au(111)+CO/O': decorate(au, [co, Atoms('O')], 8, 0),
        'Cu(100)+H/OH': decorate(cu, [Atoms('H'), oh], 12, 1),
        'Ni(111)+N/NH': decorate(ni, [Atoms('N'), nh], 10, 2),
        'CeO2(111)+Au13': ceo2_au13(),
    }


def timed_ef(calc, atoms, R=10):
    a = atoms.copy()
    a.calc = calc
    a.get_forces()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(R):
        a.positions[-1] += 1e-4          # invalidate the ASE cache
        F = a.get_forces()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / R * 1e3, a.get_potential_energy(), F


def main():
    # MACE-MP-0 is stored in float64, which MPS cannot hold: load on CPU as float32, then move to the GPU
    model = mace_mp(model='medium', device='cpu', default_dtype='float32').models[0].float().to('mps')
    ref = MACECalculator(models=[model], device='mps', default_dtype='float32')
    met = MACECalculator(models=[copy.deepcopy(model)], device='mps', default_dtype='float32')
    rep = patch_model(met.models[0])
    print('patched:', rep['patched'], ' skipped:', rep['skipped'])
    S = systems()
    print('\n[1] single points (energy + forces)')
    for name, atoms in S.items():
        t0, E0, F0 = timed_ef(ref, atoms)
        t1, E1, F1 = timed_ef(met, atoms)
        print(f'  {name:16s} {len(atoms):4d} atoms  e3nn {t0:6.1f} ms  metal {t1:5.1f} ms  ×{t0 / t1:4.1f}   '
              f'|ΔE| {1e3 * abs(E1 - E0):.3f} meV  max|ΔF| {np.abs(F1 - F0).max():.1e} eV/Å')
    print('\n[2] relaxation of Au13 on CeO2(111): LBFGS fmax 0.05, ≤500 steps')
    for tag, calc in (('e3nn', ref), ('metal', met)):
        a = S['CeO2(111)+Au13'].copy()
        a.calc = calc
        opt = LBFGS(a, logfile=None)
        t0 = time.perf_counter()
        conv = opt.run(fmax=0.05, steps=500)
        dt = time.perf_counter() - t0
        print(f'  {tag:5s} converged {conv!s:5s} steps {opt.nsteps:3d}  E {a.get_potential_energy():.4f} eV  '
              f'{dt:6.1f} s ({1e3 * dt / max(opt.nsteps, 1):.0f} ms/step)')
        if tag == 'metal':
            from ase.io import write
            write(os.path.join(HERE, 'ceo2_au13_relaxed.extxyz'), a)


if __name__ == '__main__':
    main()
