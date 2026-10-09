"""Random MACE models + synthetic Pt(100)/adsorbate structures for the tests (no external data or models needed)."""
import copy
import os
import numpy as np
import torch
from ase import Atoms
from ase.build import fcc100, molecule

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from mace import modules  # noqa: E402  (import mace before e3nn: it allow-lists e3nn's constants for torch.load)
from mace.calculators import MACECalculator  # noqa: E402
from e3nn import o3  # noqa: E402

ELEMENTS = [1, 7, 8, 78]          # H, N, O, Pt

CONFIGS = {
    # name: (hidden_irreps, max_ell, correlation, use_reduced_cg)
    'l3_corr3_reduced': ('32x0e+32x1o', 3, 3, True),
    'l3_corr3_fullcg': ('32x0e+32x1o', 3, 3, False),
    'l2_corr2_scalar': ('64x0e', 2, 2, True),
    'l3_corr3_l2hidden': ('32x0e+32x1o+32x2e', 3, 3, True),
}


def make_model(hidden='32x0e+32x1o', max_ell=3, correlation=3, reduced_cg=True, seed=0, r_max=5.0):
    torch.manual_seed(seed)
    n = len(ELEMENTS)
    model = modules.ScaleShiftMACE(
        atomic_inter_scale=1.0, atomic_inter_shift=0.0, r_max=r_max, num_bessel=8, num_polynomial_cutoff=5,
        max_ell=max_ell,
        interaction_cls=modules.interaction_classes['RealAgnosticResidualInteractionBlock'],
        interaction_cls_first=modules.interaction_classes['RealAgnosticInteractionBlock'],
        num_interactions=2, num_elements=n, hidden_irreps=o3.Irreps(hidden), MLP_irreps=o3.Irreps('16x0e'),
        atomic_energies=np.zeros(n), avg_num_neighbors=20.0, atomic_numbers=ELEMENTS, correlation=correlation,
        gate=torch.nn.functional.silu, use_reduced_cg=reduced_cg)
    with torch.no_grad():                         # make every weight non-trivial (some are zero-initialised)
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model.float()


def calculator(model):
    return MACECalculator(models=[model], device='mps', default_dtype='float32')


def pair(config):
    """(reference calculator, patched calculator) for a CONFIGS entry; patched = deep copy + patch_model."""
    from mace_mps_kernels import patch_model
    hidden, lmax, corr, red = CONFIGS[config]
    ref = calculator(make_model(hidden, lmax, corr, red))
    met = calculator(copy.deepcopy(ref.models[0]))
    report = patch_model(met.models[0], verbose=False)
    return ref, met, report


def make_structure(seed=0, n_frag=10):
    """4×4×3 Pt(100) slab with random NH3/NH2/NH/N/OH/O fragments 1.2–2.2 Å above the surface (min distance 1.6 Å)."""
    rng = np.random.default_rng(seed)
    slab = fcc100('Pt', size=(4, 4, 3), vacuum=8.0)
    ztop = slab.positions[:, 2].max()
    frags = [molecule('NH3'), Atoms('NH2', [[0, 0, 0], [0.8, 0, 0.6], [-0.8, 0, 0.6]]),
             Atoms('NH', [[0, 0, 0], [0, 0, 1.03]]), Atoms('N'), Atoms('OH', [[0, 0, 0], [0, 0, 0.98]]), Atoms('O')]
    atoms = slab.copy()
    tries = 0
    while len(atoms) < len(slab) + 3 * n_frag and tries < 2000:
        tries += 1
        f = frags[rng.integers(len(frags))].copy()
        f.rotate(rng.uniform(0, 360), 'z', center=(0, 0, 0))
        xy = rng.uniform(0, 1, 2) @ slab.cell[:2, :2]
        f.positions += np.array([xy[0], xy[1], ztop + rng.uniform(1.2, 2.2)]) - f.positions[0]
        trial = atoms + f
        d = trial.get_all_distances(mic=True)[len(atoms):, :len(atoms)]
        if d.min() > 1.6:
            atoms = trial
        if sum(1 for a in atoms if a.symbol != 'Pt') >= 3 * n_frag:
            break
    atoms.pbc = True
    return atoms


def energy_forces(calc, atoms, R=None):
    a = atoms.copy()
    if R is not None:
        a.positions = a.positions @ R.T
        a.cell = a.cell @ R.T
    a.calc = calc
    return a.get_potential_energy(), a.get_forces()
