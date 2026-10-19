"""
Correctness tests (pytest, or `python tests/test_kernels.py`).  Apple-silicon GPU required.

* every CONFIGS model: all convolutions and symmetric contractions are patched, energies and forces of the patched
  model match e3nn (fp32 tolerance) on synthetic structures
* rotation equivariance of the patched model (energy invariant, forces rotate)
* double backward (training on forces) raises instead of silently returning wrong gradients
* an unsupported block (multiplicity not a multiple of 32) is skipped and keeps working on e3nn
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import helpers  # noqa: E402

E_TOL = 1e-5        # eV per atom
F_TOL = 1e-5        # relative to the largest force component


def _check_pair(config, seeds=(0, 1, 2)):
    ref, met, report = helpers.pair(config)
    assert not report['skipped'], report['skipped']
    assert len(report['patched']) == 4, report
    for s in seeds:
        atoms = helpers.make_structure(seed=s)
        E0, F0 = helpers.energy_forces(ref, atoms)
        E1, F1 = helpers.energy_forces(met, atoms)
        dE = abs(E1 - E0) / len(atoms)
        dF = np.abs(F1 - F0).max() / max(np.abs(F0).max(), 1e-3)
        assert dE < E_TOL, (config, s, dE)
        assert dF < F_TOL, (config, s, dF)
    return dE, dF


def test_matches_e3nn_all_configs():
    for config in helpers.CONFIGS:
        _check_pair(config)


def test_rotation_equivariance():
    _, met, _ = helpers.pair('l3_corr3_reduced')
    atoms = helpers.make_structure(seed=3)
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    E0, F0 = helpers.energy_forces(met, atoms)
    E1, F1 = helpers.energy_forces(met, atoms, R)
    assert abs(E1 - E0) / len(atoms) < E_TOL
    assert np.abs(F1 - F0 @ R.T).max() / np.abs(F0).max() < 1e-4


def test_double_backward_raises():
    _, met, _ = helpers.pair('l3_corr3_reduced')
    model = met.models[0]
    batch = met._atoms_to_batch(helpers.make_structure(seed=4)).to_dict()
    batch['positions'].requires_grad_(True)
    out = model(batch, compute_force=False, training=True)
    try:                                   # forces with create_graph=True (as in force training) must refuse
        (f,) = torch.autograd.grad(out['energy'].sum(), batch['positions'], create_graph=True)
        torch.autograd.grad(f.sum(), batch['positions'])
    except RuntimeError as e:
        assert 'double backward' in str(e), e
        return
    raise AssertionError('double backward did not raise')


def test_unsupported_block_is_skipped():
    import copy
    from mace_mps_kernels import patch_model
    ref = helpers.calculator(helpers.make_model('48x0e+48x1o', 3, 3, True))
    met = helpers.calculator(copy.deepcopy(ref.models[0]))
    report = patch_model(met.models[0], verbose=False)
    assert all('conv_tp' in n for n, _ in report['skipped']) and len(report['skipped']) == 2, report
    atoms = helpers.make_structure(seed=5)
    E0, F0 = helpers.energy_forces(ref, atoms)
    E1, F1 = helpers.energy_forces(met, atoms)          # symmetric contractions patched, convolutions on e3nn
    assert abs(E1 - E0) / len(atoms) < E_TOL
    assert np.abs(F1 - F0).max() / np.abs(F0).max() < F_TOL


if __name__ == '__main__':
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as e:                     # noqa: BLE001
                print(f'FAIL {name}: {type(e).__name__}: {e}')
                failures.append(name)
    if failures:
        raise SystemExit(f'{len(failures)} test(s) failed: {", ".join(failures)}')
