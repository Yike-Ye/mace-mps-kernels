"""
bench.py — energy+force timing and agreement, e3nn vs Metal kernels, for any MACE model and structure.

  python benchmarks/bench.py --model my.model --structure POSCAR [--batch 1 4 16]
  python benchmarks/bench.py                   # random model (tests/helpers.py) + synthetic Pt(100) slab

Single structure: MACECalculator.  --batch n: n copies with small random displacements in one forward pass
(the batched graph MACE builds for several structures).
"""
import argparse
import copy
import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'tests'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.io import read  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402
from mace import data as mace_data  # noqa: E402
from mace.tools import torch_geometric, torch_tools  # noqa: E402
from mace_mps_kernels import patch_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model')
    ap.add_argument('--structure')
    ap.add_argument('--batch', type=int, nargs='*', default=[1, 4, 16])
    ap.add_argument('--repeats', type=int, default=20)
    a = ap.parse_args()
    if a.model:
        ref = MACECalculator(model_paths=a.model, device='mps', default_dtype='float32')
    else:
        import helpers
        ref = helpers.calculator(helpers.make_model())
    atoms = read(a.structure) if a.structure else __import__('helpers').make_structure(seed=0)
    met = MACECalculator(models=[copy.deepcopy(ref.models[0])], device='mps', default_dtype='float32')
    print('patched:', patch_model(met.models[0])['patched'])
    print(f'{len(atoms)} atoms, torch {torch.__version__}')

    def batch_of(n):
        """n slightly displaced copies → one batched graph (as MACE builds it for several structures)."""
        rng = np.random.default_rng(0)
        keyspec = mace_data.KeySpecification(info_keys=ref.info_keys, arrays_keys=ref.arrays_keys)
        graphs = []
        with torch_tools.default_dtype('float32'):
            for _ in range(n):
                b = atoms.copy()
                b.positions += rng.normal(0, 0.01, b.positions.shape)
                cfg = mace_data.config_from_atoms(b, key_specification=keyspec, head_name=ref.head)
                graphs.append(mace_data.AtomicData.from_config(cfg, z_table=ref.z_table, cutoff=ref.r_max,
                                                               heads=ref.available_heads))
        return torch_geometric.Batch.from_data_list(graphs).to('mps')

    for n in a.batch:
        batch = batch_of(n)
        res = {}
        for tag, calc in (('e3nn', ref), ('metal', met)):
            model = calc.models[0]

            def run():
                return model(batch.to_dict(), compute_force=True, training=False)
            for _ in range(3):
                run()
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.repeats):
                out = run()
            torch.mps.synchronize()
            res[tag] = ((time.perf_counter() - t0) / a.repeats * 1e3 / n,
                        out['energy'].detach().cpu().double(), out['forces'].detach().cpu().double())
        dE = float((res['metal'][1] - res['e3nn'][1]).abs().max())
        dF = float((res['metal'][2] - res['e3nn'][2]).abs().max())
        print(f'n={n:3d}: e3nn {res["e3nn"][0]:7.1f} ms/structure   metal {res["metal"][0]:6.1f} ms/structure   '
              f'×{res["e3nn"][0] / res["metal"][0]:.1f}   max|ΔE| {1e3 * dE:.3f} meV  max|ΔF| {dF:.1e} eV/Å')


if __name__ == '__main__':
    main()
