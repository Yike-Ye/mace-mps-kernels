"""
mace-mps-kernels — fused Metal kernels for MACE inference on Apple-silicon GPUs (PyTorch MPS).

    from mace_mps_kernels import patch_model
    report = patch_model(model)          # model already on device='mps', float32; modified in place

Replaces, where supported:
  * each interaction block's convolution (gather → channel-wise tensor product → scatter-sum) via MACE's own
    ``conv_fusion`` hook,
  * each product block's symmetric contraction.
Everything else stays on the original e3nn / PyTorch path.  Inference only: the backward pass is first order
(energies, forces, stress, finite-difference Hessians, MD, NEB, relaxations); double backward — needed for training on
forces — raises an error.
"""
import inspect
import warnings

import torch

from .conv import MetalConvTP, conv_supported
from .symmetric import MetalSymmetricContraction, sc_supported

__version__ = '0.1.0'
__all__ = ['patch_model', 'MetalConvTP', 'MetalSymmetricContraction']


def _block_has_conv_fusion(blk):
    try:
        return 'conv_fusion' in inspect.getsource(type(blk).forward)
    except (OSError, TypeError):
        return False


def patch_model(model, conv=True, symmetric=True, verbose=True):
    """Swap supported ops for Metal kernels, in place.  Returns {'patched': [...], 'skipped': [(name, reason)]}."""
    dev = next(model.parameters()).device
    if dev.type != 'mps':
        raise RuntimeError(f'model must be on the mps device (found {dev})')
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError('float32 models only (Metal has no float64)')
    patched, skipped = [], []
    if conv:
        for i, blk in enumerate(getattr(model, 'interactions', [])):
            name = f'interactions.{i}.conv_tp'
            if isinstance(getattr(blk, 'conv_tp', None), MetalConvTP):
                continue
            if hasattr(blk, 'conv_fusion'):
                skipped.append((name, 'block already uses another fused convolution'))
                continue
            if not _block_has_conv_fusion(blk):
                skipped.append((name, f'{type(blk).__name__} has no conv_fusion path'))
                continue
            ok, why = conv_supported(blk.conv_tp)
            if not ok:
                skipped.append((name, why))
                continue
            blk.conv_tp = MetalConvTP(blk.conv_tp)
            blk.conv_fusion = True
            patched.append(name)
    if symmetric:
        for i, prod in enumerate(getattr(model, 'products', [])):
            name = f'products.{i}.symmetric_contractions'
            sc = getattr(prod, 'symmetric_contractions', None)
            if isinstance(sc, MetalSymmetricContraction):
                continue
            cfg = getattr(prod, 'cueq_config', None)
            if cfg is not None and getattr(cfg, 'enabled', False):
                skipped.append((name, 'cuequivariance layout'))
                continue
            ok, why = sc_supported(sc)
            if not ok:
                skipped.append((name, why))
                continue
            prod.symmetric_contractions = MetalSymmetricContraction(sc, device=dev)
            patched.append(name)
    if verbose:
        for name, why in skipped:
            warnings.warn(f'mace-mps-kernels: kept e3nn for {name}: {why}')
    return dict(patched=patched, skipped=skipped)
