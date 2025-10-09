# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3.8
#     language: python
#     name: p38
# ---

# % [markdown]
# ## PyTorch Utilities

# % code_folding=[5, 12, 42, 373]


import gc
import math
import pprint
import numpy as np
import torch
import warnings
import platform
from torch import nn
from textwrap import dedent
from itertools import product
from types import MappingProxyType
from collections import defaultdict
from collections.abc import Callable
from torch.nn.init import calculate_gain
from partnn.io_utils import eval_formula
from collections import OrderedDict
from partnn.io_utils import deep2hie, hie2deep
from torch.nn.init import _calculate_correct_fan
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch._guards import detect_fake_mode

# +
warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd "
    "dilation may require a zero-padded copy of the input")

(pyver_major, pyver_minor, pyver_patch) = map(int, platform.python_version_tuple())
is_dictordrd = (pyver_major > 3) or ((pyver_major == 3) and (pyver_minor >= 6))
odict = dict if is_dictordrd else OrderedDict

# +

def isscalar(v):
    if torch.is_tensor(v):
        return v.numel() == 1
    else:
        return np.isscalar(v)


def torch_qr_eff(a):
    """
    Due to a bug in MAGMA, qr on cuda is super slow for small matrices. 
    Therefore, this step must be performed on the cpu.
    
    See the following:
        https://github.com/pytorch/pytorch/issues/22573
        https://github.com/cornellius-gp/gpytorch/pull/1224
    """
    assert not a.requires_grad
    q, r = torch.qr(a.detach().cpu(), some=False)
    return q.to(device=a.device), r.to(device=a.device)


def torch_astype(input, dtype: str | torch.dtype):
    assert isinstance(torch.bool, (str, torch.dtype)), dedent(f'''
        dtype must be either a string or a torch.dtype:
            dtype: {dtype}''')

    name2dtype = {
        'float32': torch.float32, 'float': torch.float, 'float64': torch.float64,
        'double': torch.double, 'complex64': torch.complex64, 'cfloat': torch.cfloat,
        'complex128': torch.complex128, 'cdouble': torch.cdouble, 'float16': torch.float16,
        'half': torch.half, 'bfloat16': torch.bfloat16, 'uint8': torch.uint8,
        'int8': torch.int8, 'int16': torch.int16, 'short': torch.short,
        'int32': torch.int32, 'int': torch.int, 'int64': torch.int64,
        'long': torch.long, 'bool': torch.bool}
    name2dtype = {**name2dtype, **{f'torch.{dtname}': dtype
        for dtname, dtype in name2dtype.items()}}

    tch_dtype = name2dtype[dtype] if isinstance(dtype, str) else dtype

    return input.to(dtype=tch_dtype)


def profmem():
    """
    profiles the memory usage by alive pytorch tensors.
    
    Outputs
    -------
    stats (dict): a torch.device mapping to the number of 
        bytes used by the pytorch tensors.
    """
    fltr_msg  = "torch.distributed.reduce_op is deprecated, "
    fltr_msg += "please use torch.distributed.ReduceOp instead"
    warnings.filterwarnings("ignore", message=fltr_msg)
    fltr_msg  = "`torch.distributed.reduce_op` is deprecated, "
    fltr_msg += "please use `torch.distributed.ReduceOp` instead"
    warnings.filterwarnings("ignore", message=fltr_msg)

    stats = defaultdict(lambda: 0)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') 
                and torch.is_tensor(obj.data)):            
                stats[str(obj.device)] += obj.numel() * obj.element_size()
        except:
            pass
    
    n_cuda = torch.cuda.device_count()
    for name_dvc in [f'cuda:{i}' for i in range(n_cuda)]:
        mem_maxdvc = torch.cuda.max_memory_allocated(device=name_dvc)
        stats[name_dvc] = max(mem_maxdvc, stats.get(name_dvc, 0))
    
    return stats


def batch_kaiming_uniform_(tensor, batch_rng, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    n_seeds, = batch_rng.shape
    assert tensor.shape[0] == n_seeds
    
    fan = _calculate_correct_fan(tensor[0], mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        u = (2 * bound) * batch_rng.rand(*tensor.shape) - bound
        return tensor.copy_(u)


def torch_quantile(input, q, dim=None, keepdim=False, interpolation='linear', 
    method='sort', qdim=None):
    """
    This is a substitute for the `torch.quantile` function that can handle 
    multiple dimensions in the input.

    This function provides two underlying torch methods to choose 
    from: `sort`, `quantile`, and `kthvalue`.

    The `quantile` method seems slow with large input sizes:
        https://github.com/pytorch/pytorch/issues/64947

    If input is large and `q` has a small number of elements, running 
    `kthvalue` multiple times may be the most efficient choince.

    When q has many elements, it may be worth it to sort the input once, 
    and extract the quantiles.

    It also allows the user to determine where the quantiles dimension 
    should be inserted in the output tensor if/when a vector q is given.

    If you want the exact same behavior of `torch.quantile` regarding the 
    quantile dimension in the output, then you must specify `qdim=0`

        * In other words, with `qdim=0`, the output tensor will be shaped 
          just like `torch.quantile` does: the `q` dimension will be 
          prepended to the left of the input shape.

    Args:
        input: (torch.tensor) The input tensor. See `torch.quantile`.

        q: (float | torch.tensor) The quantiles. Must be either zero or 
            one-dimensional. See `torch.quantile`.

        dim: (None | int | list) The reduction dimensions. If `None`, all 
            dimensions will be reduced, and it will be as if all dimensions 
            were specified for reduction.

            If `dim` is a list, it should have at least one element.

        keepdim: (bool) Whether the reduced dimensions should be retained 
            as singleton dimensions in the output or not.

        method: (str) The torch method to use for computing the quantiles. 
            Should be one of `['sort', 'quantile', 'kthvalue']`.
        
        qdim: (int | None) The dimension for the quantile data to be inserted. 
            
            If not provided, we make the following substitution:
                1. If `qdim==None` and `len(dim)==1`, we set `qdim=dim[0]`
                2. If `q` is scalar, qdim is unnecessary.
                3. If `len(dim)>1`, then an integer `qdim` must be 
                    specified by the user.
            
            When `q` is a vector, the `output.shape[qdim]` is guaranteed 
            to be the number of quantiles (i.e., `len(q)`).
    """
    n_inpdims = input.ndim

    # Sanity checking the provided reduction dimensions
    rdc_dims0 = list(range(n_inpdims)) if (dim is None) else dim
    rdc_dims1 = [rdc_dims0] if isinstance(rdc_dims0, int) else rdc_dims0
    assert isinstance(rdc_dims1, (list, tuple, set)), f'Bad dim: {dim}'
    rdc_dims2 = []
    for i_dim in rdc_dims1:
        assert isinstance(i_dim, int), f'Bad dim: {dim}'
        i_dim2 = i_dim + n_inpdims if i_dim < 0 else i_dim
        assert (n_inpdims > i_dim2 >= 0), f'Bad dim: {dim}'
        rdc_dims2.append(i_dim2)
    assert len(rdc_dims2) == len(set(rdc_dims2)), f'Bad dim: {dim}'
    assert len(rdc_dims2) > 0, f'Bad dim: {dim}'

    # Sanity checking q
    is_qscalar = isinstance(q, float) or (torch.is_tensor(q) and (len(q.shape) == 0))
    q2 = (torch.tensor(q, dtype=input.dtype, device=input.device) 
        if isinstance(q, float) else q)
    q2 = q2.reshape(1) if is_qscalar else q2
    assert torch.is_tensor(q2), f'Bad q type: {q}'
    assert q2.device == input.device, f'Bad q device: {x.device} != {q.device}'
    assert q2.dtype == input.dtype, f'Bad q dtype: {x.dtype} != {q.dtype}'
    assert q2.ndim == 1, f'Too many q dims: {q2.shape}'
    assert (q2 >= 0).all(), f'Bad q vals: q2.min() = {q2.min()}'
    assert (q2 <= 1).all(), f'Bad q vals: q2.max() = {q2.max()}'
    n_qnts, = q2.shape

    # Applying any necessary transpositions to the input to make sure the reduction 
    # dimensions are placed all in one place.
    rdc_dims = sorted(rdc_dims2)
    n_rdcdims = len(rdc_dims)

    if (qdim is None) and is_qscalar:
        qdim2 = 0
    elif (qdim is None) and (not is_qscalar):
        assert len(rdc_dims) == 1, dedent(f'''
            qdim must be specified when multiple reduction 
            dimensions are specified:
                qdim = {qdim}
                dim = {dim}''')
        qdim2, = rdc_dims
    else:
        qdim2 = qdim
    assert isinstance(qdim2, int), f'Bad qdim: {qdim}'

    # If the reduction dimensions are sequential, no transpositions are needed.
    avoid_trnsps = (rdc_dims == list(range(rdc_dims[0], rdc_dims[-1] + 1)))
    n_qntldim = math.prod(input.shape[i_dim] for i_dim in rdc_dims)
    if avoid_trnsps:
        input2_shp = list(input.shape)
        for i_dim in rdc_dims:
            input2_shp[i_dim] = (1 if keepdim else None)
        input2_shp = [i_dim for i_dim in input2_shp if i_dim is not None]

        rdc_dim = rdc_dims[0]
        input2_shp.insert(rdc_dim, n_qntldim)
        input2 = input.reshape(*input2_shp)
    else:
        # The reduction dimensions are all over the place, so we have to move 
        # them to the right.
        rdc_dim = -1
        input2 = input.unsqueeze(-1)
        for i_dim in rdc_dims:
            input2 = input2.unsqueeze(-1).transpose(i_dim, -1).flatten(-2, -1)
        if not keepdim:
            assert all((input2.ndim - 1) > i_dim >= 0 for i_dim in rdc_dims)
            input2 = input2.squeeze(rdc_dims)

    # Next stage is to remove the single `rdc_dim` and replace it with `qdim2`
    assert ((-(input2.ndim + 1)) < qdim2 < input2.ndim), dedent(f'''
        Bad qdim:
            input.shape = {input.shape}
            dim: {dim}
            qdim: {qdim}
            keepdim: {keepdim}''')
    
    rdc_dim2 = (rdc_dim + input2.ndim) if (rdc_dim < 0) else rdc_dim
    qdim3 = (qdim2 + input2.ndim) if (qdim2 < 0) else qdim2
    if (qdim3 > rdc_dim2):
        rdc_dim3, qdim4 = rdc_dim2, (qdim3 + 1)
    else:
        rdc_dim3, qdim4 = (rdc_dim2 + 1), qdim3
    input3 = input2.unsqueeze(qdim4).transpose(rdc_dim3, qdim4).squeeze(rdc_dim3)
    assert input3.shape[qdim2] == n_qntldim

    # We will run the quantile function with a vector q and `keepdim=True`.
    if method == 'quantile':
        input_qntls1 = input3.quantile(dim=qdim2, keepdim=True, q=q2, 
            interpolation=interpolation)
    elif method == 'sort':
        input3_srtd, _ = input3.sort(dim=qdim2)
        n_qntldim = input3.shape[qdim2]
        q_bcstshp = [1 for _ in range(input3.ndim)]
        q_bcstshp[qdim2], = q2.shape
        i_qntl = q2.reshape(*q_bcstshp) * (n_qntldim - 1)
        if interpolation in ('linear', 'midpoint'):
            i_qntlow = i_qntl.floor().to(dtype=torch.long)
            i_qnthigh = (i_qntlow + 1).clamp(max=n_qntldim - 1)
            v_qntlow = input3_srtd.take_along_dim(i_qntlow, dim=qdim2)
            v_qnthigh = input3_srtd.take_along_dim(i_qnthigh, dim=qdim2)
            if interpolation == 'linear':
                weight = i_qntl - i_qntl.floor()
            elif interpolation == 'midpoint':
                weight = 0.5
            else:
                weight = None
            input_qntls1 = (1 - weight) * v_qntlow + weight * v_qnthigh
        elif interpolation in ('lower', 'higher', 'nearest'):
            interper = {'lower': torch.floor, 'higher': torch.ceil, 
                'nearest': torch.round}[interpolation]
            i_qntl2 = interper(i_qntl).to(dtype=torch.long)
            input_qntls1 = input3_srtd.take_along_dim(i_qntl2.reshape(*q_bcstshp), dim=qdim2)
        else:
            raise ValueError(f'undefined interpolation={interpolation}')
    elif method == 'kthvalue':
        kthvals_list = []
        for i_q in range(n_qnts):
            i_qntl = q2[i_q] * (n_qntldim - 1)
            if interpolation in ('linear', 'midpoint'):
                i_qntllow = i_qntl.floor().to(dtype=torch.long)
                i_qntlhigh = (i_qntllow + 1).clamp(max=n_qntldim - 1)
                v_qntlow, _ = input3.kthvalue(1+i_qntllow, dim=qdim2, keepdim=True)
                v_qnthigh, _ = input3.kthvalue(1+i_qntlhigh, dim=qdim2, keepdim=True)
                if interpolation == 'linear':
                    weight = i_qntl - i_qntl.floor()
                elif interpolation == 'midpoint':
                    weight = 0.5
                else:
                    weight = None
                v_qnt = (1 - weight) * v_qntlow + weight * v_qnthigh
            elif interpolation in ('lower', 'higher', 'nearest'):
                interper = {'lower': torch.floor, 'higher': torch.ceil, 
                    'nearest': torch.round}[interpolation]
                i_qntl2 = interper(i_qntl).to(dtype=torch.long)
                v_qnt, _ = input3.kthvalue(1+i_qntl2, dim=qdim2, keepdim=True)
            else:
                raise ValueError(f'undefined interpolation={interpolation}')
            kthvals_list.append(v_qnt)
        input_qntls1 = torch.cat(kthvals_list, dim=qdim2)
    else:
        raise ValueError(f'undefined method={method}')
    
    if method == 'quantile':
        # Due to setting `keepdim == True`, we have the following condition
        assert input_qntls1.shape[qdim3 + 1] == 1
        assert input_qntls1.shape[0] == n_qnts
        input_qntls2 = input_qntls1.transpose(0, qdim3 + 1).squeeze(0)
    else:
        input_qntls2 = input_qntls1

    # If q was a scalar, we squeeze out its dimension
    if is_qscalar:
        input_qntls3 = input_qntls2.squeeze(qdim2)
    else:
        input_qntls3 = input_qntls2
        assert input_qntls3.shape[qdim2] == n_qnts, dedent(f'''
            Our output guarantee failed somehow:
                output shape: {input_qntls3.shape}
                qdim: {qdim}
                qdim2: {qdim2}
                n_qnts: {n_qnts}''')

    return input_qntls3


def test_quantile():
    torch.manual_seed(12345)
    x = 8 + torch.randn(3, 5, 4, 2, 6, 2)

    keepdim_lst = [True, False]
    method_lst = ['sort', 'quantile', 'kthvalue']
    interp_lst = ['linear', 'lower', 'higher', 'midpoint', 'nearest']
    combs = list(product(keepdim_lst, method_lst, interp_lst))

    # Test 1: Scalar q, single dim, qdim=None
    q = 0.43
    dim = 0
    for keepdim, method, interpolation in combs:
        y = torch_quantile(x, q, dim=dim, keepdim=keepdim, 
            interpolation=interpolation, method=method, qdim=None)
        y_ref = x.quantile(q, dim=dim, keepdim=keepdim, 
            interpolation=interpolation)
        
        assert y.shape == y_ref.shape, f'{y.shape} != {y_ref.shape}'
        y_relerr = (y - y_ref).abs() / (y.abs() + y_ref.abs())
        assert y_relerr.max() < 1e-6

    # Test 2: Scalar q, multi-dim, qdim=None
    q = torch.tensor(0.57)
    for dim in ([1, 2, 3], [1, 3, 4]):
        for keepdim, method, interpolation in combs:
            y = torch_quantile(x, q, dim=dim, keepdim=keepdim, 
                interpolation=interpolation, method=method, qdim=None)
            
            x2 = x.unsqueeze(-1)
            for d in dim:
                x2 = x2.unsqueeze(-1).transpose(-1, d).flatten(-2, -1)
            if not keepdim:
                assert all(d > 0 for d in dim)
                x2 = x2.squeeze(dim)
            y_ref = x2.quantile(q, dim=-1, keepdim=True, 
                interpolation=interpolation).squeeze(-1)
            
            assert y.shape == y_ref.shape, f'{y.shape} != {y_ref.shape}'
            y_relerr = (y - y_ref).abs() / (y.abs() + y_ref.abs())
            assert y_relerr.max() < 1e-6

    # Test 3: Multi-q, multi-dim, various qdim
    q = torch.tensor([0.43, 0.61, 0.23, 0.74, 0.11, 0.83])
    for qdim in [0, 3, -1]:
        for dim in ([1, 2, 3], [1, 3, 4]):
            for keepdim, method, interpolation in combs:
                y = torch_quantile(x, q, dim=dim, keepdim=keepdim, 
                    interpolation=interpolation, method=method, qdim=qdim)
                
                x2 = x.unsqueeze(-1)
                for d in dim:
                    x2 = x2.unsqueeze(-1).transpose(-1, d).flatten(-2, -1)
                if not keepdim:
                    assert all(d > 0 for d in dim)
                    x2 = x2.squeeze(dim)
                y_ref = x2.quantile(q, dim=-1, keepdim=False, 
                    interpolation=interpolation)
                qdimp = (qdim + y_ref.ndim) if (qdim < 0) else qdim
                y_ref = y_ref.unsqueeze(qdimp + 1).transpose(0, qdimp + 1).squeeze(0)
                
                assert y.shape == y_ref.shape, f'{y.shape} != {y_ref.shape}'
                y_relerr = (y - y_ref).abs() / (y.abs() + y_ref.abs())
                assert y_relerr.max() < 1e-6


def get_srtalgnidxs(src_pnts, trg_pnts, dim):
    """
    Computes the sort-alignment indices for the target tensor such that 
    indexing the target with them would produce identically-ranked elements 
    to the source tensor.

    This is useful for sliced wasserstein calculations. In particular, consider 
    the following example:

        ```
        src_pnts = torch.rand(10)
        trg_pnts = torch.rand(10)

        i_trgalgn = get_srtalgnidxs(src_pnts, trg_pnts, dim=0)

        wass_ref = (src_pnts.sort().values - trg_pnts.sort().values).square().sum()
        wass_pred = (src_pnts - trg_pnts[i_trgalgn]).square().sum()

        assert torch.isclose(wass_ref, wass_pred)
        ```

    Notice how `src_pnts` did not require any indexing for computing `wass_pred`, 
    and all the necessary alignment indexing was performed on `trg_pnts` using 
    the `i_trgalgn` index.

    Parameters
    ----------
    src_pnts: (torch.tensor) The source points. These are expected to remain 
        intact for wasserstein computations.

    trg_pnts: (torch.tensor) The target points. These are expected to be re-indexed 
        using this function's output for wasserstein computations.
    
    dim: (int) The alignment dimension. The rest of the dimensions are treated as 
        batch dimensions.

    Returns
    -------
    i_algntrg: (torch.tensor) A long tensor with the same shape as `trg_pnts`. 
        Indexing the target tensor with these indices will make sure that the 
        source and target tesors are aligned with respect to their sorted ranks.
    """
    assert isinstance(dim, int), f'only integer dims are accepted: dim={dim}'
    assert src_pnts.shape == trg_pnts.shape

    # Getting the argsort of source
    i_sortsrc = src_pnts.argsort(dim=dim)
    assert i_sortsrc.shape == src_pnts.shape

    # Getting the ranking of the source
    rnk_srcpnts = i_sortsrc.argsort(dim=dim)
    assert rnk_srcpnts.shape == src_pnts.shape

    # Getting the argsort of target
    i_sorttrg = trg_pnts.argsort(dim=dim)
    assert i_sorttrg.shape == src_pnts.shape

    # The indices aligning the target to the source
    i_algntrg = torch.take_along_dim(i_sorttrg, rnk_srcpnts, dim=dim)
    assert i_algntrg.shape == src_pnts.shape

    return i_algntrg


def test_srtalgnidxs():
    torch.manual_seed(12345)

    ################################################################
    ################# One-Dimensional Simple Test ##################
    ################################################################
    src_pnts1d = torch.rand(10)
    trg_pnts1d = torch.rand(10)

    i_trgalgn = get_srtalgnidxs(src_pnts1d, trg_pnts1d, dim=0)

    wass_ref1d = (src_pnts1d.sort().values - trg_pnts1d.sort().values).square().sum()
    wass_pred1d = (src_pnts1d - trg_pnts1d[i_trgalgn]).square().sum()

    assert torch.isclose(wass_ref1d, wass_pred1d)

    ################################################################
    ################ Multi-Dimensional Simple Test #################
    ################################################################

    n_seeds, n_slc, n_mb, dim = 10, 25, 32, -1

    src_pnts = 3 + 5 * torch.rand(n_seeds, n_slc, n_mb)
    assert src_pnts.shape == (n_seeds, n_slc, n_mb)

    trg_pnts = 1 + 2 * torch.rand(n_seeds, n_slc, n_mb)
    assert trg_pnts.shape == (n_seeds, n_slc, n_mb)

    #################################################
    ### The Reference Sliced Wasserstein Distance ###
    #################################################
    src_pntssrtd = src_pnts.sort(dim=-1).values
    assert src_pntssrtd.shape == (n_seeds, n_slc, n_mb)

    trg_pntssrtd = trg_pnts.sort(dim=-1).values
    assert trg_pntssrtd.shape == (n_seeds, n_slc, n_mb)

    wass_ref = (src_pntssrtd - trg_pntssrtd).abs().square().sum(dim=-1)
    assert wass_ref.shape == (n_seeds, n_slc)

    #################################################
    ### The Predicted Sliced Wasserstein Distance ###
    #################################################
    i_algntrg = get_srtalgnidxs(src_pnts, trg_pnts, dim=-1)
    assert i_algntrg.shape == (n_seeds, n_slc, n_mb)

    trg_pntsalgnd = torch.take_along_dim(trg_pnts, i_algntrg, dim=-1)
    assert trg_pntsalgnd.shape == (n_seeds, n_slc, n_mb)

    wass_pred = (src_pnts - trg_pntsalgnd).abs().square().sum(dim=-1)
    assert wass_pred.shape == (n_seeds, n_slc)

    #################################################
    ### Test 1: Aligned and Non-Aligned Equality ####
    #################################################
    assert torch.allclose(wass_ref, wass_pred)

    #################################################
    ############ Test 2: Slicing Ability ############
    #################################################
    n_mb1, n_mb2 = 5, n_mb - 5

    # Splitting the alignment indices
    i_algntrg1 = i_algntrg[..., :n_mb1]
    assert i_algntrg1.shape == (n_seeds, n_slc, n_mb1)

    i_algntrg2 = i_algntrg[..., n_mb1:]
    assert i_algntrg2.shape == (n_seeds, n_slc, n_mb2)

    # Getting the target aligned splits
    trg_pntsalgnd1 = torch.take_along_dim(trg_pnts, i_algntrg1, dim=-1)
    assert trg_pntsalgnd1.shape == (n_seeds, n_slc, n_mb1)

    trg_pntsalgnd2 = torch.take_along_dim(trg_pnts, i_algntrg2, dim=-1)
    assert trg_pntsalgnd2.shape == (n_seeds, n_slc, n_mb2)

    # Getting the splitted wass predictions
    wass_pred1 = (src_pnts[..., :n_mb1] - trg_pntsalgnd1).abs().square().sum(dim=-1)
    assert wass_pred1.shape == (n_seeds, n_slc)

    wass_pred2 = (src_pnts[..., n_mb1:] - trg_pntsalgnd2).abs().square().sum(dim=-1)
    assert wass_pred2.shape == (n_seeds, n_slc)

    assert torch.allclose(wass_ref, wass_pred1 + wass_pred2)


def torch_slice(tensor, dim, slc):
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None), ) + (slc, )]


def torch_cdf(input, qntls, dim, frame='domain', domain=None):
    """
    Computes the CDF of the input values with respect to a quantiles tensor.

    Parameters
    ----------
    input: (torch.tensor) the input tensor, where the mini-batch dimension is `dim`.

    qntls: (torch.tensor) the quantiles tensor, where the quantile dimension is `dim`.

    dim: (int) the cdf calculation dimension.

    frame: (str) either 'domain' or 'data'.

        * when `frame == 'domain'`, the output is guaranteed to be between [0, 1].

        * when `frame == 'data'`, the output is between [-1/(n-1), 1+1/(n-1)], where 
            `n` is the number of input quantiles.

    domain: (None | tuple) a `(min, max)` domain bounds tuple. When specified, it 
        will be manually conctenated to the quantiles tensor as an extreme quantile.

    Returns
    -------
    cdfs: (torch.tensor) the calculated cdf tensor.
    """
    cdf_maxerr = 1e-5    
    assert frame in ('data', 'domain')

    if domain is None:
        qntls2, n_qnts2 = qntls, qntls.shape[dim]
        assert qntls2.shape[dim] == n_qnts2
    else:
        assert len(domain) == 2
        domain_min, domain_max = min(domain), max(domain)
        n_qnts2 = qntls.shape[dim] + 2

        min_qntlval = torch_slice(qntls, dim=dim, slc=slice(0,     1))
        assert (min_qntlval.min() >= domain_min).all(), dedent(f'''
            The quantiles are outside the specified domain range (i.e., 
            the lower domain bound is not low enough):
                min quantile value: {min_qntlval}
                domain min: {domain_min}''')
        max_qntlval = torch_slice(qntls, dim=dim, slc=slice(-1, None))
        assert (max_qntlval.max() <= domain_max).all(), dedent(f'''
            The quantiles are outside the specified domain range (i.e., 
            the higher domain bound is not high enough):
                max quantile value: {min_qntlval}
                domain max: {domain_max}''')
        
        shp1 = list(qntls.shape)
        shp1[dim] = 1

        col_domainmin = torch.full(shp1, domain_min, device=qntls.device, dtype=qntls.dtype)
        col_domainmax = torch.full(shp1, domain_max, device=qntls.device, dtype=qntls.dtype)

        qntls2 = torch.cat([col_domainmin, qntls, col_domainmax], dim=dim)
        assert qntls2.shape[dim] == n_qnts2
        
        # Making sure the quantiles are sorted
        qntlsl = torch_slice(qntls2, dim, slice(0,   -1))
        qntlsr = torch_slice(qntls2, dim, slice(1, None))
        assert (qntlsr >= qntlsl).all(), 'The quantiles are not sorted'

    min_qntlval = torch_slice(qntls2, dim=dim, slc=slice(0,     1))
    max_qntlval = torch_slice(qntls2, dim=dim, slc=slice(-1, None))

    # Note: `torch.searchsorted` is currently limited in two ways:
    #    1. The search dimension should be the rightmost dimension
    #    2. The sorted sequence and the values should have identical dimensions up to the last one.
    # To fix these, we will have to do appropriate transposition and expansion calls.

    # Moving the `n_mb` dimension to be the rightmost dimension.
    srch_values = input.unsqueeze(-1).transpose(dim, -1).contiguous()
    # Making sure the sorted sequence is broadcasted to have the same shape as the search values.
    srch_srtdsq = qntls2.unsqueeze(-1).transpose(dim, -1).expand(
        *srch_values.shape[:-1], n_qnts2).contiguous()
    # Making the `torch.searchsorted` call
    srch_idxs1 = torch.searchsorted(srch_srtdsq, srch_values, right=True)
    # Moving the `n_mb` dimension from -1 to its original position.
    srch_idxs = srch_idxs1.transpose(dim, -1).squeeze(-1)
    assert srch_idxs.shape == input.shape

    # Finding the search bins low value
    srch_idxslo = (srch_idxs - 1).clamp(min=0)
    assert srch_idxslo.shape == input.shape
    srch_valslo = torch.take_along_dim(qntls2, srch_idxslo, dim=dim)
    assert srch_valslo.shape == input.shape
    # Finding the search bins high value
    srch_idxshi = srch_idxs.clamp(max=n_qnts2 - 1)
    assert srch_idxshi.shape == input.shape
    srch_valshi = torch.take_along_dim(qntls2, srch_idxshi, dim=dim)
    assert srch_valshi.shape == input.shape

    # If the domain min assertion fails, the second assertion can also fail.
    assert (input >= min_qntlval).all(), dedent(f'''
        The forward input data does not lie within the cdf domain:
            input min = {input[input < min_qntlval]}
            domain min = {min_qntlval[input < min_qntlval]}''')
    assert (input >= srch_valslo).all()
    
    # If the domain max assertion fails, the second assertion can also fail.
    assert (input <= max_qntlval).all(), dedent(f'''
        The forward input data does not lie within the cdf domain:
            input max = {input[input > max_qntlval]}
            domain max = {max_qntlval[input > max_qntlval]}''')
    assert (input <= srch_valshi).all()

    srch_idxshilodiff = (srch_idxshi - srch_idxslo)
    assert (srch_idxshilodiff <= 1).all()
    assert (srch_idxshilodiff >= 0).all()

    alpha1 = (input - srch_valslo) / (srch_valshi - srch_valslo)
    assert alpha1.shape == input.shape
    
    alpha2 = torch.where(srch_valshi == srch_valslo, 0.0, alpha1)
    assert alpha2.shape == input.shape

    alpha3 = torch.where(srch_valshi.isposinf(), 0.0, alpha2)
    assert alpha3.shape == input.shape

    alpha4 = torch.where(srch_valslo.isneginf(), 1.0, alpha3)
    assert alpha4.shape == input.shape
    
    assert (alpha4 >= 0).all()
    # Since we used `right=True`, then `alpha2<1` must strictly hold.
    # However, due to numerical stability issues near the domain boundaries,
    # Values slightly smaller than one are getting replaced with one.
    # For this reason, we will allow the assertion to be `alpha2 <= 1` rather 
    # than `alpha2 < 1`. Also, there is the neg inf case to consider!
    assert (alpha4 <= 1).all()

    cdfs1 = (srch_idxs - 1).clamp(min=0) / (n_qnts2 - 1) + alpha4 / (n_qnts2 - 1)
    assert cdfs1.shape == input.shape
    assert (cdfs1 <= (1 + cdf_maxerr)).all()
    assert (cdfs1 >= (0 - cdf_maxerr)).all()

    if frame == 'domain':
        cdfs = cdfs1
        assert cdfs.shape == input.shape
    elif frame == 'data':
        # Pushing the domain min and max out of the [0, 1] cdf region
        q0, q1 = 1 / (n_qnts2 - 1), (n_qnts2 - 2) / (n_qnts2 - 1)
        cdfs = (cdfs1 - q0) / (q1 - q0)
        assert cdfs.shape == input.shape
    else:
        raise ValueError(f'undefined frame={frame}')

    return cdfs


class EMA:
    def __init__(self, gamma, gamma_sq):
        self.gamma = gamma
        self.gamma_sq = gamma_sq
        self.ema = dict()
        self.ema_sq = dict()
    
    def __call__(self, key, val):
        gamma = self.gamma
        gamma_sq = self.gamma_sq
        ema, ema_sq = self.ema, self.ema_sq
        with torch.no_grad():
            val_ema = ema.get(key, val)
            val_ema = gamma * val_ema + (1-gamma) * val
            n_seeds = val_ema.numel()
            ema[key] = val_ema
            
            val_ema_sq = ema_sq.get(key, val**2)
            val_ema_sq = gamma_sq * val_ema_sq + (1-gamma_sq) * (val**2)
            ema_sq[key] = val_ema_sq

            val_popvar = (val_ema_sq - (val_ema**2)).detach().cpu().numpy()
            val_popvar[val_popvar < 0] = 0
            val_ema_std = np.sqrt(val_popvar) * np.sqrt((1-gamma)/(1+gamma))
            
            val_ema_mean = val_ema.mean()
            val_ema_std_mean = np.sqrt((val_ema_std**2).sum()) / n_seeds
        return val_ema_mean, val_ema_std_mean

###############################################################################
######################## Batch Random Number Generator ########################
###############################################################################

class BatchRNG:
    is_batch = True

    def __init__(self, shape, lib, device, dtype,
                 unif_cache_cols=1_000_000,
                 norm_cache_cols=5_000_000):
        assert lib in ('torch', 'numpy')

        self.lib = lib
        self.device = device

        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)
        self.reset_shape_attrs(shape)

        if self.lib == 'torch':
            self.rngs = [torch.Generator(device=self.device)
                         for _ in range(self.shape_prod)]
        else:
            self.rngs = [None for _ in range(self.shape_prod)]
        self.dtype = dtype

        self.unif_cache_cols = unif_cache_cols
        if self.lib == 'torch':
            self.unif_cache = torch.empty((self.shape_prod, self.unif_cache_cols),
                                          device=self.device, dtype=dtype)
        else:
            self.unif_cache = np.empty(
                (self.shape_prod, self.unif_cache_cols), dtype=dtype)
        # So that it would get refilled immediately
        self.unif_cache_col_idx = self.unif_cache_cols
        self.unif_cache_rng_states = None

        self.norm_cache_cols = norm_cache_cols
        if self.lib == 'torch':
            self.norm_cache = torch.empty((self.shape_prod, self.norm_cache_cols),
                                          device=self.device, dtype=dtype)
        else:
            self.norm_cache = np.empty(
                (self.shape_prod, self.norm_cache_cols), dtype=dtype)
        # So that it would get refilled immediately
        self.norm_cache_col_idx = self.norm_cache_cols
        self.norm_cache_rng_states = None
        self.was_seeded = False

        self.np_qr = np.linalg.qr
        try:
            ver = tuple(int(x) for x in np.__version__.split('.'))
            np_majorver, np_minorver, np_patchver, *_ = ver
            if not ((np_majorver >= 1) and (np_minorver >= 22)):
                def np_qr_fakebatch(a):
                    b = a.reshape(-1, a.shape[-2], a.shape[-1])
                    q, r = list(zip(*[np.linalg.qr(x) for x in b]))
                    qs = np.stack(q, axis=0)
                    rs = np.stack(r, axis=0)
                    qout = qs.reshape(
                        *a.shape[:-2], qs.shape[-2], qs.shape[-1])
                    rout = rs.reshape(
                        *a.shape[:-2], rs.shape[-2], rs.shape[-1])
                    return qout, rout
                self.np_qr = np_qr_fakebatch
        except Exception:
            pass

    def reset_shape_attrs(self, shape):
        self.shape = shape
        self.shape_prod = int(np.prod(self.shape))
        self.shape_len = len(self.shape)

    def seed(self, seed_arr):
        # Collecting the rng_states after seeding
        assert isinstance(seed_arr, np.ndarray)
        assert len(self.rngs) == seed_arr.size
        flat_seed_arr = seed_arr.copy().reshape(-1)
        if self.lib == 'torch':
            np_random = np.random.RandomState(seed=0)
            for seed, rng in zip(flat_seed_arr, self.rngs):
                np_random.seed(seed)
                balanced_32bit_seed = np_random.randint(
                    0, 2**31-1, dtype=np.int32)
                rng.manual_seed(int(balanced_32bit_seed))
        else:
            self.rngs = [np.random.RandomState(
                seed=seed) for seed in flat_seed_arr]

        if self.unif_cache_col_idx < self.unif_cache_cols:
            self.refill_unif_cache()
            # The cache has been used before, so in order to be able to
            # concat this sampler with the non-reseeded sampler, we should not
            # change the self.unif_cache_cols.

            # Note: We should not refill the uniform cache if the model
            # has not been initialized. This is done to keep the backward
            # compatibility and reproducibility properties with the old scripts.
            # Otherwise, the order of random samplings will change. Remember that
            # the old script first uses dirichlet and priors, and then refills
            # the unif/norm cache. In order to be similar, we should avoid
            # refilling the cache upon the first .seed() call
        if self.norm_cache_col_idx < self.norm_cache_cols:
            self.refill_norm_cache()
        self.was_seeded = True

    def get_state(self):
        state_dict = dict(unif_cache_rng_states=self.unif_cache_rng_states,
                          norm_cache_rng_states=self.norm_cache_rng_states,
                          norm_cache_col_idx=self.norm_cache_col_idx,
                          unif_cache_col_idx=self.unif_cache_col_idx,
                          rng_states=self.get_rng_states(self.rngs))
        return state_dict

    def set_state(self, state_dict):
        unif_cache_rng_states = state_dict['unif_cache_rng_states']
        norm_cache_rng_states = state_dict['norm_cache_rng_states']
        norm_cache_col_idx = state_dict['norm_cache_col_idx']
        unif_cache_col_idx = state_dict['unif_cache_col_idx']
        rng_states = state_dict['rng_states']

        if unif_cache_rng_states is not None:
            self.set_rng_states(unif_cache_rng_states, self.rngs)
            self.refill_unif_cache()
            self.unif_cache_col_idx = unif_cache_col_idx
        else:
            self.unif_cache_col_idx = self.unif_cache_cols
            self.unif_cache_rng_states = None

        if norm_cache_rng_states is not None:
            self.set_rng_states(norm_cache_rng_states, self.rngs)
            self.refill_norm_cache()
            self.norm_cache_col_idx = norm_cache_col_idx
        else:
            self.norm_cache_col_idx = self.norm_cache_cols
            self.norm_cache_rng_states = None

        self.set_rng_states(rng_states, self.rngs)
        self.was_seeded = True

    def get_rngs(self):
        return self.rngs

    def set_rngs(self, rngs, shape):
        assert isinstance(rngs, list)
        self.reset_shape_attrs(shape)
        self.rngs = rngs
        assert len(
            self.rngs) == self.shape_prod, f'{len(self.rngs)} != {self.shape_prod}'

    def get_rng_states(self, rngs):
        """
        getting state in ByteTensor
        """
        rng_states = []
        for i, rng in enumerate(rngs):
            rng_state = rng.get_state()
            if self.lib == 'torch':
                rng_state = rng_state.detach().clone()
            rng_states.append(rng_state)
        return rng_states

    def set_rng_states(self, rng_states, rngs):
        """
        rng_states should be ByteTensor (RNG state must be a torch.ByteTensor)
        """
        assert isinstance(
            rng_states, list), f'{type(rng_states)}, {rng_states}'
        for i, rng in enumerate(rngs):
            rs = rng_states[i]
            if self.lib == 'torch':
                rs = rs.cpu()
            rng.set_state(rs)
        self.was_seeded = True

    def __call__(self, gen, sample_shape):
        assert self.lib == 'torch'
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, rng in enumerate(self.rngs):
            rng_state = rng.get_state()
            rng_state = rng_state.detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(
                self.device).detach().clone())
        rv = torch.stack(random_vars, dim=0).reshape(*sample_shape)
        return rv

    def dirichlet(self, gen_list, sample_shape):
        assert self.lib == 'torch'
        sample_shape_rightmost = sample_shape[self.shape_len:]
        random_vars = []
        for i, (gen_, rng) in enumerate(zip(gen_list, self.rngs)):
            rng_state = rng.get_state().detach().clone()
            torch.cuda.set_rng_state(rng_state, self.device)
            random_vars.append(gen_.sample(sample_shape_rightmost))
            rng.set_state(torch.cuda.get_rng_state(
                self.device).detach().clone())

        rv = torch.stack(random_vars, dim=0)
        rv = rv.reshape(*self.shape, *rv.shape[1:])
        return rv

    def refill_unif_cache(self, unif_cache=None, strtcol=0):
        assert self.was_seeded
        if (self.lib == 'torch') and (detect_fake_mode() is not None) and (self.unif_cache_cols > 0):
            warnings.warn(dedent(f'''
                Attempting to refill the uniform cache in fake mode is dangerous. I will try 
                to unset the fake mode temporarily for refilling the unif cache.'''), 
                category=RuntimeWarning)
        with unset_fake_temporarily():
            unif_cache = unif_cache if unif_cache is not None else self.unif_cache
            self.unif_cache_rng_states = self.get_rng_states(self.rngs)
            if self.lib == 'torch':
                for row, rng in enumerate(self.rngs):
                    unif_cache[row, strtcol:].uniform_(generator=rng)
            else:
                for row, rng in enumerate(self.rngs):
                    unif_cache[row, strtcol:] = rng.rand(self.unif_cache_cols - strtcol)

    def refill_norm_cache(self, norm_cache=None, strtcol=0):
        assert self.was_seeded
        if (self.lib == 'torch') and (detect_fake_mode() is not None) and (self.norm_cache_cols > 0):
            warnings.warn(dedent(f'''
                Attempting to refill the normal cache in fake mode is dangerous. I will try 
                to unset the fake mode temporarily for refilling the normal cache.'''), 
                category=RuntimeWarning)

        with unset_fake_temporarily():
            norm_cache = norm_cache if norm_cache is not None else self.norm_cache
            self.norm_cache_rng_states = self.get_rng_states(self.rngs)
            if self.lib == 'torch':
                for row, rng in enumerate(self.rngs):
                    norm_cache[row, strtcol:].normal_(generator=rng)
            else:
                for row, rng in enumerate(self.rngs):
                    norm_cache[row, strtcol:] = rng.randn(self.norm_cache_cols - strtcol)

    @torch.no_grad()
    def rand(self, *sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        assert sample_shape_tuple[:self.shape_len] == self.shape

        sample_shape_rightmost = sample_shape[self.shape_len:]
        cols = np.prod(sample_shape_rightmost)
        if cols > self.unif_cache_cols:
            if self.lib == 'torch':
                samples = torch.empty((self.shape_prod, cols),
                    device=self.device, dtype=self.dtype)
            else:
                samples = np.empty((self.shape_prod, cols), 
                    dtype=self.dtype)

            # Transferring the remaining columns from the cache
            n_rmngcols = self.unif_cache_cols - self.unif_cache_col_idx
            copy_source = self.unif_cache[:, self.unif_cache_col_idx:]
            if self.lib == 'torch':
                samples[:, :n_rmngcols].copy_(copy_source)
            else:
                samples[:, :n_rmngcols] = copy_source
            
            # Filling out the rest of the uninitialized columns
            self.refill_unif_cache(samples, n_rmngcols)
            samples = samples.reshape(*sample_shape)

            # Re-filling the internal cache
            self.refill_unif_cache(self.unif_cache, 0)
            self.unif_cache_col_idx = 0
        else:
            if self.unif_cache_col_idx + cols > self.unif_cache_cols:
                n_rmngcols = self.unif_cache_cols - self.unif_cache_col_idx
                copy_source = self.unif_cache[:, self.unif_cache_col_idx:]
                if self.lib == 'torch':
                    self.unif_cache[:, :n_rmngcols].copy_(copy_source.clone())
                else:
                    self.unif_cache[:, :n_rmngcols] = copy_source
                self.refill_unif_cache(self.unif_cache, n_rmngcols)
                self.unif_cache_col_idx = 0
            samples_cache = self.unif_cache[:, self.unif_cache_col_idx: (
                self.unif_cache_col_idx + cols)]
            samples = samples_cache.clone().reshape(*sample_shape)
            self.unif_cache_col_idx += cols
        return samples

    @torch.no_grad()
    def randn(self, *sample_shape):
        sample_shape_tuple = tuple(sample_shape)
        assert sample_shape_tuple[:self.shape_len] == self.shape

        sample_shape_rightmost = sample_shape[self.shape_len:]
        cols = np.prod(sample_shape_rightmost)
        if cols > self.norm_cache_cols:
            if self.lib == 'torch':
                samples = torch.empty((self.shape_prod, cols),
                    device=self.device, dtype=self.dtype)
            else:
                samples = np.empty((self.shape_prod, cols), 
                    dtype=self.dtype)

            # Transferring the remaining columns from the cache
            n_rmngcols = self.norm_cache_cols - self.norm_cache_col_idx
            copy_source = self.norm_cache[:, self.norm_cache_col_idx:]
            if self.lib == 'torch':
                samples[:, :n_rmngcols].copy_(copy_source)
            else:
                samples[:, :n_rmngcols] = copy_source
            
            # Filling out the rest of the uninitialized columns
            self.refill_norm_cache(samples, n_rmngcols)
            samples = samples.reshape(*sample_shape)

            # Re-filling the internal cache
            self.refill_norm_cache(self.norm_cache, 0)
            self.norm_cache_col_idx = 0
        else:
            if self.norm_cache_col_idx + cols > self.norm_cache_cols:
                n_rmngcols = self.norm_cache_cols - self.norm_cache_col_idx

                self.norm_cache[:, :n_rmngcols] = self.norm_cache[:, self.norm_cache_col_idx:]
                copy_source = self.norm_cache[:, self.norm_cache_col_idx:]
                if self.lib == 'torch':
                    self.norm_cache[:, :n_rmngcols].copy_(copy_source)
                else:
                    self.norm_cache[:, :n_rmngcols] = copy_source
                    
                self.refill_norm_cache(self.norm_cache, n_rmngcols)
                self.norm_cache_col_idx = 0
            samples_cache = self.norm_cache[:, self.norm_cache_col_idx: (
                self.norm_cache_col_idx + cols)]
            samples = samples_cache.clone().reshape(*sample_shape)
            self.norm_cache_col_idx += cols
        
        # assert not (samples == 0).any()
        return samples

    def so_n(self, *sample_shape):        
        sample_shape_tuple = tuple(sample_shape)

        assert sample_shape_tuple[-2] == sample_shape_tuple[-1]
        n_bch, d = self.shape_prod, sample_shape_tuple[-1]
        sample_numel = np.prod(sample_shape_tuple)
        n_v = sample_numel // (self.shape_prod * d * d)
        assert sample_numel == (n_bch * n_v * d * d)
        qr_factorizer = torch_qr_eff if self.lib == 'torch' else self.np_qr
        diagnalizer = torch.diagonal if self.lib == 'torch' else np.diagonal
        signer = torch.sign if self.lib == 'torch' else np.sign

        norms = self.normal((n_bch, n_v, d, d))
        assert norms.shape == (n_bch, n_v, d, d)
        q, r = qr_factorizer(norms)
        assert q.shape == (n_bch, n_v, d, d)
        assert r.shape == (n_bch, n_v, d, d)
        r_diag = diagnalizer(r, 0, -2, -1)
        assert r_diag.shape == (n_bch, n_v, d)
        r_diag_sign = signer(r_diag)
        assert r_diag_sign.shape == (n_bch, n_v, d)
        q_signed = q * r_diag_sign.reshape(n_bch, n_v, 1, d)
        assert q_signed.shape == (n_bch, n_v, d, d)
        so_n = q_signed.reshape(*sample_shape_tuple)
        assert so_n.shape == sample_shape_tuple
        
        return so_n

    @classmethod
    def Merge(cls, sampler1, sampler2):
        assert sampler1.shape_len == sampler2.shape_len == 1

        device = sampler1.device
        dtype = sampler1.dtype
        chain_size = (sampler1.shape[0]+sampler2.shape[0],)

        state_dict1, state_dict2 = sampler1.get_state(), sampler2.get_state()

        merged_state_dict = dict()
        for key in state_dict1:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states', 'rng_states'):
                # saba modified
                if (state_dict1[key] is None) and (state_dict2[key] is None):
                    merged_state_dict[key] = None
                elif (state_dict1[key] is None) or (state_dict2[key] is None):
                    raise ValueError(f"{key} with None occurance")
                else:
                    merged_state_dict[key] = state_dict1[key] + \
                        state_dict2[key]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                assert state_dict1[key] == state_dict2[key]
                merged_state_dict[key] = state_dict1[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size, dtype)
        sampler.set_state(merged_state_dict)
        return sampler

    @classmethod
    def Subset(cls, sampler, inds):
        assert sampler.shape_len == 1

        device = sampler.device
        dtype = sampler.dtype
        chain_size_sub = (len(inds),)

        state_dict = sampler.get_state()

        sub_state_dict = dict()
        for key in state_dict:
            if key in ('unif_cache_rng_states', 'norm_cache_rng_states',
                       'rng_states'):
                sub_state_dict[key] = [state_dict[key][ind] for ind in inds]
            elif key in ('norm_cache_col_idx', 'unif_cache_col_idx'):
                sub_state_dict[key] = state_dict[key]
            else:
                raise ValueError(f'Unknown rule for {key}')

        sampler = cls(device, chain_size_sub, dtype)
        sampler.set_state(sub_state_dict)
        return sampler


###############################################################################
############################# Batch Architectures #############################
###############################################################################

class NNVersionError:
    def __init__(self, layer):
        self.layer = layer
    
    def __call__(self, *args, **kwargs):
        raise ValueError(dedent(f'''
            Your architecture requires an "nn.{self.layer}" module. 
            However, your pytorch version ({torch.__version__}) is 
            too old, and does not have that implemented. Please 
            upgrade your pytorch, or wait until they implement 
            this module.'''))


act_modulenamescap = ['ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 
    'Hardswish', 'LeakyReLU', 'LogSigmoid', 'PReLU', 'ReLU', 'ReLU6', 
    'RReLU', 'SELU', 'CELU', 'GELU', 'Sigmoid', 'SiLU', 'Mish', 'Softplus', 
    'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold', 'GLU', 
    'Identity']

act2fn = {**{act.lower(): getattr(nn, act, NNVersionError(act)) 
             for act in act_modulenamescap},
          **{act: getattr(nn, act, NNVersionError(act)) 
             for act in act_modulenamescap}}

class BatchFCN(nn.Module):
    """
    Batched Fully-Connected Network for Function Approximation.
    """
    def __init__(self, dims: list, act: str | Callable, 
        bias: bool, shape: tuple, device: any, 
        dtype: any, rng: any, do_actout: bool = False, 
        mb_order: int = 1, norm: str | None = None, 
        do_normout: bool | None = None):
        """
        Creates and initializes a set of fully connected layers.

        :param dims: (list) The list of latent dimensions in the
            fully-connected network. The list should include the
            input and output dimensions.

            The following example defines a 3-layer network that
            takes a 4-dimensional input and outputs a 2-dimensional
            tensor.
                `dims = [4, 64, 64, 2]`

        :param act: (callable) The activation function of the neural
            network.

        :param bias: (bool) Whether to include bias in the neural
            parameters or not.

        :param shape: (tuple) The batched shape of the fully-connected
            network.

        :param device: (torch.device) The pytorch device for creating,
            computing, and storing the tensors.

        :param dtype: (torch.dtype) The pytorch data type for floating
            point tensors.

        :param rng: (BatchRNG) The batched random number generator for
            initializing the neural networks.
        """
        super().__init__()

        assert len(dims) >= 2, 'The first and last dimension'
        assert mb_order in (0, 1)
        w, b, bn = [], [], []
        do_normout = do_actout if do_normout is None else do_normout
        for i_layer, (indim, odim) in enumerate(zip(dims[:-1], dims[1:])):
            layer_w, layer_b = self.init_linear(shape, indim, odim, rng, None, None)
            w.append(nn.Parameter(layer_w) if indim is not None else layer_w)

            # To make sure the RNG sequences remain synced, we will sample the bias
            # random numbers even if we are going to throw them away.
            layer_b_ = nn.Parameter(layer_b) if indim is not None else layer_b
            b.append(None if not bias else layer_b_)
            
            is_hdnlayer = (i_layer < (len(dims) - 2))
            has_bn1 = (is_hdnlayer or do_normout)
            has_bn2 = (norm not in ('none', None))
            has_bn = has_bn2 and has_bn1

            bn.append(None if not(has_bn) else 
                BatchNorm(norm=norm, n_spdims=1, n_channels=odim, 
                shape=shape, x_spdims=[1], norm_kwargs=None, 
                mb_order=1, device=device, dtype=dtype))
        
        activation = act2fn[act]() if isinstance(act, str) else act
        assert callable(activation)

        self.weights = nn.ParameterList(w)
        self.biases = b if not bias else nn.ParameterList(b)
        self.norms = bn if norm in ('none', None) else nn.ModuleList(bn)
        self.shape = shape
        self.ndim = len(shape)
        self.dims = dims
        self.act = activation
        self.do_actout = do_actout
        self.do_normout = do_normout
        self.device = device
        self.dtype = dtype
        self.rng = rng
        self.mb_order = mb_order
        self.n_layers = len(w)
        self.normtype = norm
        assert rng.device == device
        assert rng.dtype == dtype

    def forward(self, x):
        """
        Applies the forward propagation over the neural network

        :param x: (torch.tensor) The input tensor which satisfies
            the following shape:

            `assert x.shape == (*self.shape, ..., self.dims[0])`

            Note that `self.dims[0]` is the input dimension to
            the fully connected network.

        :return: (torch.tensor) The output tensor which satisfies
            the following shape:

            `assert z.shape == (*self.shape, ..., self.dims[-1])`

            Note that `self.dims[-1]` is the output dimension of
            the fully connected network.
        """
        if self.dims[0] is None:
            self.dims[0] = x.shape[-1]
            self.weights[0], self.biases[0] = self.init_linear(self.shape, 
                self.dims[0], self.dims[1], self.rng, self.weights[0], self.biases[0])

        self_size = int(np.prod(self.shape))
        if self.mb_order == 1:
            x_bchshape = x.shape[len(self.shape):-1]
            x_bchsize = int(np.prod(x_bchshape))
            assert x.shape == (*self.shape, *x_bchshape, self.dims[0])
            z = x.reshape(*self.shape, x_bchsize, self.dims[0])
        elif self.mb_order == 0:
            x_bchshape = x.shape[:-(len(self.shape) + 1)]
            x_bchsize = int(np.prod(x_bchshape))
            assert x.shape == (*x_bchshape, *self.shape, self.dims[0])
            
            z2 = x.reshape(x_bchsize, self_size, self.dims[0])
            assert z2.shape == (x_bchsize, self_size, self.dims[0])
            
            z3 = z2.transpose(0, 1)
            assert z3.shape == (self_size, x_bchsize, self.dims[0])

            z = z3.reshape(*self.shape, x_bchsize, self.dims[0])
            assert z.shape == (*self.shape, x_bchsize, self.dims[0])
        else:
            raise ValueError(f'undefind self.mb_order = {self.mb_order}')

        n_actcalls, n_bncalls = 0, 0
        for i, (w, b, bn) in enumerate(zip(self.weights, self.biases, self.norms)):
            z = z.matmul(w)
            if b is not None:
                z = z + b
            if bn is not None:
                z = bn(z.unsqueeze(-1)).squeeze(-1)
                n_bncalls += 1
            if (i <= (len(self.weights) - 2)) or self.do_actout:
                z = self.act(z)
                n_actcalls += 1
        
        if self.do_actout: 
            assert n_actcalls == self.n_layers
        else: 
            assert n_actcalls == (self.n_layers - 1)
        
        if self.normtype in ('none', None):
            assert n_bncalls == 0
        elif self.do_normout:
            assert n_bncalls == self.n_layers
        else: 
            assert n_bncalls == (self.n_layers - 1)

        if self.mb_order == 1:
            out = z.reshape(*self.shape, *x_bchshape, z.shape[-1])
            assert out.shape == (*self.shape, *x_bchshape, z.shape[-1])
        elif self.mb_order == 0:
            z4 = z.reshape(self_size, x_bchsize, z.shape[-1])
            assert z4.shape == (self_size, x_bchsize, z.shape[-1])

            z5 = z4.transpose(0, 1)
            assert z5.shape == (x_bchsize, self_size, z.shape[-1])

            out = z5.reshape(*x_bchshape, *self.shape, z.shape[-1])
            assert out.shape == (*x_bchshape, *self.shape, z.shape[-1])
        else:
            raise ValueError(f'undefind self.mb_order = {self.mb_order}')

        return out

    def init_linear(self, shape, in_features, out_features, rng, w, b):
        """
        Instantiating a set of random linear layer weights and biases.
            The distribution of weights and biases follow Pytorch's default:
            https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        :param shape: (tuple) The batch shape of the neural module.

        :param in_features: (int) The input feature dimension.

        :param out_features: (int) The output feature dimension.

        :param rng: (BatchRNG) The batched random number generator.

        :return: a pair of weight and bias tensor.
        """
        assert (w is None) == (b is None)
        assert out_features is not None

        if in_features is None:
            assert w is None
            assert b is None
            w_param = nn.UninitializedParameter(device=rng.device, 
                dtype=rng.dtype)
            b_param = nn.UninitializedParameter(device=rng.device, 
                dtype=rng.dtype)
        else:
            k = 1. / np.sqrt(in_features).item()
            with torch.no_grad():
                w_unit = rng.rand(*shape, in_features, out_features)
                b_unit = rng.rand(*shape,           1, out_features)
                w_tensor = w_unit * (2 * k) - k                
                b_tensor = b_unit * (2 * k) - k
            if w is not None:
                assert isinstance(w, nn.UninitializedParameter)
                assert isinstance(b, nn.UninitializedParameter)
                assert in_features is not None
                assert b is not None
                w.materialize((*shape, in_features, out_features))
                b.materialize((*shape,           1, out_features))
                w.data.copy_(w_tensor)
                b.data.copy_(b_tensor)
                w_param, b_param = w, b
            else:
                assert b is None
                w_param = nn.Parameter(w_tensor)
                b_param = nn.Parameter(b_tensor)

        return w_param, b_param

    def state_dict(self, *args, **kwargs):
        sdict = super().state_dict(*args, **kwargs)
        return get_state_dict_seeded(sdict, self.shape)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        sdict = get_state_dict_unseeded(state_dict, self.shape)
        super().load_state_dict(sdict, *args, **kwargs)


def int2tup2d(hp: int | str | list | tuple, n_layers: int, n_convdims: int):
    '''
    Converts an integer or a list of `hp` values (such as the convolutional 
    kernel size, padding, etc.) to a 2d last of the shape `(n_layers, n_convdims)`.

    Args:
        hp (int | str | list | tuple): 

            Example:
            ```
                # This means k=3 for all layers and height/width/depth dimensions
                hp = 3

                # This means k=1 for height and k=3 for width. All layers use the same.
                hp = [1, 3]

                # This means that 
                #   1. the 1st layer has k=1 for height and k=3 for width,
                #   2. the 2nd layer has k=2 for height and k=4 for width, and
                #   3. the 3rd layer has k=5 for height and k=6 for width.
                hp = [[1, 3], [2, 4], [5, 6]]
            ```

        n_layers (int): The number of layers

        n_convdims (int): The number of dims

    Returns:
        out (list[list[int]]): a 2d list resulting from broadcasting the `hp` values 
            into the `(n_layers, n_convdims)` shape.
    '''
    if isinstance(hp, (str, int, float)):
        col = [hp for _ in range(n_layers)]
    elif isinstance(hp, (list, tuple)):
        col = list(hp)
    else:
        raise ValueError(f'undefined type ({type(hp)}) for hp="{hp}"')
    assert isinstance(col, list)
    assert len(col) == n_layers

    out = []
    for val in col:
        if isinstance(val, (int, float)):
            row = [val for _ in range(n_convdims)]
        elif isinstance(val, str):
            row = val
        elif isinstance(val, (list, tuple)):
            row = list(val)
        else:
            raise ValueError(f'undefined type ({type(val)}) for val="{val}"')

        assert isinstance(row, str) or (len(row) == n_convdims)
        out.append(row)
    return out


def get_xconvshapes(x_inspdims, n_convdims, channels, transpose_conv, padding, 
    dilation, kernel_size, stride, output_padding):
    
    # n_convdims = self.n_convdims
    # channels = self.channels
    # transpose_conv = self.transpose_conv
    # padding, dilation = self.padding, self.dilation
    # kernel_size, stride  = self.kernel_size, self.stride
    # output_padding = self.output_padding
    
    assert len(x_inspdims) == n_convdims
    x_inshape = (channels[0], *x_inspdims)

    x_shapes = []
    h_lastdims = x_inspdims
    for i_layer, h_channels in enumerate(channels[1:]):
        h_dims = []
        for i_dim, h_lastdim in enumerate(h_lastdims):
            if h_lastdim is None:
                h_dim = None
            else:
                h_padding = padding[i_layer][i_dim]
                assert isinstance(h_padding, (int, str)), f'h_padding = {h_padding}'
                h_dilation = dilation[i_layer][i_dim]
                assert isinstance(h_dilation, int)
                h_kernel_size = kernel_size[i_layer][i_dim]
                assert isinstance(h_kernel_size, int)
                h_stride = stride[i_layer][i_dim]
                assert isinstance(h_stride, int)
                h_output_padding = output_padding[i_layer][i_dim]
                assert isinstance(h_output_padding, int)
                if h_padding == 'same':
                    h_dim = h_lastdim
                    assert (np.array(h_stride) == 1).all()
                elif not transpose_conv:
                    h_dim_ = h_lastdim + 2 * h_padding - h_dilation * (h_kernel_size - 1) - 1
                    h_dim = int((h_dim_ / h_stride) + 1)
                else:
                    h_dim = ((h_lastdim - 1) * h_stride - 2 * h_padding +
                        h_dilation * (h_kernel_size - 1) + h_output_padding + 1)
            h_dims.append(h_dim)
        x_shapes.append([h_channels] + h_dims)
        h_lastdims = h_dims
    
    return x_inshape, x_shapes


def get_state_dict_seeded(sdict, shape):
    n_seeds = math.prod(shape)
    ndim = len(shape)

    sdict_seeded = dict()
    for name, param in sdict.items():
        p_shape = param.shape
        n_pseeds = p_shape[0] if len(p_shape) > 0 else None
        if (param.numel() == 1) and name.endswith('.num_batches_tracked'):
            param_seeded = param.reshape(*([1] * ndim)).expand(*shape)
        elif ('convs.' in name) or ('norms.' in name) or ('bn_layer.' in name):
            assert n_pseeds is not None
            assert n_pseeds % n_seeds == 0
            # Note: I have verified that CNN layers do not require any transpositions 
            # between the `n_seeds` and `n_inplay` dimensions.
            n_inplay = n_pseeds // n_seeds
            param_seeded = param.unflatten(0, (*shape, n_inplay))
        elif p_shape[:ndim] == shape:
            param_seeded = param
        else:
            raise ValueError(dedent(f'''
                checkpointing parameter name 
                    "{name}"
                has an odd shape:
                    "{p_shape}"
                I dont have a rule to make it (n_seeds,*)-compatible.
            '''))
        assert param_seeded.shape[:ndim] == shape
        sdict_seeded[name] = param_seeded
    
    return sdict_seeded


def get_state_dict_unseeded(sdict_seeded, shape):
    sdict = dict()
    for name, param_seeded in sdict_seeded.items():
        assert param_seeded.shape[:len(shape)] == shape, dedent(f'''
            The "{name}" parameter does not have a conforming 
            shape to a seeded tensor:
                name: {name}
                model shape: {shape}
                param shape: {param.shape}''')
        
        if name.endswith('.num_batches_tracked'):
            param = param_seeded.ravel()[0]
            assert param_seeded == param.all()
        elif ('.convs.' in name) or ('.norms.' in name) or ('.bn_layer.' in name):
            param = param_seeded.flatten(0, len(shape))
        else:
            param = param_seeded
        
        sdict[name] = param
    
    return sdict


class BatchLayerNorm(nn.Module):
    '''
    Let's say `x.shape == (n_seeds, n_mb, n_chnls, n_height, n_width)` 
    and that we're talking about nn.LayerNorm2d.
    
    At training mode, layer normalization does the following:

    ```
    weights_ = torch.ones(n_seeds, 1, n_chnls, n_height, n_width) 
    biases_ = torch.zeros(n_seeds, 1, n_chnls, n_height, n_width)
    weights, biases = nn.Parameter(weights_), nn.Parameter(biases)
    
    mu = x.mean(dim=[-1, -2, -3], keepdim=True)
    assert mu.shape == (n_seeds, n_mb, 1, 1, 1)

    sigmasq = x.var(dim=[-1, -2, -3], keepdim=True, unbiased=True)
    assert sigmasq.shape == (n_seeds, n_mb, 1, 1, 1)

    y = weights * (x - mu) / (sigmasq + eps).sqrt() + biases
    ```

    The problem with adapting pytorch's layer normalization is that:

        1. The weights and biases should be indpendent across the 
            `(n_seeds, n_chnls, n_height, n_width)` dimensions.

        2. The `E[x]` and `Var[x]` (i.e., `mu` and `sigmasq`) should be 
            computed by reducing only the `(n_chnls, n_height, n_width)` 
            dimensions.
    
    Pytorch assumes the same set of weights/biases dimensions are used 
    for the mean and std inference. However, `n_seeds` should be invloved 
    in the weights/biases creation, but the `mu` and `sigmasq` inference 
    should not average over the `n_seeds` dimension.

    References:
        1. https://isaac-the-man.dev/posts/normalization-strategies/
        2. https://arxiv.org/pdf/1701.02096
        3. https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        4. https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html
        4. https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    '''

    def __init__(self, normalized_shape: list, eps: float = 1e-05, 
        elementwise_affine: bool = True, bias: bool = True, 
        shape: tuple = None, mb_order : int = 1,
        device: any = None, dtype: any = None):

        '''
        Args:

          normalized_shape (list): The dimensions over which the mean and std 
            will be calculated.

            Let's say `x.shape == (n_seeds, n_mb, n_chnls, n_height, n_width)` 
            and that we're talking about nn.LayerNorm2d.

            If we set `normalized_shape = [n_chnls, n_height, n_width]`, we 
            will have:

            ```
            weights_ = torch.ones(n_seeds, 1, n_chnls, n_height, n_width) 
            biases_ = torch.zeros(n_seeds, 1, n_chnls, n_height, n_width)
            weights, biases = nn.Parameter(weights_), nn.Parameter(biases)

            mu = x.mean(dim=[-1, -2, -3], keepdim=True)
            assert mu.shape == (n_seeds, n_mb, 1, 1, 1)

            sigmasq = x.var(dim=[-1, -2, -3], keepdim=True, unbiased=True)
            assert sigmasq.shape == (n_seeds, n_mb, 1, 1, 1)

            y = weights * (x - mu) / (sigmasq + eps).sqrt() + biases
            ```

          eps (float): a value added to the denominator for numerical stability. 
            Default: 1e-5

          elementwise_affine (bool): a boolean value that when set to `True`, this 
            module has learnable per-element affine parameters initialized to ones 
            (for weights) and zeros (for biases). Default: True.

          bias (bool): If set to False, the layer will not learn an additive bias 
            (only relevant if elementwise_affine is True). Default: True.

          shape (tuple): The model parallelization dimensions at the beginning 
            of all input and parameter tensors. Usually this is specified as 
            `(n_seeds,)`, where `n_seeds` is the number of independent models 
            being trained at the same time.

            Example:
            ```
              shape = (n_seeds,)
            ```
        
          mb_order (int): The mini-batch dimension index. should be either 0 or 1. This 
            determines where `n_mb` is located in the input's shape:
            ```
            if mb_order == 0:
                assert x.shape == (n_mb, n_seeds, n_channel, *x_spatial_dims)
            elif mb_order == 1:
                assert x.shape == (n_seeds, n_mb, n_channel, *x_spatial_dims)
            else:
                raise NotImplementedError
            ```
        '''
        super().__init__()

        assert shape is not None
        assert device is not None
        assert dtype is not None
        assert mb_order in (0, 1)

        assert len(shape) == 1, dedent(f'''
            The BatchLayerNorm module only support single-dimensional shapes
            for now. The following module shape has a non-unit length:
                shape = {shape}
        ''')
        n_seeds, = shape

        normdims = normalized_shape
        n_normdims = len(normdims)        
        be_lazy = any(d is None for d in normdims)
        lshape = (n_seeds, 1) if (mb_order == 1) else (1, n_seeds)

        if elementwise_affine and be_lazy:
            weights = nn.UninitializedParameter(device=device, dtype=dtype)
        elif elementwise_affine and (not be_lazy):
            weights_ = torch.ones(*lshape, *normdims, device=device, dtype=dtype)
            weights = nn.Parameter(weights_)
        else:
            weights = None
        
        if elementwise_affine and bias and be_lazy:
            biases = nn.UninitializedParameter(device=device, dtype=dtype)
        elif elementwise_affine and bias and (not be_lazy):
            biases_ = torch.zeros(*lshape, *normdims, device=device, dtype=dtype)
            biases = nn.Parameter(biases_)
        else:
            biases = None
        
        self.n_seeds = n_seeds
        self.normdims = normdims
        self.n_normdims = n_normdims
        self.mb_order = mb_order
        self.eps = eps
        self.weights = weights
        self.biases = biases

        self.needs_init = False
        if be_lazy:
            self.needs_init = True
    
    def init_wb(self):
        shape = ((self.n_seeds, 1, *self.normdims) 
            if (self.mb_order == 1) 
            else (1, self.n_seeds, *self.normdims))

        if self.weights is not None:
            assert isinstance(self.weights, nn.UninitializedParameter)
            self.weights.materialize(shape)
            nn.init.ones_(self.weights)
        
        if self.biases is not None:
            assert isinstance(self.biases, nn.UninitializedParameter)
            self.biases.materialize(shape)
            nn.init.zeros_(self.biases)

    def forward(self, x):
        for i_d, d_nm in enumerate(self.normdims):
            if d_nm is not None:
                assert x.shape[-self.n_normdims + i_d] == d_nm

        if self.needs_init:
            self.normdims = x.shape[-self.n_normdims:]
            self.init_wb()
            self.needs_init = False

        n_seeds, normdims, n_normdims = self.n_seeds, self.normdims, self.n_normdims
        n_mb = x.shape[1] if (self.mb_order == 1) else x.shape[0]
        
        leftshape = (n_seeds, n_mb) if (self.mb_order == 1) else (n_mb, n_seeds)
        assert x.shape == (*leftshape, *normdims)
        
        dims_rdction = list(range(-1, -n_normdims-1, -1))
        # Example: 
        #   n_normdims = 3
        #   dims_rdction == [-1, -2, -3]

        mu = x.mean(dim=dims_rdction, keepdim=True)
        assert mu.shape == (*leftshape, *([1] * n_normdims))

        sigmasq = x.var(dim=dims_rdction, keepdim=True, unbiased=True)
        assert sigmasq.shape == (*leftshape, *([1] * n_normdims))

        y = (x - mu) / (sigmasq + self.eps).sqrt()
        assert y.shape == (*leftshape, *normdims)

        if self.weights is not None:
            y = self.weights * y
            assert y.shape == (*leftshape, *normdims)
            
        if self.biases is not None:
            y = y + self.biases
            assert y.shape == (*leftshape, *normdims)
        
        return y

    def state_dict(self, *args, **kwargs):
        n_seeds = self.n_seeds
        sdict = super().state_dict(*args, **kwargs)
        return get_state_dict_seeded(sdict, n_seeds)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        sdict = get_state_dict_unseeded(state_dict, n_seeds)
        super().load_state_dict(sdict, *args, **kwargs)

class BatchNorm(nn.Module):
    def __init__(self, norm: str, n_spdims: int, n_channels: int | None, 
        shape: tuple, x_spdims: list | None = None, norm_kwargs: dict | None = None, 
        mb_order: int = 1, device: any = None, dtype: any = None):
        '''
        The super class for all kinds of normalization (batch, instance, and layer).
        This class supports model-level batching.

        Args:

          norm (str | None): The type of batch / layer / instance normalization.
            if specified as `None` or `'none'`, no batch normalization will be used.

            `assert norm in (None, 'none', 'batch', 'layer', 'instance')`

          n_channels (int | None): The number of input channels.
            If `None` is provided, we will use lazy initializations.

          n_spdims (int): The spatial dimensions. Set 1 for nn.*Norm1d, 
              2 for nn.*Norm2d, and 3 for nn.*Norm3d.

          shape (tuple): The model parallelization dimensions at the beginning 
            of all input and parameter tensors. Usually this is specified as 
            `(n_seeds,)`, where `n_seeds` is the number of independent models 
            being trained at the same time.

            Example:
            ```
              shape = (n_seeds,)
            ```

          x_inspdims (None | list): The "spatial" dimensions of the input. If provided, this 
            will be mostly used for shape assertions and should have the same length as 
            `n_spdims`. 

            Example:
            ```
              x_inspdims = (depth, height, width)
              assert (x_inspdims is None) or (len(x_inspdims) == n_spdims)
            ```
        
          mb_order (int): The mini-batch dimension index. should be either 0 or 1. This 
            determines where `n_mb` is located in the input's shape:
            ```
            if mb_order == 0:
                assert x.shape == (n_mb, n_seeds, n_channel, *x_spatial_dims)
            elif mb_order == 1:
                assert x.shape == (n_seeds, n_mb, n_channel, *x_spatial_dims)
            else:
                raise NotImplementedError
            ```
        '''

        super().__init__()
        assert len(shape) == 1, dedent(f'''
            The *Norm modules only support single-dimensional shapes
            for now. The following module shape has a non-unit length:
                shape = {shape}
        ''')
        n_seeds, = shape

        if norm in ('batch', 'instance'):
            assert n_spdims in (1, 2, 3), 'Only *Norm1d, *Norm2d, or *Norm3d layers exist.'
        assert norm in ('batch', 'instance', 'layer', 'none', None)
        be_lazy = n_channels is None
        norm_kwargs = dict() if norm_kwargs is None else norm_kwargs
        x_spdims = [None] * n_spdims if (x_spdims is None) else x_spdims
        assert len(x_spdims) == n_spdims

        # Getting the 1-, 2-, or 3-dimensional normalization modules from `torch.nn``
        nn_BatchNormXd = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}.get(n_spdims, None)
        nn_LazyBatchNormXd = {1: nn.LazyBatchNorm1d, 2: nn.LazyBatchNorm2d, 3: nn.LazyBatchNorm3d}.get(n_spdims, None)
        nn_InstanceNormXd = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}.get(n_spdims, None)
        nn_LazyInstanceNormXd = {1: nn.LazyInstanceNorm1d, 2: nn.LazyInstanceNorm2d, 3: nn.LazyInstanceNorm3d}.get(n_spdims, None)
        
        # Finding the right normalization layer module
        nn_BatchNormLayer = nn_LazyBatchNormXd if be_lazy else nn_BatchNormXd
        nn_InstanceNormLayer = nn_LazyInstanceNormXd if be_lazy else nn_InstanceNormXd

        ###############################################
        ##### Compiling the Norm Keyword Arguments ####
        ###############################################
        bn_kwargs = dict(device=device, dtype=dtype)
        if (norm in ('none', None)):
            pass
        elif (norm in ('batch', 'instance')) and be_lazy:
            pass
        elif (norm in ('batch', 'instance')) and (not be_lazy):
            bn_kwargs['num_features'] = n_seeds * n_channels
        elif norm == 'layer':
            bn_kwargs['shape'] = (n_seeds,)
            bn_kwargs['normalized_shape'] = [n_channels] + x_spdims
            bn_kwargs['mb_order'] = mb_order
        else:
            raise ValueError('undefined case')
        
        for key, val in norm_kwargs.items():
            assert key not in bn_kwargs
            bn_kwargs[key] = val
            
        ###############################################
        ##### Constructing the Normalization Layer ####
        ###############################################
        if norm in (None, 'none'):
            bn_layer = None
        elif norm == 'batch':
            bn_layer = nn_BatchNormLayer(**bn_kwargs)
        elif norm == 'instance':
            bn_layer = nn_InstanceNormLayer(**bn_kwargs)
        elif norm == 'layer':
            bn_layer = BatchLayerNorm(**bn_kwargs)
        else:
            raise ValueError(f'undefined normalization type "{norm}"')
        
        #########################################################
        ############# Storing the Class Variables ###############
        #########################################################
        self.normtype = norm
        self.n_spdims = n_spdims
        self.n_channels = n_channels
        self.shape = (n_seeds,)
        self.mb_order = mb_order
        self.n_seeds = n_seeds
        self.device = device
        self.dtype = dtype
        self.mb_order = mb_order
        self.bn_layer = bn_layer
        self.x_spdims = x_spdims
     
    def forward(self, x):
        norm = self.normtype
        bn_layer = self.bn_layer

        if norm in ('none', None):
            pass
        elif norm == 'layer':
            x = bn_layer(x)
        elif norm in ('batch', 'instance'):
            n_spdims = self.n_spdims
            n_seeds = self.n_seeds
            x_spdims = self.x_spdims
            n_channels = self.n_channels

            sp_dims = [h_dim if h_dim is not None else x_dim 
            for h_dim, x_dim in zip(x_spdims, x.shape[-n_spdims:])]

            if self.mb_order == 1:
                n_samps = x.shape[1]
                assert x.shape == (n_seeds, n_samps, n_channels, *sp_dims)
                x = x.transpose(0, 1)
            else:
                n_samps = x.shape[0]
            assert x.shape == (n_samps, n_seeds, n_channels, *sp_dims)

            x = x.reshape(n_samps, n_seeds * n_channels, *sp_dims)
            assert x.shape == (n_samps, n_seeds * n_channels, *sp_dims)

            x = bn_layer(x)                    
            assert x.shape == (n_samps, n_seeds * n_channels, *sp_dims)

            x = x.reshape(n_samps, n_seeds, n_channels, *sp_dims)
            assert x.shape == (n_samps, n_seeds, n_channels, *sp_dims)

            if self.mb_order == 1:
                x = x.transpose(0, 1)
                assert x.shape == (n_seeds, n_samps, n_channels, *sp_dims)
        else:
            raise ValueError(f'undefined normalization type "{norm}"')
        return x

    def state_dict(self, *args, **kwargs):
        sdict = super().state_dict(*args, **kwargs)
        return get_state_dict_seeded(sdict, self.shape)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        sdict = get_state_dict_unseeded(state_dict, self.shape)
        super().load_state_dict(sdict, *args, **kwargs)

class BatchCNN(nn.Module):
    def __init__(self, channels: list, n_convdims: int, kernel_size: int | list, 
        act: str | Callable, norm: None | str, do_actout: bool, shape: tuple, 
        rng: any, device: any, dtype: any, stride: int | list = 1, padding: int | list = 0, 
        dilation: int | list = 1, bias: bool | list = True, output_padding: int | list = 0, 
        padding_mode: str | list = 'zeros', norm_kwargs: dict | None = None,
        x_inspdims: None | list = None, transpose_conv: bool = False, mb_order: int = 1, 
        do_normout: bool | None = None):
        '''
        A fully fledged CNN implementation that supports:

          1. Model-level Parallelism with running many seeds

          2. Deep Network Implementations

          3. Activation and (batch/layer/instance) normalization layers.

          4. 1D, 2D, or, 3D Convolutions.

        Args:

          channels (list): The list of input, hidden, and output channels. 
            The first element is considered the number of input channels.
            The last element is considered the number of output channels.
            If the first element is None, we will use lazy initializations.

            Example:
              ```
              channels = [3, 64, 64, 32]

              # This is a lazy network with a single convolutional layer.
              channels = [None, 64]
              ```

          n_convdims (int): The convolutional dimensions. Set 1 for nn.*Conv1d, 
              2 for nn.*Conv2d, and 3 for nn.*Conv3d.

          kernel_size (int | list): The kernel sizes. Note that this should be 
            torch-broadcastable to `(n_layers, n_convdims)`.

            Example:
              ```
              channels = [3, 64, 32]
              n_convdims = 2

              # This means k=3 for all layers and height/width/depth dimensions
              kernel_size = 3

              # This means k=1 for height and k=3 for width. All layers use the same.
              kernel_size = [1, 3]

              # This means that 
              #   1. the 1st layer has k=1 for height and k=3 for width,
              #   2. the 2nd layer has k=2 for height and k=4 for width, and
              #   3. the 3rd layer has k=5 for height and k=6 for width.
              kernel_size = [[1, 3], [2, 4], [5, 6]]
              ```
          
          act (str | callable): The activation function.

          norm (str | None): The type of batch / layer / instance normalization.
            if specified as `None` or `'none'`, no batch normalization will be used.

            `assert norm in (None, 'none', 'batch', 'layer', 'instance')`

          do_actout (bool): Whether to activate the output of the network.

          do_normout (bool): Whether to *-normalize the output of the network.

          shape (tuple): The model parallelization dimensions at the beginning 
            of all input and parameter tensors. Usually this is specified as 
            `(n_seeds,)`, where `n_seeds` is the number of independent models 
            being trained at the same time.

            Example:
            ```
              shape = (n_seeds,)
            ```

          rng (BatchRNG): The batch random number generator. This should 
            have the same shape.

          stride (int | list): The convolutional stride. See `nn.Conv*d` for 
            more information. When provided a list, the same rules as `kernel_size` 
            will be applied.


          padding (int | list): The convolutional padding. See `nn.Conv*d` for 
            more information. When provided a list, the same rules as `kernel_size` 
            will be applied.

          dilation (int | list): The convolutional dilation. See `nn.Conv*d` for 
            more information. When provided a list, the same rules as `kernel_size` 
            will be applied.
          
          output_padding (int | list): The de-convolutional output_padding. See 
            `nn.ConvTranspose*d` for more information. When provided a list, the 
            same rules as `kernel_size` will be applied.

          padding_mode (str | list): The convolutional padding mode. See `nn.Conv*d` 
            for more information. When provided a list, the same rules as 
            `kernel_size` will be applied.

          
          x_inspdims (None | list): The "image" dimensions of the input. If provided, this 
            will be mostly used for shape assertions and should have the same length as 
            `n_convdims`. 

            Example:
            ```
              x_inspdims = (depth, height, width)
              assert (x_inspdims is None) or (len(x_inspdims) == n_convdims)
            ```

          transpose_conv (bool): Whether to use classical convolution (`True`) or 
            de-convolution (`False`) layers.
        
          mb_order (int): The mini-batch dimension index. should be either 0 or 1. This 
            determines where `n_mb` is located in the input's shape:
            ```
            if mb_order == 0:
                assert x.shape == (n_mb, n_seeds, n_channel, *x_spatial_dims)
            elif mb_order == 1:
                assert x.shape == (n_seeds, n_mb, n_channel, *x_spatial_dims)
            else:
                raise NotImplementedError
            ```

        TODO: 
          2. Handle general `shape` other than (n_seeds,)
        '''
        super().__init__()
        assert len(shape) == 1, dedent(f'''
            Conv/CNN modules only support single-dimensional shapes for now.
            The following module shape has a non-unit length:
                shape = {shape}
        ''')
        
        n_seeds, = shape
        
        assert rng.device == device
        assert rng.dtype == dtype

        assert n_convdims in (1, 2, 3), 'Only conv1d, conv2d, or conv3d layers exist.'
        in_channels = channels[0]
        be_lazy = in_channels is None
        do_normout = do_actout if do_normout is None else do_normout
        norm_kwargs = dict() if norm_kwargs is None else norm_kwargs

        # Getting the 1-, 2-, or 3-dimensional convolutional or normalization modules from `torch.nn``
        nn_BatchNormXd = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[n_convdims]
        nn_LazyBatchNormXd = {1: nn.LazyBatchNorm1d, 2: nn.LazyBatchNorm2d, 3: nn.LazyBatchNorm3d}[n_convdims]
        nn_InstanceNormXd = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[n_convdims]
        nn_LazyInstanceNormXd = {1: nn.LazyInstanceNorm1d, 2: nn.LazyInstanceNorm2d, 3: nn.LazyInstanceNorm3d}[n_convdims]
        nn_ConvXd = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[n_convdims]
        nn_LazyConvXd = {1: nn.LazyConv1d, 2: nn.LazyConv2d, 3: nn.LazyConv3d}[n_convdims]
        nn_ConvTransposeXd = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}[n_convdims]
        nn_LazyConvTransposeXd = {1: nn.LazyConvTranspose1d, 2: nn.LazyConvTranspose2d, 3: nn.LazyConvTranspose3d}[n_convdims]

        # Finding the right convolutional layer module
        if be_lazy and transpose_conv:
            nn_ConvLayer = nn_LazyConvTransposeXd
        elif be_lazy and (not transpose_conv):
            nn_ConvLayer = nn_LazyConvXd
        elif (not be_lazy) and transpose_conv:
            nn_ConvLayer = nn_ConvTransposeXd
        else:
            nn_ConvLayer = nn_ConvXd
        
        # Finding the right normalization layer module
        nn_BatchNormLayer = nn_LazyBatchNormXd if be_lazy else nn_BatchNormXd
        nn_InstanceNormLayer = nn_LazyInstanceNormXd if be_lazy else nn_InstanceNormXd

        #########################################################
        ################ Creating the CNN Layers ################
        #########################################################
        n_layers = len(channels) - 1 
        kernel_size = int2tup2d(kernel_size, n_layers=n_layers, n_convdims=n_convdims)
        stride = int2tup2d(stride, n_layers=n_layers, n_convdims=n_convdims)
        padding = int2tup2d(padding, n_layers=n_layers, n_convdims=n_convdims)
        dilation = int2tup2d(dilation, n_layers=n_layers, n_convdims=n_convdims)
        output_padding = int2tup2d(output_padding, n_layers=n_layers, n_convdims=n_convdims)
        padding_mode = int2tup2d(padding_mode, n_layers=n_layers, n_convdims=n_convdims)
        bias = [bias] * n_layers if isinstance(bias, bool) else bias
        assert len(bias) == n_layers
        assert isinstance(bias, (tuple, list))

        x_inspdims = [None] * n_convdims if (x_inspdims is None) else x_inspdims
        x_inshape, x_shapes = get_xconvshapes(x_inspdims, n_convdims, channels, transpose_conv, padding, 
            dilation, kernel_size, stride, output_padding)
        # Note: `x_shapes` and `x_inshape` do not include the `n_seeds, n_mb` part of the shape. 
        #       For example, with 2d convolution we can have
        #          x_inshape == (n_chnls, n_height, n_width)
        #          x_shapes ==  [(n_chnls, n_height, n_width), ...]
        
        conv_layers, bn_layers = [], []
        h_channels_in = channels[0]
        for i_layer, h_channels_out in enumerate(channels[1:]):
            # Getting the hidden layer-specific hyper-parameters
            h_kernel_size = kernel_size[i_layer]
            h_stride = stride[i_layer]
            h_padding = padding[i_layer]
            h_dilation = dilation[i_layer]
            h_output_padding = output_padding[i_layer]
            h_padding_mode = padding_mode[i_layer]
            h_bias = bias[i_layer]

            ###############################################
            ##### Compiling the Conv Keyword Arguments ####
            ###############################################
            conv_kwargs = dict(out_channels=n_seeds*h_channels_out, 
                kernel_size=h_kernel_size, stride=h_stride, padding=h_padding, 
                groups=n_seeds, dilation=h_dilation, bias=h_bias, device=device,
                dtype=dtype)
            if not be_lazy:
                conv_kwargs['in_channels'] = n_seeds * h_channels_in
            if transpose_conv:
                conv_kwargs['output_padding'] = h_output_padding
            else:
                conv_kwargs['padding_mode'] = h_padding_mode
            
            ###############################################
            ##### Constructing the Convolutional Layer ####
            ###############################################
            conv_layer = nn_ConvLayer(**conv_kwargs)
            conv_layers.append(conv_layer)

            ###############################################
            ##### Compiling the Norm Keyword Arguments ####
            ###############################################
            bn_kwargs = dict(device=device, dtype=dtype)
            if (norm in ('none', None)):
                pass
            elif (norm in ('batch', 'instance')) and be_lazy:
                pass
            elif (norm in ('batch', 'instance')) and (not be_lazy):
                bn_kwargs['num_features'] = n_seeds * h_channels_out
            elif norm == 'layer':
                bn_kwargs['shape'] = (n_seeds,)
                bn_kwargs['normalized_shape'] = x_shapes[i_layer]
                bn_kwargs['mb_order'] = 0 
                # `mb_order = 0` means that the inputs are gonna be 
                # shaped like `(n_mb, n_seeds, ...)`.
            else:
                raise ValueError('undefined case')
            
            for key, val in norm_kwargs.items():
                assert key not in bn_kwargs
                bn_kwargs[key] = val
            
            ###############################################
            ##### Constructing the Normalization Layer ####
            ###############################################
            is_hdnlayer = (i_layer < (n_layers - 1))
            has_bn1 = (is_hdnlayer or do_normout)
            has_bn2 = (norm not in ('none', None))
            has_bn = has_bn1 and has_bn2

            if not(has_bn):
                bn_layer = None
            elif norm == 'batch':
                bn_layer = nn_BatchNormLayer(**bn_kwargs)
            elif norm == 'instance':
                bn_layer = nn_InstanceNormLayer(**bn_kwargs)
            elif norm == 'layer':
                bn_layer = BatchLayerNorm(**bn_kwargs)
            else:
                raise ValueError(f'undefined normalization type "{norm}"')
            bn_layers.append(bn_layer)
            
            h_channels_in = h_channels_out
        
        activation = act2fn[act]() if isinstance(act, str) else act
        assert callable(activation)
        
        #########################################################
        ############# Storing the Class Variables ###############
        #########################################################
        self.channels = channels
        self.shape = (n_seeds,)
        self.n_convdims = n_convdims
        self.n_seeds = n_seeds
        self.device = device
        self.dtype = dtype
        self.rng = rng
        self.transpose_conv = transpose_conv
        self.mb_order = mb_order
        self.activation = activation
        self.do_actout = do_actout
        self.do_normout = do_normout
        self.convs = nn.ParameterList(conv_layers)
        self.norms = nn.ParameterList(bn_layers) if norm not in (None, 'none') else bn_layers
        self.x_inshape = x_inshape
        self.x_shapes = x_shapes
        self.x_inspdims = x_inspdims
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        self.normtype = norm

        self.needs_init = True
        if not be_lazy:
            # Initializing the layers
            self.init_layers()
            self.needs_init = False
       
    def forward(self, x):
        if self.needs_init:
            n_xchannels = x.shape[-(self.n_convdims + 1)]
            self.channels[0] = n_xchannels
            self.x_inshape, self.x_shapes = get_xconvshapes(self.x_inspdims, self.n_convdims, 
                self.channels, self.transpose_conv, self.padding, self.dilation, 
                self.kernel_size, self.stride, self.output_padding)

        x_input = x
        n_convdims = self.n_convdims
        n_seeds = self.n_seeds
        h_channels, *h_dims = self.x_inshape
        do_actout, do_normout, normtype = self.do_actout, self.do_normout, self.normtype
        cnn_layers, bn_layers, x_shapes = self.convs, self.norms, self.x_shapes
        n_layers = len(cnn_layers)
        assert len(bn_layers) == n_layers
        assert len(x_shapes) == n_layers
        
        h_dims_ = [h_dim if h_dim is not None else x_dim 
            for h_dim, x_dim in zip(h_dims, x.shape[-n_convdims:])]
        
        if self.mb_order == 1:
            n_samps = x.shape[1]
            assert x.shape == (n_seeds, n_samps, h_channels, *h_dims_)
            x = x.transpose(0, 1)
        else:
            n_samps = x.shape[0]
        assert x.shape == (n_samps, n_seeds, h_channels, *h_dims_)

        x = x.reshape(n_samps, n_seeds * h_channels, *h_dims_)
        assert x.shape == (n_samps, n_seeds * h_channels, *h_dims_)
        
        n_actcalls, n_bncalls = 0, 0
        for i_layer, (cnn_layer, bn_layer, x_shape) in enumerate(zip(cnn_layers, bn_layers, x_shapes)):
            x = cnn_layer(x)

            h_channels, *h_dims = x_shape
            h_dims_ = [h_dim if h_dim is not None else x_dim 
                for h_dim, x_dim in zip(h_dims, x.shape[-n_convdims:])]
            
            assert x.shape == (n_samps, n_seeds * h_channels, *h_dims_), dedent(f'''
                {x.shape} != {(n_samps, n_seeds * h_channels, *h_dims_)}''')
            
            if (self.normtype == 'layer') and (bn_layer is not None):
                x = x.reshape(n_samps, n_seeds, h_channels, *h_dims_)
                assert x.shape == (n_samps, n_seeds, h_channels, *h_dims_)

                x = bn_layer(x)
                assert x.shape == (n_samps, n_seeds, h_channels, *h_dims_)

                x = x.reshape(n_samps, n_seeds * h_channels, *h_dims_)
                assert x.shape == (n_samps, n_seeds * h_channels, *h_dims_)
                n_bncalls += 1
            elif (bn_layer is not None):
                x = bn_layer(x)                    
                assert x.shape == (n_samps, n_seeds * h_channels, *h_dims_)
                n_bncalls += 1
            
            if do_actout or (i_layer < (n_layers - 1)):
                x = self.activation(x)
                assert x.shape == (n_samps, n_seeds * h_channels, *h_dims_)
                n_actcalls += 1
        
        if do_actout: 
            assert n_actcalls == n_layers
        else: 
            assert n_actcalls == (n_layers - 1)
        
        if normtype in ('none', None):
            assert n_bncalls == 0
        elif do_normout:
            assert n_bncalls == n_layers
        else: 
            assert n_bncalls == (n_layers - 1)
        
        x = x.reshape(n_samps, n_seeds, h_channels, *h_dims_)
        assert x.shape == (n_samps, n_seeds, h_channels, *h_dims_)
        
        if self.mb_order == 1:
            x = x.transpose(0, 1)
            assert x.shape == (n_seeds, n_samps, h_channels, *h_dims_)
        
        if self.needs_init:
            self.init_layers()
            self.needs_init = False
            x = self.forward(x_input)

        return x
    
    def init_layers(self):
        assert self.needs_init
        assert self.channels[0] is not None, 'The number of input channels is still null'

        n_seeds = self.n_seeds
        h_channels_in = self.channels[0]
        rng = self.rng
        transpose_conv = self.transpose_conv
        
        with torch.no_grad():
            #########################################################
            ############## Initializing the CNN Layers ##############
            #########################################################
            for layer, h_channels_out, h_kernel_size in zip(self.convs, self.channels[1:], self.kernel_size):
                if not transpose_conv:
                    h1, h2 = h_channels_out, h_channels_in
                else:
                    h1, h2 = h_channels_in, h_channels_out
                
                w = layer.weight
                assert w.shape == (n_seeds * h1, h2, *h_kernel_size), f'{w.shape} != {(n_seeds * h1, h2, *h_kernel_size)}'
                w2 = w.reshape(n_seeds, h1, h2, *h_kernel_size)
                assert w2.shape == (n_seeds, h1, h2, *h_kernel_size)
                w2 = batch_kaiming_uniform_(w2, rng, a=math.sqrt(5))
                assert w2.shape == (n_seeds, h1, h2, *h_kernel_size)
                
                b = layer.bias
                if b is not None:
                    assert b.shape == (n_seeds * h_channels_out,)
                    fan_in, _ = _calculate_fan_in_and_fan_out(w2[0])
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        u1 = rng.rand(n_seeds, h_channels_out)
                        assert u1.shape == (n_seeds, h_channels_out)
                        u2 = (2 * bound) * u1.reshape(n_seeds * h_channels_out) - bound
                        assert u2.shape == (n_seeds * h_channels_out,)
                        assert u2.shape == b.shape
                        b.copy_(u2)
                        
                h_channels_in = h_channels_out

    def state_dict(self, *args, **kwargs):
        sdict = super().state_dict(*args, **kwargs)
        return get_state_dict_seeded(sdict, self.shape)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        sdict = get_state_dict_unseeded(state_dict, self.shape)
        super().load_state_dict(sdict, *args, **kwargs)


class Flattener(nn.Module):
    def __init__(self, dims: dict):
        '''
        Takes the dimensions information for the input variables 
        and constructs a flattener module.

        Args:

            dims (dict): the dimensions information for the input 
                variables.

                Example:

                    dims = {'m_prthst': (n_chnls, n_len),
                            'n_prthst': (3,)}
        '''
        self.dims = odict([(x_name, (tuple(x_dims), np.prod(x_dims)))
             for x_name, x_dims in dims.items()])
        
        super().__init__()
        
    def forward(self, x: dict):
        '''
        Takes a dictionary of tensors and flattens them.

        Args:
            x (dict): The dictionary of tensors with the same names as 
                specified before.

                Example:

                    x = {'m_prthst': torch.randn(n_seeds, n_mb, n_chnls, n_len)
                         'n_prthst': torch.randn(n_seeds, n_mb,       3)}

        Returns:
            x_flat (torch.tensor): The flattened and concatenated version of 
                the input.

                Example:

                    x_flat = torch.randn(n_seeds, n_mb, n_chnls * n_len + 3)
        '''
        dims = self.dims
        x_cp = dict(x)

        # Checking that all x members have identical `(n_seeds, n_mb)` dimensions
        x2dims = {x_name: tuple(x_mb.shape[:2]) for x_name, x_mb in x_cp.items()}
        assert len(set(x_dims for x_name, x_dims in x2dims.items())) == 1, dedent(f'''
            the n_seed and n_mb dimensions do not match among variables: {x2dims}''')

        cat_lst = []
        for x_name, (x_dims, x_size) in dims.items():
            x_mb = x_cp.pop(x_name)

            n_seeds, n_mb, *x_dims_ = x_mb.shape
            x_dims, x_size = self.dims[x_name]
            assert tuple(x_dims_) == x_dims, dedent(f'''
                dimensions of the {x_name} do no match the dims dictionary:
                    {x_dims_} != {x_dims}''')

            x_mbflat = x_mb.reshape(n_seeds, n_mb, x_size)
            cat_lst.append(x_mbflat)
        
        assert len(x_cp) == 0, f'unused x variables: {x_cp.keys()}'

        x_flat = torch.cat(cat_lst, dim=-1)

        return x_flat

        
class Deflattener(nn.Module):
    def __init__(self, dims: dict):
        '''
        Takes the dimensions information for the input variables 
        and constructs an unflattener module.

        Args:

            dims (dict): the dimensions information for the input 
                variables.

                Example:

                    dims = {'m_prthst': (n_chnls, n_len),
                            'n_prthst': (3,)}
        '''
        self.dims = odict([(x_name, (x_dims, np.prod(x_dims)))
             for x_name, x_dims in dims.items()])

        self.dim_flat = sum(x_size for x_name, (x_dims, x_size) in self.dims.items())
        
        super().__init__()
        
    def forward(self, x_flat):
        '''
        Takes a dictionary of tensors and unflattens them.

        Args:
            x_flat (torch.tensor): The flattened and concatenated version of 
                the input.

                Example:

                    x_flat = torch.randn(n_seeds, n_mb, n_chnls * n_len + 3)
        
        Returns:
            x (dict): The dictionary of tensors with the same names as 
                specified before.

                Example:

                    x = {'m_prthst': torch.randn(n_seeds, n_mb, n_chnls, n_len)
                         'n_prthst': torch.randn(n_seeds, n_mb,       3)}
            
        '''
        dims = self.dims
        
        assert x_flat.shape[-1] == self.dim_flat, f'''
            x_flat should have a shape of `(n_seeds, n_mb, ..., {self.dim_flat})`: 
            x_flat.shape = {x_flat.shape}, dims = {dims}'''

        i_dim, x = 0, odict()
        for x_name, (x_dims, x_size) in dims.items():
            x[x_name] = x_flat[..., i_dim: i_dim+x_size].unflatten(-1, x_dims)
            i_dim += x_size

        return x

###############################################################################
######################### Configurable Module Definer #########################
###############################################################################

#######################################
###### Config Module Utitlities #######
#######################################

class ShapeChecker:
    def __init__(self):
        self.varshist = dict()
        self.check_stack = list()
        self.expctd_shapes = dict()
        self.actual_shapes = dict()

    @property
    def shapes(self):
        return self.actual_shapes

    @property
    def vars(self):
        return {expctd_dim: actual_dim for expctd_dim, (arg_name, actual_dim) in self.varshist.items()}
        
    @staticmethod
    def validate(shapes, kwargs=None):
        assert isinstance(shapes, dict), dedent(f'''
            The `shapes` option does not have a valid type, and 
            has a "{type(shapes)}" type. The `shapes` option
            should be `dict(str: list(int | str))`:
                shapes = {shapes}''')

        if kwargs is not None:
            assert isinstance(kwargs, dict), dedent(f'''
                The `kwargs` option does not have a valid type, and 
                has a "{type(kwargs)}" type. The `kwargs` option
                should be `dict(str: torch.tensor | np.ndarray)`:
                    kwargs = {kwargs}''')

        for arg_name, arg_shape in shapes.items():
            assert isinstance(arg_name, str),  dedent(f'''
                The `shapes` option is not a proper argument name (str) to 
                list[int|str] mapping. One of the argument names "{arg_name}" 
                is not a string and has a type of "{type(arg_name)}":
                    shape = {arg_shape}
                    shapes = {shapes}''')
            assert all(isinstance(d_inp, (int, str)) for d_inp in shapes), dedent(f'''
                The `shapes` option is not in a proper list[list[int|str]] format.
                One of the `shapes` members is offending the `list[int|str]` format:
                    offending member = {shapes}
                    shapes = {shapes}''')

            if kwargs is not None:
                # Making sure the argument has a value in kwargs
                assert arg_name in kwargs, dedent(f'''
                    The keyword arguments had different keys than `shapes`. 
                    These two should match:
                        missing key = {arg_name}
                        shapes.keys() = {shapes.keys()}
                        kwargs.keys() = {kwargs.keys()}''')

                arg_tnsr = kwargs[arg_name]
                # Making sure the argument value has a shape
                assert (hasattr(arg_tnsr, 'shape') or (isinstance(arg_tnsr, (list, tuple)) 
                    and all(isinstance(x, int) for x in arg_tnsr))), dedent(f'''
                    The "{arg_name}" argument has a corresponding value 
                    that is not a tensor or a numpy array, and has a 
                    "{type(arg_tnsr)}" type instead. There is no `shape`
                    attribute to this type:
                        arg name = {arg_name}
                        arg_tnsr = {arg_tnsr}
                        shapes = {shapes}
                        kwargs = {kwargs}''')

        for arg_name, arg_shape in shapes.items():
            assert isinstance(arg_shape, list), dedent(f'''
                The shape for the "{arg_name}" member does not have a valid 
                type, and has a "{type(arg_shape)}" type. It should be a 
                `list(int | str)`:
                    Offending arg name = {arg_name}
                    Offending shape = {arg_shape}
                    The shapes = {shapes}''')

            n_stars = 0
            for d_inp in arg_shape:
                assert isinstance(d_inp, (str, int)), dedent(f'''
                    The shape for the "{arg_name}" member does not have a valid form, 
                    since the elements must be identifiable strings or integers:
                        Offending shape member = {d_inp}
                        Offending arg name = {arg_name}
                        Offending shape = {arg_shape}
                        The shapes = {shapes}''')

                if isinstance(d_inp, str):
                    d_inpstr = d_inp.strip()
                    if d_inpstr.startswith('*'):
                        n_stars += 1
                        # d_inpstrname = d_inpstr[1:].strip()
                        # # Note: no need to be strict with the following. As 
                        # #       long as there will be a single star, the 
                        # #        evaluation process can take care of this.
                        # assert d_inpstrname.isidentifier(), dedent(f'''
                        #     The "{arg_name}" `shapes` member includes a non-identifier 
                        #     string (i.e., a string that cannot be a valid python variable 
                        #     name). If you're trying to specify this variable as something 
                        #     to be evaluated, make sure you don't start it with a star. All 
                        #     such elements must be identifiable strings:
                        #         violating shape member = {d_inp}
                        #         arg_shape = {arg_shape}
                        #         shapes = {shapes}''')

                    if n_stars > 1:
                        raise ValueError(dedent(f'''
                            Too many * comprehensions found in `arg_shape`. Only a single 
                            element can be a string starting with *:
                                arg_shape = {arg_shape}
                                shapes = {shapes}'''))

    def append(self, shapes: dict, kwargs: dict):
        check_stack = self.check_stack
        expctd_shapes = self.expctd_shapes
        actual_shapes = self.actual_shapes

        ShapeChecker.validate(shapes, kwargs)

        for arg_name, expctd_shape in shapes.items():
            arg_tnsr = kwargs[arg_name]
            actual_shape = arg_tnsr.shape if hasattr(arg_tnsr, 'shape') else arg_tnsr

            # Storing these for safe keeping
            expctd_shapes[arg_name] = tuple(expctd_shape)
            actual_shapes[arg_name] = tuple(actual_shape)

            # Making sure the number of star comprehensions is reasonable
            n_stars = sum((isinstance(expctd_dim, str) and expctd_dim.strip().startswith('*'))
                          for expctd_dim in expctd_shape)
            if n_stars == 0:
                assert len(actual_shape) == len(expctd_shape), dedent(f'''
                    The `shape` has {len(expctd_shape)} members, but the input 
                    tensor is {len(actual_shape)}-dimensional:
                        arg name = {arg_name}
                        actual shape = {actual_shape}
                        expctd shape = {expctd_shape}''')
            elif n_stars == 1:
                assert len(actual_shape) >= len(expctd_shape), dedent(f'''
                    The `shape` has {len(expctd_shape)} members, but the input 
                    tensor is {len(actual_shape)}-dimensional:
                        arg name = {arg_name}
                        actual shape = {actual_shape}
                        expctd shape = {expctd_shape}''')
            else:
                raise ValueError('''unsafe code manipulation detected. This 
                    case should have been ruled out in the constructor.''')

            # Plan: Because there may be a star comprehension, we have to update 
            #       the varshist in three rounds:
            #          (1) iterate through `shape` and `actual_shape` from left to right, 
            #              pop what you process, and break when you see a star.
            #          (2) iterate through `shape` and `actual_shape` from right to left, 
            #              pop what you process, and break when you see a star.
            #          (3) if anything remained in `shape`, it should be a single star
            #              string. assign whatever remained to

            # Taking a temporary copy of `shape` and `actual_shape` for popping
            expctd_shapecp = list(expctd_shape)
            actual_shapecp = list(actual_shape)

            # Step (1): iterate through `shape` and `actual_shape` from left to right, 
            #           pop what you process, and break when you see a star.
            # Step (2): iterate through `shape` and `actual_shape` from left to right, 
            #           pop what you process, and break when you see a star.
            for idx_direction in (0, -1):
                while len(expctd_shapecp) > 0:
                    if (isinstance(expctd_shapecp[idx_direction], str) and
                            expctd_shapecp[idx_direction].strip().startswith('*')):
                        break
                    expctd_dim = expctd_shapecp.pop(idx_direction)
                    actual_dim = actual_shapecp.pop(idx_direction)
                    if isinstance(expctd_dim, int):
                        assert expctd_dim == actual_dim, dedent(f'''
                            The input tensor does not match the input shape 
                            expectation:
                                arg name = {arg_name}
                                actual shape = {actual_shape}
                                expctd shape = {expctd_shape}
                                offense: {expctd_dim} != {actual_dim}''')
                    elif isinstance(expctd_dim, str):
                        check_stack.append((arg_name, expctd_dim, actual_dim))
                    else:
                        raise ValueError('unsafe code manipulation detected.')

            # Step (3): if anything remained in `shape`, it should be a single star
            #           string. assign whatever remained to
            if len(expctd_shapecp) > 0:
                assert n_stars == 1, dedent('''
                    unsafe code manipulation detected. a single star comprehension must have 
                    remained after the previous left-to-right and right-to-left rundowns.''')

                assert len(expctd_shapecp) == 1, dedent(f'''
                    unsafe code manipulation detected. a single star comprehension must have 
                    remained after the previous left-to-right and right-to-left rundowns:
                        expctd_shapecp = {expctd_shapecp}''')

                expctd_dim_ = expctd_shapecp.pop(0)

                assert (isinstance(expctd_dim_, str) and
                        expctd_dim_.strip().startswith('*')), dedent(f'''
                    unsafe code manipulation detected. a single star comprehension must have 
                    remained after the previous left-to-right and right-to-left rundowns:
                        expctd_dim = {expctd_dim_}''')

                # We'll remove the starting star, and assign the remaining parts of the 
                # arg shape to it.
                expctd_dim = expctd_dim_.strip()[1:].strip(' ')
                actual_dim = actual_shapecp

                # # Note: no need to be strict with the following. As 
                # #       long as there will be a single star, the 
                # #       evaluation process can take care of this.
                # assert expctd_dim.isidentifier(), dedent(f'''
                #     When processing the expected shape of the "{arg_name}" argument, 
                #     We found an expected dimension that is a string and starts with 
                #     a "*" comprehension:
                #         expected dimension = "{expctd_dim_}" 
                #     However, the name of the comprehended variable does not qualify 
                #     to be a python variable name:
                #         "{expctd_dim}".isidentifier() == False
                #     Make sure the assigned variable names are identifiable.
                #         arg name = {arg_name}
                #         actual shape = {actual_shape}
                #         expctd shape = {expctd_shape}
                #         offense: {expctd_dim} != {actual_dim}''')

                check_stack.append((arg_name, expctd_dim, actual_dim))

    def check(self):
        check_stack = self.check_stack
        varshist = self.varshist
        expctd_shapes = self.expctd_shapes
        actual_shapes = self.actual_shapes

        # Any `expctd_dim` member in the `check_stack` must be a string by now (since 
        # they're only added to `check_stack` if they were a string).
        # `expctd_dim` can be one of the two types:
        #    Type (I):  A string that is an identifier (i.e., can be a 
        #               valid python variable name).
        #    Type (II): A string that should be evaluated.

        # Phase 1: We will start with processing the Type (II) expected shapes. 
        #          They only need to be stored in the `varshist` dictionary 
        #          and be checked for conflicts.
        eval_checkstack = []
        for arg_name, expctd_dim, actual_dim in check_stack:
            expctd_shape = expctd_shapes[arg_name]
            actual_shape = actual_shapes[arg_name]
            assert isinstance(expctd_dim, str)

            # Let's skip the type (II) strings that need to be evaluated.
            if not expctd_dim.isidentifier():
                eval_checkstack.append((arg_name, expctd_dim, actual_dim))
                continue

            if (expctd_dim not in varshist):
                varshist[expctd_dim] = (arg_name, actual_dim)
            else:
                (old_argname, old_actualdim) = varshist[expctd_dim]
                if old_argname == 'hparam':
                    old_expctdshape = None
                    old_actualshape = None
                else:
                    old_expctdshape = expctd_shapes[old_argname]
                    old_actualshape = actual_shapes[old_argname]

                if (old_actualdim != actual_dim):
                    raise ValueError(dedent(f'''
                        The "{expctd_dim}" shape variable is conflicting between multiple 
                        arguments or dimensions:
                        
                        Conflicting dimensions:
                            (1) Argument "{old_argname}" --> "{expctd_dim}" = {old_actualdim}
                            (2) Argument "{   arg_name}" --> "{expctd_dim}" = {   actual_dim}

                        More shape information:
                            (1) Argument "{old_argname}" --> "{old_expctdshape}" = {old_actualshape}
                            (2) Argument "{   arg_name}" --> "{expctd_shape   }" = {actual_shape   }

                        This could be due to thre reasons:
                            (a) The input variables may have not satisfied the required 
                                shape constraints. 
                            (b) The defined architecture is not producing the expected 
                                shape constraints.
                            (c) The defined architecture is working well, but the defined 
                                shape constraints are just incorrect.
                        
                        Either provide inputs with correct shapes, or fix your architecture 
                        and its shape constraints.'''))

        # Phase 2: Next, we will process the Type (I) expected shapes. 
        #          They need to be evaluated and checked for correctness.

        # Dropping the `arg_name` variables from the `varshist`
        shapevars = {expctd_dim: actual_dim for expctd_dim, (arg_name, actual_dim) in varshist.items()}
        for arg_name, expctd_dim, actual_dim in eval_checkstack:
            expctd_shape = expctd_shapes[arg_name]
            actual_shape = actual_shapes[arg_name]
            assert isinstance(expctd_dim, str)
            assert not expctd_dim.isidentifier()

            try:
                expctd_dimeval = eval_formula(expctd_dim, shapevars, catch=False)
            except Exception as exc:
                msg = dedent(f'''
                    When checking shapes for the "{arg_name}" argument,
                    The "{expctd_dim}" dimension could not be evaluated 
                    properly and python threw an exception during its 
                    evaluation.
                        argument name = {arg_name}
                        expcted dimension = {expctd_dim}
                        actual  dimension = {actual_dim}
                        expected shape = {expctd_shape}
                        actual shape = {actual_shape}
                    ''')

                msg += '\nThe infered shape variables are:\n'
                for expctd_dim_, (arg_name_, actual_dim_) in varshist.items():
                    msg += f'    {expctd_dim_} = {actual_dim_} <- inferred from "{arg_name_}"\n'

                exc.add_note(msg)
                raise exc

            if (expctd_dimeval != actual_dim):
                msg = dedent(f'''
                    When checking shapes for the "{arg_name}" argument,
                    The "{expctd_dim}" dimension did not match the actual 
                    dimension value of "{actual_dim}". This means that the 
                    infered shape variables did not satisfy the required 
                    shape constraints, and there must be something wrong 
                    with the architecture or the constraint definition:
                        argument name = {arg_name}
                        actual  dimension = {actual_dim}
                        expcted dimension = {expctd_dimeval}
                        expcted dimension expression = {expctd_dim}
                        expected shape = {expctd_shape}
                        actual shape = {actual_shape}
                ''')
                msg += f'\nThe infered shape variables are:\n'
                for expctd_dim_, (arg_name_, actual_dim_) in varshist.items():
                    msg += f'    {expctd_dim_} = {actual_dim_} <- inferred from "{arg_name_}"\n'
                raise ValueError(msg)

        # Since everything checked out, we can clear our checking queue
        check_stack.clear()

    def eval(self, shape):
        varshist = self.varshist

        # Dropping the `arg_name` variables from the `varshist`
        shapevars = {expctd_dim: actual_dim for expctd_dim, (arg_name, actual_dim) in varshist.items()}
        assert isinstance(shape, (list, tuple, str))
        
        try:
            if isinstance(shape, list):
                shape_eval = [eval_formula(str(val), shapevars, catch=False) for val in shape]
            elif isinstance(shape, tuple):
                shape_eval = tuple(eval_formula(str(val), shapevars, catch=False) for val in shape)
            elif isinstance(shape, str):
                shape_eval = eval_formula(str(shape), shapevars, catch=False)
            else:
                shape_eval = None
        except Exception as exc:
            msg = dedent(f'''
                When evaluating the shape "{shape}", python could not 
                evaluate it properly and threw an exception during its 
                evaluation.
                ''')

            msg += '\nThe infered shape variables were:\n'
            for expctd_dim_, (arg_name_, actual_dim_) in varshist.items():
                msg += f'    {expctd_dim_} = {actual_dim_} <- inferred from "{arg_name_}"\n'

            exc.add_note(msg)
            raise exc

        assert isinstance(shape_eval, (list, tuple)), dedent(f'''
            The evaluated shape of the "{shape}" expression produced a non-list/tuple type, 
            and has a "{type(shape_eval)}" type. The evaluated shape must follow a 
            `list[int]` structure.
                shape expression = {shape}
                evaluated shape type = {type(shape_eval)}
                evaluated shape = {shape_eval}
            ''')

        assert all(isinstance(d_inp, int) for d_inp in shape_eval), dedent(f'''
            The evaluated shape of the "{shape}" expression produced some non-integer
            values in the list. The evaluated shape must follow a `list[int]` structure.
                shape expression = {shape}
                evaluated shape type = {type(shape_eval)}
                evaluated shape = {shape_eval}
            ''')

        return shape_eval

    def update(self, hpdefs):
        assert isinstance(hpdefs, dict), dedent(f'''
            The provided `hpdefs` is not a dict, and has a type of 
            "{type(hpdefs)}":
                hpdefs = {hpdefs}''')
        varshist = self.varshist

        arg_name = 'hparam'
        for hp_name, hp_val in hpdefs.items():
            assert isinstance(hp_name, str), dedent(f'''
                The `hpdefs` keys must be identifier strings. However, 
                one of the keys ("{hp_name}") is not a string:
                    hpdefs.keys() = {hpdefs.keys()}
                    hpdefs = {hpdefs}''')
            assert hp_name.isidentifier(), dedent(f'''
                The `hpdefs` keys must be identifier strings. However, 
                one of the keys ("{hp_name}") is not an identifier:
                    hpdefs.keys() = {hpdefs.keys()}
                    hpdefs = {hpdefs}''')
            expctd_dim = hp_name
            actual_dim = hp_val
            varshist[expctd_dim] = (arg_name, actual_dim)


def parse_config(name, config, pop_opts: bool = False):
    '''
        This function parses the configs of a module. This function is 
        shared between the Pre-Processing and the ModuleMaker Classes.

        Args:
            name (str): The moodule name. Mainly used to identify 
                the error location when raising an exception.
            
            config (dict): The module description. Usually has the 
                `input`, `output`, `shape`, `hparams`, `hpdefs` and 
                `layers` keys.
            
            pop_opts (bool): whether to pop the options from the config 
                dictionary.
            
        Returns:
            parsed_config (dict): A dictionary with the following 
                parsed keys:
                    [`input`, `output`, `shape`, `hparams`, 
                     `hpdefs`, `layers`]
    '''
    module_name, module_config = name, config
    if not pop_opts:
        module_config = odict(module_config)
    #########################################################
    #################### Module Options #####################
    #########################################################

    ############## Part I: Module input names ###############
    ##### Processing the input names list to the module #####
    module_input_ = module_config.pop('input', None)
    module_ishape__ = None
    if module_input_ is None:
        module_input = ['input']
    elif isinstance(module_input_, str):
        module_input = [module_input_]
    elif isinstance(module_input_, (list, tuple)):
        module_input = list(module_input_)
    elif isinstance(module_input_, dict):
        module_input = list(module_input_.keys())
        module_ishape__ = list(module_input_.values())
    else:
        msg = f'''
            Module "{module_name}" specified an unusual input with 
            the type "{type(module_input_)}" as opposed to a list:
                input = {module_input_}
        '''
        raise ValueError(dedent(msg))

    # Getting the input shape namings and checkings. Note
    # that the `ishape` variable will be checked by the
    # `ShapeChecker` class.
    module_ishape_ = module_config.pop('shape', module_ishape__)
    if module_ishape_ is None:
        module_ishape = dict()
    elif isinstance(module_ishape_, (tuple, list)) and (module_input_ is None):
        assert all(isinstance(d_inp, (str, int))
                    for d_inp in module_ishape_), dedent(f'''
            Module "{module_name}" specified an unusual shape.
            Since there was no input specified to the module, the shape 
            should have a `list[int | str]` structure. However, it is 
            not following this structure:
                shape = {module_ishape_}
            ''')
        module_ishape = {module_input[0]: module_ishape_}
    elif isinstance(module_ishape_, (tuple, list)) and (module_input_ is not None):
        assert len(module_ishape_) == len(module_input), dedent(f'''
            Module "{module_name}" specified an unusual shape.
            There were "{len(module_input)}" input names specified to 
            the module, but the shape option has "{len(module_ishape_)}" 
            members. The `shape` and `input` options must have 
            corresponding elements, and the shape option must have a 
            `list[list[int | str]]` or a `dict(str: list[int | str]):
                shape = {module_ishape_}
                input = {module_input}
            ''')
        module_ishape = {arg_name: arg_shape
            for arg_name, arg_shape in zip(module_input, module_ishape_)}

        ShapeChecker.validate(module_ishape)
    elif isinstance(module_ishape_, dict):
        for argname in module_ishape_:
            assert argname in module_input, dedent(f'''
                Module "{module_name}" specified an unusual shape key.
                The argument "{argname}" appeared in the shape dictionary, 
                but such an argument name was not specified as an input:
                    input = {module_input}
                    shape = {module_ishape_}
                ''')
        module_ishape = module_ishape_
    else:
        msg = f'''
            Module "{module_name}" specified an unusual input shape with 
            the type "{type(module_ishape_)}":
                ishape = {module_ishape_}'''
        raise ValueError(dedent(msg))

    try:
        ShapeChecker.validate(module_ishape)
    except Exception as exc:
        exc.add_note(dedent(f'''
            This error happened while validating the shape option 
            of the "{module_name}" module.
        '''))
        raise exc

    ############## Part II: Module output names #############
    ##### Processing the output names list to the module ####
    module_output = module_config.pop('output', None)
    assert isinstance(module_output, (type(None), str, list, tuple, dict))

    ### Part III: Module hyper-parameters and definitions ###
    ########## Part III.I: Module hyper parameters ##########
    module_hparams_ = module_config.pop('hparams', None)
    if module_hparams_ is None:
        module_hparams = dict()
    elif isinstance(module_hparams_, (dict, odict)):
        module_hparams = odict(module_hparams_)
    else:
        msg = f'''
            Module "{module_name}" specified unusual hyper-paramerters
            with the type "{type(module_hparams_)}" as opposed to a dict:
                hparams = {module_hparams_}
        '''
        raise ValueError(dedent(msg))

    ############ Part III.II: Module Definitions ############
    module_defs_ = module_config.pop('defs', None)
    if module_defs_ is None:
        module_defs = dict()
    elif isinstance(module_defs_, (dict, odict)):
        module_defs = odict(module_defs_)
    else:
        msg = f'''
            Module "{module_name}" specified unusual definitions
            with the type "{type(module_defs_)}":
                defs = {module_defs_}
        '''
        raise ValueError(dedent(msg))

    # Translating the definitions and stroing them in
    # conjunction with the hyper-parameters.
    # `module_hpdefs` will contain both
    #   (1) the user-provided hyper-params in `module_hparams`, and
    #   (2) the other evaluated definitions in `module_defs`.
    module_hpdefs = odict(module_hparams)
    for hp_name, hp_def in module_defs.items():
        assert isinstance(hp_name, str)
        hp_defstr = str(hp_def)
        try:
            hp_val = eval_formula(hp_defstr, module_hpdefs, catch=True)
        except Exception as exc:
            exc.add_note(dedent(f'''
                This error happened while evaluating a formula:
                    formula: {hp_defstr}
                    variables: {module_hpdefs}
            '''))
            raise exc
        module_hpdefs[hp_name] = hp_val

    #########################################################
    ##################### Module Layers #####################
    #########################################################
    module_layers_ = module_config.pop('layers')
    assert isinstance(module_layers_, (dict, odict))
    assert len(module_layers_) > 0
    module_layers = odict(module_layers_)

    # If the user specified none as the output, they must have 
    # specified the outputs in the layers dictionary.
    if module_output == '{}':
        module_output = dict()

    # Assigning the last layer as the default output
    if module_output is None:
        last_layername = list(module_layers.keys())[-1]
        module_output = last_layername

    # Defaulting the layer inputs as the previous layer's output
    last_input = module_input
    for layer_name, layer_opts in module_layers.items():
        layer_opts.setdefault('input', last_input)
        last_input = layer_name

    # Making sure there are no leftover options
    msg = f'''Module "{module_name}" specified the following 
        unused options: {module_config}'''
    assert len(module_config) == 0, dedent(msg)

    # The method outputs
    parsed_config = dict(
        input=module_input, 
        shape=module_ishape, 
        output=module_output, 
        hparams=module_hparams, 
        hpdefs=module_hpdefs, 
        layers=module_layers)
    
    return parsed_config

#######################################
####### Configurable nn.Modules #######
#######################################

class ConfigModule(nn.Module):
    use_nnmoduledict = False
    def __init__(self, module_name: str, module_config: dict, lgstcs_dict: dict,
                 module_encyclopedia: dict, pop_opts: bool = False):
        super().__init__()
        
        parsed_config = parse_config(module_name, module_config, pop_opts=pop_opts)
        module_input = parsed_config['input'] 
        module_ishape = parsed_config['shape'] 
        module_output = parsed_config['output'] 
        module_hparams = parsed_config['hparams'] 
        module_hpdefs = parsed_config['hpdefs'] 
        module_layers = parsed_config['layers'] 

        # Creating the module layers
        layers, layers_shape = self.make_layers(module_name, module_layers,
            module_hpdefs, lgstcs_dict, module_encyclopedia)

        #########################################################
        ################ Writing the attributes #################
        #########################################################
        if self.use_nnmoduledict:
            self.layers = nn.ModuleDict(modules=layers)
        else:
            for layer_name, layer in layers.items():
                assert layer_name.isidentifier(), dedent(f'''
                    There is an error in connecting the layer 
                    "{layer_name}" to module "{module_name}.
                    Since the connection is not using `nn.ModuleDict`, 
                    the layer names need to be identifieable (i.e., be 
                    valid python variable names). However, "{layer_name}" 
                    does not satisfy this property.''')
                self.add_module(layer_name, layer)
            self.layers = layers

        self.layers_shape = layers_shape
        self.module_name = module_name
        self.module_input = module_input
        self.module_ishape = module_ishape
        self.module_layers = module_layers
        self.module_output = module_output
        self.module_hpdefs = module_hpdefs
        self.module_encyclopedia = module_encyclopedia

    def make_layers(self, module_name: str, module_layers: dict, module_hpdefs: dict,
                    lgstcs_dict: dict, module_encyclopedia: dict):

        all_modulemakers = {layer_type: layer_maker
                            for module_family, module_makers in module_encyclopedia.items()
                            for layer_type, layer_maker in module_makers.items()}

        layers = dict()
        layers_shape = dict()
        for layer_name, layer_opts in module_layers.items():
            '''
            Example:
            layer_name = '01_conv'
            layer_type = 'conv1d'

            layer_opts = {
                'type': 'conv1d',
                'in_channels': 'n_inpch',
                'out_channels': 'n_outch',
                'kernel_size': 'k',
                'padding': 'padding',
                'stride': 'stride',
                'bias': True,
                'padding_mode': 'zeros'}
            '''
            layer_opts = odict(layer_opts)

            layer_type = layer_opts.pop('type')
            assert isinstance(layer_type, str)
            layer_type = layer_type.lower()

            layer_input = layer_opts.pop('input')
            assert isinstance(layer_input, (str, list, tuple, dict))

            layer_shape = layer_opts.pop('shape', None)
            assert isinstance(layer_shape, (type(None), list, tuple, dict))

            assert layer_type in all_modulemakers, dedent(f'''
                Layer type {layer_type} is not defined. This was needed 
                when making the layer "{layer_name}" of type "{layer_type}" 
                in module "{module_name}":
                    module = {module_name}
                    layer_name = {layer_name}
                    layer_type = {layer_type}''')

            # Preparing the `layers_shape` dictionary
            if layer_shape is not None:
                layer_shape = deep2hie({layer_name: layer_shape},
                                       dictcls=odict, maxdepth=None, sep="/", digseqs=True,
                                       isleaf=lambda val: isinstance(val, (list, tuple)) and
                                                          all(isinstance(itm, (str, int)) for itm in val))
                try:
                    ShapeChecker.validate(layer_shape)
                except Exception as exc:
                    exc.add_note(dedent(f'''
                        This error happened while validating the shape option 
                        of the "{layer_name}" layer in the "{module_name}" module.
                    '''))
                    raise exc
            else:
                layer_shape = dict()
            layers_shape[layer_name] = layer_shape

            # The remaining options should be passed to the layer constructor
            # Therefore, we evaluate and translate their dictrionary values.
            layer_optseval = odict()
            for key, val in layer_opts.items():
                layer_optseval[key] = eval_formula(val, module_hpdefs, catch=True)

            # There might be some options unresolved yet in the repeat modules due to the
            # inaccessibility of the `i_rep` variable yet. We'll have to pass these on.
            if layer_type in ('repeat', 'dense'):
                layer_optseval['hpdefs'] = module_hpdefs

            if layer_type in all_modulemakers:
                #####################################
                ######## Layers Construction ########
                #####################################
                layer_cnstrctr = all_modulemakers[layer_type]

                assert layer_cnstrctr is not None

                try:
                    layer = layer_cnstrctr(**layer_optseval)
                except Exception as exc:
                    exc.add_note(dedent(f'''
                        This error happened while making the layer "{layer_name}" of 
                        type "{layer_type}" in module "{module_name}":
                            module = {module_name}
                            layer_name = {layer_name}
                            layer_type = {layer_type}'''))
                    raise exc
            else:
                raise ValueError(dedent(f'''
                    In module "{module_name}", layer "{layer_name}" was specified 
                    with the type "{layer_type}". This type is undefined for 
                    construction.'''))

            layers[layer_name] = layer

        return layers, layers_shape

    def forward(self, *args, **kwargs):
        module_name = self.module_name
        module_input = self.module_input
        module_layers = self.module_layers
        module_output = self.module_output
        layers_shape = self.layers_shape
        module_hpdefs = self.module_hpdefs
        module_ishape = self.module_ishape
        module_shapechecker = ShapeChecker()
        check_shapestepwise = True
        self_layers = self.layers

        node2vals = dict()
        #########################################################
        #### Validating the Arguments and Keyword Arguments #####
        #########################################################

        if len(args) > 0:
            assert len(kwargs) == 0, dedent(f'''
                In module "{module_name}", either the arguments or 
                keyword arguments must be empty. However, I got {len(args)} 
                arguments and {len(kwargs)} keyword arguments.''')

            assert len(args) == len(module_input), dedent(f'''
                In module "{module_name}", the number of input arguments 
                were expected to be {len(module_input)}, but {len(args)} 
                arguments were provided.''')

        if len(kwargs) > 0:
            assert len(args) == 0, dedent(f'''
                In module "{module_name}", either the arguments or 
                keyword arguments must be empty. However, I got {len(args)} 
                arguments and {len(kwargs)} keyword arguments.''')

            assert len(kwargs) == len(module_input), dedent(f'''
                In module "{module_name}", the number of input arguments 
                were expected to be {len(module_input)}, but {len(kwargs)} 
                keyword arguments were provided.''')

        # Making sure the arguments are not too many.
        assert len(args) <= len(module_input), dedent(f'''
            In module "{module_name}", we expected at most {len(module_input)} 
            arguments, but {len(args)} arguments were provided: 
                arg names = {module_input}
                args = {args}''')

        # Adding the arguments to the evaluated dictionary
        for node_name, node_mb in zip(module_input, args):
            node2vals[node_name] = node_mb

        for node_name, node_mb in kwargs.items():
            # Making sure the input is not over-defined.
            assert node_name not in node2vals, dedent(f'''
                In module "{module_name}", the "{node_name}" argument was 
                both specified as an argument and a keyword argument:
                arg names = {module_input}
                args = {args}
                kwargs.keys() = {kwargs.keys()}''')
            # Making sure the input name exists in the `module_input`.
            assert node_name in module_input, dedent(f'''
                In module "{module_name}", the "{node_name}" keyword argument 
                was provided, but such an argument name was not expected:
                expected arg names = {module_input}
                kwargs.keys() = {kwargs.keys()}''')
            node2vals[node_name] = node_mb

        # Making sure all inputs are specified
        for node_name in module_input:
            assert node_name in node2vals, dedent(f'''
                In module "{module_name}", the "{node_name}" input was 
                neither specified as an argument nor a keyword argument:
                arg names = {module_input}
                args = {args}
                kwargs.keys() = {kwargs.keys()}''')

        module_shapechecker.append(module_ishape, node2vals)
        module_shapechecker.update(module_hpdefs)
        if check_shapestepwise:
            module_shapechecker.check()

        #########################################################
        ############ Calling the Layers Sequentially ############
        #########################################################
        for layer_name, layer_opts in module_layers.items():
            layer_type = layer_opts['type']
            assert isinstance(layer_type, str)
            layer_type = layer_type.lower()

            layer_input = layer_opts['input']
            assert isinstance(layer_input, (str, list, tuple, dict))

            layer_input = [layer_input] if isinstance(layer_input, str) else layer_input
            assert isinstance(layer_input, (list, tuple, dict))

            layer_shape = layer_opts.get('shape', None)
            assert isinstance(layer_shape, (type(None), list, tuple, dict))

            layer_module = self_layers[layer_name]
            assert callable(layer_module)

            #####################################
            ####### Layer Input Collection ######
            #####################################
            layer_args = []
            layer_kwargs = dict()
            if isinstance(layer_input, (tuple, list)):
                for node_name in layer_input:
                    assert node_name in node2vals, dedent(f'''
                        In module "{module_name}", and layer "{layer_name}", an input node 
                        "{node_name}" was specified, but this node was not produced in prior 
                        steps or provided as an input to the whole module:
                            layer_input = {layer_input}
                            node2vals.keys() = {node2vals.keys()}''')
                    layer_args.append(node2vals[node_name])
            elif isinstance(layer_input, dict):
                for layer_argname, node_name in layer_input.items():
                    assert isinstance(layer_argname, str)
                    assert isinstance(node_name, str)
                    assert node_name in node2vals, dedent(f'''
                        In module "{module_name}", and layer "{layer_name}", an input node 
                        "{node_name}" was specified, but this node was not produced in prior 
                        steps or provided as an input to the whole module:
                            layer_input = {layer_input}
                            node2vals.keys() = {node2vals.keys()}''')
                    layer_kwargs[layer_argname] = node2vals[node_name]
            else:
                raise ValueError(dedent(f'''
                    In module "{module_name}", and layer "{layer_name}", an undefined 
                    type of input "{type(layer_input)}" was provided:
                        layer_input = {layer_input}'''))

            #####################################
            ######### Layer Forward Call ########
            #####################################
            try:
                layer_outvals = layer_module(*layer_args, **layer_kwargs)
            except Exception as exc:
                exc.add_note(dedent(f'''
                    This error happened while making a forward call to the 
                    "{module_name}" module's "{layer_name}" layer with the 
                    "{layer_type}" type:
                        number of arguemnts = {len(layer_args)}
                        keyword argument names = {layer_kwargs.keys()}
                '''))
                raise exc

            #####################################
            ##### Storing the Layer Output ######
            #####################################
            # First, we convert the output into a hierarchical dictionary
            layer_outvalshie = deep2hie({layer_name: layer_outvals},
                dictcls=odict, maxdepth=None, sep="/", digseqs=True)

            # aaa = {kk: vv.shape for kk, vv in layer_outvalshie.items() if torch.is_tensor(vv)}
            # print(f'Module "{self.module_name}", Layer "{layer_name}": Output Types = {aaa}', flush=True)

            # Checking that the shapes match the expectations
            layer_shape = layers_shape[layer_name]
            module_shapechecker.append(layer_shape, layer_outvalshie)
            if check_shapestepwise:
                module_shapechecker.check()

            # Making sure we're not over-writing any previous nodes
            for key in layer_outvalshie:
                assert key not in node2vals, dedent(f'''
                    In module "{module_name}", and layer "{layer_name}", a duplicate 
                    ouput "{key}" was produced, which was already produced in prior 
                    layers or given as an input to the module:
                        node2vals.keys() = {node2vals.keys()}''')

            # Then, we put all the results in the `node2vals` dictionary
            node2vals.update(layer_outvalshie)

        #####################################
        ######### Output Collection #########
        #####################################
        if not check_shapestepwise:
            module_shapechecker.check()

        assert isinstance(module_output, (str, list, tuple, dict))

        if isinstance(module_output, str):
            assert module_output in node2vals, dedent(f'''
                In module "{module_name}", the output was speciefied as the "{module_output}" 
                node, but this node was not produced in prior steps or provided as an input 
                to the whole module:
                    node2vals.keys() = {node2vals.keys()}''')
            output = node2vals[module_output]
        elif isinstance(module_output, (list, tuple)):
            for node_name in module_output:
                assert node_name in node2vals, dedent(f'''
                    In module "{module_name}", the output node "{node_name}" was specified, 
                    but this node was not produced in prior steps or provided as an input 
                    to the whole module:
                        module_output = {module_output}
                        node2vals.keys() = {node2vals.keys()}''')
            output = tuple(node2vals[node_name] for node_name in module_output)
        elif isinstance(module_output, dict):
            output = odict()
            for out_argname, node_name in module_output.items():
                assert node_name in node2vals, dedent(f'''
                    In module "{module_name}", the output node "{node_name}" was specified, 
                    but this node was not produced in prior steps or provided as an input 
                    to the whole module:
                        module_output = {module_output}
                        node2vals.keys() = {node2vals.keys()}''')
                output[out_argname] = node2vals[node_name]
        else:
            raise ValueError(dedent(f'''
                    In module "{module_name}", an undefined type of output 
                    "{type(module_output)}" was provided:
                        module_output = {module_output}'''))

        return output


class RepeatLayer(nn.Module):
    use_nnmoduledict = False
    def __init__(self, n_reps: int, module: str, hparams: dict, hpdefs: dict,
                 module_encyclopedia: dict):
        super().__init__()

        all_modulemakers = {layer_type: layer_maker
                            for module_family, module_makers in module_encyclopedia.items()
                            for layer_type, layer_maker in module_makers.items()}

        assert module in all_modulemakers, dedent(f'''
            Module "{module}" was not defined when trying to 
            repeat {n_reps} copies of it.''')
        layer_cnstrctr = all_modulemakers[module]

        assert isinstance(hparams, dict), dedent(f'''
            When trying to repeat {n_reps} copies of module "{module}", 
            the hyper-parameters passed were not a dictionary and were 
            of type "{type(hparams)}":
                hparams = {hparams}"''')

        layers = []
        for i_rep in range(n_reps):
            # Translating any of the hparam elements that needed the `i_rep` variable.
            i_hparams = dict()
            for hpname, hpval in hparams.items():
                i_hparams[hpname] = eval_formula(hpval, {'i_rep': i_rep, **hpdefs}, catch=True)

            layer = layer_cnstrctr(**i_hparams)
            layers.append(layer)

        if self.use_nnmoduledict:
            self.layers = nn.ModuleList(modules=layers)
        else:
            for i_rep, layer in enumerate(layers):
                self.add_module(f'l{i_rep:02d}', layer)
            self.layers = layers

    def forward(self, *args, **kwargs):
        layers = self.layers
        input_args, input_kwargs = args, kwargs

        #########################################################
        #### Validating the Arguments and Keyword Arguments #####
        #########################################################
        if len(input_args) > 0:
            use_args = True
            assert len(input_kwargs) == 0, dedent(f'''
                When passing an inputs to a repeat layer, either the arguments or 
                keyword arguments must be empty. However, I got {len(input_args)} 
                arguments and {len(input_kwargs)} keyword arguments:
                    input_args = {input_args}:
                    input_kwargs = {input_kwargs}''')
        if len(input_kwargs) > 0:
            use_args = False
            assert len(input_args) == 0, dedent(f'''
                When passing an inputs to a repeat layer, either the arguments or 
                keyword arguments must be empty. However, I got {len(input_args)} 
                arguments and {len(input_kwargs)} keyword arguments:
                    input_args = {input_args}:
                    input_kwargs = {input_kwargs}''')

        n_args = len(input_args) + len(input_kwargs)
        assert n_args > 0, dedent('''
            No arguments or keyword arguments were passed 
            when calling the repeat module.''')

        last_args, last_kwargs = args, kwargs
        i_output = args if n_args > 1 else args[0]
        for i_rep, layer in enumerate(layers):
            i_output = layer(*last_args, **last_kwargs)
            if use_args and (len(input_args) > 1):
                assert isinstance(i_output, (list, tuple)), dedent(f'''
                    The inputs to the repeat module where a tuple of arguments, 
                    but the output of the "{i_rep}"th layer is not a tuple or 
                    a list ("{type(i_output)} is not a list"). The layer outputs 
                    must have the same type as the repeat module inputs (i.e., 
                    they should output a list/tuple if the repeat module input 
                    is a tuple of arguments):
                        i_output = {i_output},
                        input_args = {input_args}''')
                last_args, last_kwargs = i_output, dict()
                assert len(last_args) == n_args, dedent(f'''
                    There were {n_args} inputs fed to the repeat module, but the 
                    output of the "{i_rep}"th layer has a length of {len(last_args)}.
                        i_output = {i_output}
                        input_args = {input_args}''')
            elif use_args and (len(input_args) == 1):
                last_args = i_output if isinstance(i_output, (list, tuple)) else (i_output,)
                last_kwargs = dict()
                assert len(last_args) == n_args, dedent(f'''
                    There were {n_args} inputs fed to the repeat module, but the 
                    output of the "{i_rep}"th layer has a length of {len(last_args)}.
                        last_args = {last_args}
                        input_args = {input_args}''')
            else:
                assert not use_args
                assert isinstance(i_output, dict), dedent(f'''
                    The inputs to the repeat module where a dictionary of keyword
                    arguments,  but the output of the "{i_rep}"th layer is not a 
                    dictionary ()"{type(i_output)} is not a dict"). The layer outputs 
                    must have the same type as the repeat module inputs (i.e., 
                    they should output a dict if the repeat module input 
                    is a dict of keyword arguments:
                        i_output = {i_output}
                        input_kwargs = {input_kwargs}''')
                last_args, last_kwargs = tuple(), i_output
                assert len(last_kwargs) == n_args, dedent(f'''
                    There were {n_args} inputs fed to the repeat module, but the 
                    output of the "{i_rep}"th layer has a length of {len(last_kwargs)}.
                        last_kwargs = {last_kwargs}
                        input_kwargs = {input_kwargs}''')

        return i_output if use_args else last_kwargs


class DenseLayer(nn.Module):
    use_nnmoduledict = False
    def __init__(self, n_reps: int, n_hist: int, catdim: int | list | dict,
                 channel: str, module: str, hparams: dict, hpdefs: dict,
                 module_encyclopedia: dict):
        super().__init__()

        assert n_hist >= 1

        all_modulemakers = {layer_type: layer_maker
                            for module_family, module_makers in module_encyclopedia.items()
                            for layer_type, layer_maker in module_makers.items()}

        assert module in all_modulemakers, dedent(f'''
            Module "{module}" was not defined when trying to 
            repeat/dense {n_reps} copies of it.''')
        layer_cnstrctr = all_modulemakers[module]

        assert isinstance(hparams, dict), dedent(f'''
            When trying to repeat/dense {n_reps} copies of module "{module}", 
            the hyper-parameters passed were not a dictionary and were 
            of type "{type(hparams)}":
                hparams = {hparams}''')

        if (n_hist > 1):
            assert isinstance(channel, str), dedent(f'''
                When trying to repeat/dense {n_reps} copies of module "{module}", 
                the passed channel hyper-parameter name "{channel}" was not a 
                string and was of the type "{type(channel)}":
                    channel = {channel}''')
            assert channel in hparams, dedent(f'''
                When trying to repeat/dense {n_reps} copies of module "{module}", 
                the passed channel hyper-parameter name "{channel}" was not a member 
                of the module hyper-parameters dictionary:
                    channel = {channel}
                    hparams = {hparams}''')
            assert isinstance(hparams[channel], int), dedent(f'''
                When trying to repeat/dense {n_reps} copies of module "{module}", 
                the passed channel hyper-parameter "{channel}" did not have an integer 
                value in the module hyper-parameters dictionary and had a type of
                "{type(hparams[channel])}":
                    channel = {channel}
                    hparams[channel] = {hparams[channel]}
                    hparams = {hparams}''')

        layers = []
        for i_rep in range(n_reps):
            # Translating any of the hparam elements that needed the `i_rep` variable.
            i_hparams = dict()
            for hpname, hpval in hparams.items():
                i_hparams[hpname] = eval_formula(hpval, {'i_rep': i_rep, **hpdefs}, catch=True)

            if (n_hist > 1):
                i_hparams[channel] *= min(n_hist, (i_rep + 1))
            elif (n_hist == 1):
                pass
            else:
                raise ValueError(f'undefined n_hist={n_hist}')
            layer = layer_cnstrctr(**i_hparams)
            layers.append(layer)

        if self.use_nnmoduledict:
            self.layers = nn.ModuleList(modules=layers)
        else:
            for i_rep, layer in enumerate(layers):
                self.add_module(f'l{i_rep:02d}', layer)
            self.layers = layers
        self.n_hist = n_hist
        self.catdim = catdim

        # `empty_dict` is an immutable empty dictionary. It is needed when there are no
        # key-word arguments, yet you want to pass something as "**kwargs" to the next
        # layer. It helps us avoid making a ton of unnecessary empty dictionaries on the fly.
        self.empty_dict =  MappingProxyType(dict())
        # Python tuples are already immutable so, there is no need to wrap them with a
        # `MappingProxyType`.
        self.empty_tuple = tuple()

    def forward(self, *args, **kwargs):
        layers = self.layers
        n_hist = self.n_hist
        catdim = self.catdim
        empty_dict = self.empty_dict
        empty_tuple = self.empty_tuple

        input_args, input_kwargs = args, kwargs

        #########################################################
        #### Validating the Arguments and Keyword Arguments #####
        #########################################################
        n_args = len(input_args) + len(input_kwargs)
        assert n_args > 0, dedent('''
            No arguments or keyword arguments were passed 
            when calling the repeat module.''')

        if len(input_kwargs) > 0:
            use_args = False
            assert len(input_args) == 0, dedent(f'''
                When passing an input to a repeat/dense layer, either the arguments or 
                keyword arguments must be empty. However, I got {len(input_args)} 
                arguments and {len(input_kwargs)} keyword arguments:
                    input_args = {input_args}:
                    input_kwargs = {input_kwargs}''')
        else:
            use_args = True
            assert len(input_kwargs) == 0, dedent(f'''
                When passing an input to a repeat/dense layer, either the arguments or 
                keyword arguments must be empty. However, I got {len(input_args)} 
                arguments and {len(input_kwargs)} keyword arguments:
                    input_args = {input_args}:
                    input_kwargs = {input_kwargs}''')
            assert len(args) > 0, 'the code was unsafely manipulated'

        #########################################################
        ######## Validating the Concatenation Dimensions ########
        #########################################################
        if (n_hist > 1) and use_args:
            if isinstance(catdim, int):
                catdim_lst = [catdim] * n_args
            elif isinstance(catdim, (tuple, list)):
                catdim_lst = catdim
            else:
                raise ValueError(dedent(f'''
                    In the repeat/dense layer, the concatenation dimensions `catdim` must 
                    be an int, list, or tuple if the forward function inputs are 
                    sequential arguments. However, the provided `catdim` had a type 
                    of {type(catdim)}:
                        catdim = {catdim}
                        input_args = {input_args}
                        input_kwargs = {input_kwargs}'''))
            assert len(catdim_lst) == n_args, dedent(f'''
                In the repeat/dense layer, the concatenation dimensions `catdim` must 
                have the same length as the number of arguments. However, the 
                provided `catdim` had a length of {len(catdim_lst)}, while the 
                number of arguments is {n_args}:
                    catdim = {catdim_lst}
                    input_args = {input_args}
                    input_kwargs = {input_kwargs}''')
            assert all(isinstance(d, int) for d in catdim_lst), dedent(f'''
                In the repeat/dense layer, the concatenation dimensions `catdim` must 
                be either an integer, or a list/tuple of integers:
                    catdim types = {[type(d) for d in catdim_lst]}
                    catdim list = {catdim_lst}''')
        elif (n_hist > 1) and (not use_args):
            if isinstance(catdim, int):
                catdim_dict = {key: catdim for key in input_kwargs}
            elif isinstance(catdim, dict):
                catdim_dict = catdim
            else:
                raise ValueError(dedent(f'''
                    In the repeat/dense layer, the concatenation dimensions `catdim` must 
                    be an int or a dict if the forward function inputs are keyword 
                    arguments. However, the provided `catdim` had a type 
                    of "{type(catdim)}":
                        catdim = {catdim}
                        input_args = {input_args}
                        input_kwargs = {input_kwargs}'''))
            assert len(catdim_dict) == n_args, dedent(f'''
                In the repeat/dense layer, the concatenation dimensions `catdim` must 
                have the same length as the number of keyword arguments. However, 
                the provided `catdim` had a length of "{len(catdim_dict)}", while the 
                number of keyword arguments is {n_args}:
                    catdim = "{catdim}"
                    input_args = {input_args}
                    input_kwargs = {input_kwargs}''')
            assert all(isinstance(d, int) for d in catdim_dict.values()), dedent(f'''
                In the repeat/dense layer, the concatenation dimensions `catdim` must 
                be either an integer, or a list/tuple of integers:
                    catdim types = {[type(d) for d in catdim_dict.values()]}
                    catdim dict = {catdim_dict}''')

        hist_args, hist_kwargs = [args], [kwargs]
        # `ip1_args` and `ip1_kwargs` are the latest raw outputs of the layers.
        #  This can be useful for outputting them directly.
        ip1_kwargs = kwargs
        ip1_args = args if n_args > 1 else args[0]

        for i_rep, layer in enumerate(layers):
            #########################################################
            ###### Concatenating the Inputs to the i-th Layer #######
            #########################################################
            if n_hist == 1:
                i_args, i_kwargs = hist_args[-1], hist_kwargs[-1]
            elif (n_hist > 1) and use_args:
                i_argslst, i_kwargs = [], empty_dict
                for i_arg in range(n_args):
                    i_catlst = [args_[i_arg] for args_ in hist_args[-n_hist:]]
                    assert all(torch.is_tensor(tnsr) for tnsr in i_catlst), dedent(f'''
                        In the repeat/dense layer, when trying to concatenate the prior 
                        outputs to create the input to the {i_rep}-th layer, some 
                        concatenation elements were non-tensors. This happened while 
                        collecting the {i_arg}-th argument from the past {n_hist} 
                        layers outputs:
                            i_catlst types = {[type(xx) for xx in i_catlst]}
                            i_catlst = {i_catlst}''')
                    arg_i = torch.cat(i_catlst, dim=catdim_lst[i_arg])
                    i_argslst.append(arg_i)
                i_args = tuple(i_argslst)
            elif (n_hist > 1) and (not use_args):
                i_args, i_kwargs = empty_tuple, dict()
                for i_kwarg in input_kwargs:
                    i_catlst = [kwargs_[i_kwarg] for kwargs_ in hist_kwargs[-n_hist:]]
                    assert all(torch.is_tensor(tnsr) for tnsr in i_catlst), dedent(f'''
                        In the repeat/dense layer, when trying to concatenate the prior 
                        outputs to create the input to the {i_rep}-th layer, some 
                        concatenation elements were non-tensors. This happened while 
                        collecting the "{i_kwarg}" argument from the past {n_hist} 
                        layers outputs:
                            i_catlst types = {[type(xx) for xx in i_catlst]}
                            i_catlst = {i_catlst}''')
                    kwarg_i = torch.cat(i_catlst, dim=catdim_dict[i_kwarg])
                    i_kwargs[i_kwarg] = kwarg_i
            else:
                raise ValueError('the code was unsafely manipulated')

            #########################################################
            ################### Calling the Layer ###################
            #########################################################
            i_output = layer(*i_args, **i_kwargs)

            #########################################################
            ########### Adding the Outputs to the History ###########
            #########################################################
            if use_args and (len(input_args) > 1):
                ip1_args, ip1_kwargs = i_output, empty_dict
                assert isinstance(i_output, (list, tuple)), dedent(f'''
                    The inputs to the repeat module where a tuple of arguments, 
                    but the output of the "{i_rep}"th layer is not a tuple or 
                    a list ("{type(i_output)} is not a list"). The layer outputs 
                    must have the same type as the repeat module inputs (i.e., 
                    they should output a list/tuple if the repeat module input 
                    is a tuple of arguments):
                        i_output = {i_output},
                        input_args = {input_args}''')

                hist_args.append(i_output)
                hist_kwargs.append(empty_dict)
                assert len(i_output) == n_args, dedent(f'''
                    There were {n_args} inputs fed to the repeat module, but the 
                    output of the "{i_rep}"th layer has a length of {len(i_output)}.
                        i_output = {i_output}
                        input_args = {input_args}''')
            elif use_args and (len(input_args) == 1):
                ip1_args, ip1_kwargs = i_output, empty_dict

                i_outputs = i_output if isinstance(i_output, (list, tuple)) else (i_output,)
                hist_args.append(i_outputs)
                hist_kwargs.append(empty_dict)

                assert len(i_outputs) == n_args, dedent(f'''
                    There were {n_args} inputs fed to the repeat module, but the 
                    output of the "{i_rep}"th layer has a length of {len(i_outputs)}.
                        i_outputs = {i_outputs}
                        input_args = {input_args}''')
            else:
                ip1_args, ip1_kwargs = empty_tuple, i_output

                assert not use_args
                assert isinstance(i_output, dict), dedent(f'''
                    The inputs to the repeat module where a dictionary of keyword
                    arguments,  but the output of the "{i_rep}"th layer is not a 
                    dictionary ()"{type(i_output)} is not a dict"). The layer outputs 
                    must have the same type as the repeat module inputs (i.e., 
                    they should output a dict if the repeat module input 
                    is a dict of keyword arguments:
                        i_output = {i_output}
                        input_kwargs = {input_kwargs}''')

                hist_args.append(empty_tuple)
                hist_kwargs.append(i_output)

                assert len(i_output) == n_args, dedent(f'''
                    There were {n_args} inputs fed to the repeat module, but the 
                    output of the "{i_rep}"th layer has a length of {len(i_output)}.
                        i_output = {i_output}
                        input_kwargs = {input_kwargs}''')

                if (n_hist > 1):
                    assert set(i_output.keys()) == set(input_kwargs.items()), dedent(f'''
                        There output of the "{i_rep}"th layer is a dictionary with 
                        different keys than the input keyword argument keys. This is 
                        a problem, since this is a semi-dense layer with a number of 
                        history taps of {n_hist}, and need the keys to be concatenated:
                            i_output.keys() = {i_output.keys()}
                            input_kwargs.keys() = {input_kwargs.keys()}
                        ''')

        return ip1_args if use_args else ip1_kwargs


class FunctionalLayer(nn.Module):
    torch_functions = {
        'adjoint', 'argwhere', 'cat', 'concat', 'concatenate', 'conj', 'chunk', 'dsplit', 'column_stack',
        'dstack', 'gather', 'hsplit', 'hstack', 'index_add', 'index_copy', 'index_reduce', 'index_select',
        'masked_select', 'movedim', 'moveaxis', 'narrow', 'narrow_copy', 'nonzero', 'permute', 'reshape',
        'row_stack', 'select', 'scatter', 'diagonal_scatter', 'select_scatter', 'slice_scatter',
        'scatter_add', 'scatter_reduce', 'split', 'squeeze', 'stack', 'swapaxes', 'swapdims', 't', 'take',
        'take_along_dim', 'tensor_split', 'tile', 'transpose', 'unbind', 'unravel_index', 'unsqueeze',
        'vsplit', 'vstack', 'where', 'abs', 'absolute', 'acos', 'arccos', 'acosh', 'arccosh', 'add',
        'addcdiv', 'addcmul', 'angle', 'asin', 'arcsin', 'asinh', 'arcsinh', 'atan', 'arctan', 'atanh',
        'arctanh', 'atan2', 'arctan2', 'bitwise_not', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
        'bitwise_left_shift', 'bitwise_right_shift', 'ceil', 'clamp', 'clip', 'conj_physical', 'copysign',
        'cos', 'cosh', 'deg2rad', 'div', 'divide', 'digamma', 'erf', 'erfc', 'erfinv', 'exp', 'exp2',
        'expm1', 'fake_quantize_per_channel_affine', 'fake_quantize_per_tensor_affine', 'fix',
        'float_power', 'floor', 'floor_divide', 'fmod', 'frac', 'frexp', 'gradient', 'imag', 'ldexp',
        'lerp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and',
        'logical_not', 'logical_or', 'logical_xor', 'logit', 'hypot', 'i0', 'igamma', 'igammac', 'mul',
        'multiply', 'mvlgamma', 'nan_to_num', 'neg', 'negative', 'nextafter', 'polygamma', 'positive',
        'pow', 'quantized_batch_norm', 'quantized_max_pool1d', 'quantized_max_pool2d', 'rad2deg', 'real',
        'reciprocal', 'remainder', 'round', 'rsqrt', 'sigmoid', 'sign', 'sgn', 'signbit', 'sin', 'sinc',
        'sinh', 'softmax', 'sqrt', 'square', 'sub', 'subtract', 'tan', 'tanh', 'true_divide', 'trunc',
        'xlogy', 'argmax', 'argmin', 'amax', 'amin', 'aminmax', 'all', 'any', 'max', 'min', 'dist',
        'logsumexp', 'mean', 'nanmean', 'median', 'nanmedian', 'mode', 'norm', 'nansum', 'prod',
        'quantile', 'nanquantile', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive', 'var',
        'var_mean', 'count_nonzero', 'allclose', 'argsort', 'eq', 'equal', 'ge', 'greater_equal', 'gt',
        'greater', 'isclose', 'isfinite', 'isin', 'isinf', 'isposinf', 'isneginf', 'isnan', 'isreal',
        'kthvalue', 'le', 'less_equal', 'lt', 'less', 'maximum', 'minimum', 'fmax', 'fmin', 'ne',
        'not_equal', 'sort', 'topk', 'msort', 'stft', 'istft', 'bartlett_window', 'blackman_window',
        'hamming_window', 'hann_window', 'kaiser_window', 'atleast_1d', 'atleast_2d', 'atleast_3d',
        'bincount', 'block_diag', 'broadcast_tensors', 'broadcast_to', 'broadcast_shapes', 'bucketize',
        'cartesian_prod', 'cdist', 'clone', 'combinations', 'corrcoef', 'cov', 'cross', 'cummax',
        'cummin', 'cumprod', 'cumsum', 'diag', 'diag_embed', 'diagflat', 'diagonal', 'diff', 'einsum',
        'flatten', 'flip', 'fliplr', 'flipud', 'kron', 'rot90', 'gcd', 'histc', 'histogram',
        'histogramdd', 'meshgrid', 'lcm', 'logcumsumexp', 'ravel', 'renorm', 'repeat_interleave', 'roll',
        'searchsorted', 'tensordot', 'trace', 'tril', 'tril_indices', 'triu', 'triu_indices', 'unflatten',
        'vander', 'view_as_real', 'view_as_complex', 'resolve_conj', 'resolve_neg', 'addbmm', 'addmm',
        'addmv', 'addr', 'baddbmm', 'bmm', 'chain_matmul', 'cholesky', 'cholesky_inverse',
        'cholesky_solve', 'dot', 'geqrf', 'ger', 'inner', 'inverse', 'det', 'logdet', 'slogdet', 'lu',
        'lu_solve', 'lu_unpack', 'matmul', 'matrix_power', 'matrix_exp', 'mm', 'mv', 'orgqr', 'ormqr',
        'outer', 'pinverse', 'qr', 'svd', 'svd_lowrank', 'pca_lowrank', 'lobpcg', 'trapz', 'trapezoid',
        'cumulative_trapezoid', 'triangular_solve', 'vdot', 'zeros_like', 'full_like'}

    torch_nn_functionals = {'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d',
        'conv_transpose3d', 'unfold', 'fold', 'avg_pool1d', 'avg_pool2d', 'avg_pool3d', 'max_pool1d',
        'max_pool2d', 'max_pool3d', 'max_unpool1d', 'max_unpool2d', 'max_unpool3d', 'lp_pool1d',
        'lp_pool2d', 'lp_pool3d', 'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d',
        'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'fractional_max_pool2d',
        'fractional_max_pool3d', 'scaled_dot_product_attention', 'threshold', 'relu', 'hardtanh',
        'hardswish', 'relu6', 'elu', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu',
        'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softshrink',
        'gumbel_softmax', 'log_softmax',  'hardsigmoid', 'silu', 'mish', 'batch_norm', 'group_norm',
        'instance_norm', 'layer_norm', 'local_response_norm', 'rms_norm', 'normalize', 'linear',
        'bilinear', 'dropout', 'alpha_dropout', 'feature_alpha_dropout', 'dropout1d', 'dropout2d',
        'dropout3d', 'embedding', 'embedding_bag', 'one_hot', 'pairwise_distance', 'cosine_similarity',
        'pdist', 'binary_cross_entropy', 'binary_cross_entropy_with_logits', 'poisson_nll_loss',
        'cosine_embedding_loss', 'cross_entropy', 'ctc_loss', 'gaussian_nll_loss', 'hinge_embedding_loss',
        'kl_div', 'l1_loss', 'mse_loss', 'margin_ranking_loss', 'multilabel_margin_loss',
        'multilabel_soft_margin_loss', 'multi_margin_loss', 'nll_loss', 'huber_loss', 'smooth_l1_loss',
        'soft_margin_loss', 'triplet_margin_loss', 'triplet_margin_with_distance_loss', 'pixel_shuffle',
        'pixel_unshuffle', 'pad', 'interpolate', 'upsample', 'upsample_nearest', 'upsample_bilinear',
        'grid_sample', 'affine_grid', 'sigmoid', 'softmax', 'tanh'}

    def __init__(self, layer_type, layer_opts: dict = None):
        super().__init__()
        self.layer_type = layer_type
        self.layer_opts = layer_opts if layer_opts is not None else dict()

        tchnnfuncs = {f'nn.functional.{ltyp}' for ltyp in self.torch_nn_functionals}
        assert ((layer_type in self.torch_functions) or
            (layer_type in tchnnfuncs)), dedent(f'''
                The "{layer_type}" layer is not a torch function.''')

    def forward(self, *args, **kwargs):
        layer_type = self.layer_type
        layer_opts = dict(self.layer_opts)

        for i_arg, arg in enumerate(args):
            assert torch.is_tensor(arg), dedent(f'''
                The {i_arg}-th argument to the "{layer_type}" layer
                is not a torch tensor and has a "{type(arg)}" type.
                    args = {args}
                    kwargs = {kwargs}''')

        for arg_name, arg in kwargs.items():
            assert torch.is_tensor(arg), dedent(f'''
                The keyword argument "{arg_name}"  to the "{layer_type}" 
                layer is not a torch tensor and has a "{type(arg)}" type.
                    args = {args}
                    kwargs = {kwargs}''')

        n_args = len(args) + len(kwargs)
        assert n_args > 0, dedent(f'''
            No arguments or keyword arguments were passed 
            when calling the "{layer_type}" layer.''')

        try:
            if layer_type in ('cat', 'stack'):
                all_args = list(args) + list(kwargs.values())
                tch_fn = getattr(torch, layer_type)
                output = tch_fn(all_args, **layer_opts)
            elif layer_type == 'reshape':
                all_argvals = list(args) + list(kwargs.values())
                all_argkeys = [str(i) for i, arg in enumerate(args)] + list(kwargs.keys())

                ishape = layer_opts.pop('ishape')
                oshape = layer_opts.pop('oshape')

                # Making sure no options were left unused
                if len(layer_opts) > 0:
                    raise ValueError(dedent(f'''
                        Some options were left unused. Make sure you do not specify these, 
                        or specify them in a proper manner: 
                            unused options = {layer_opts}'''))

                assert n_args == 1, dedent(f'''
                    The reshape layer should take a single argument. 
                    However, {n_args} arguments were provided:
                        args = {args}
                        kwargs = {kwargs}''')

                shpchecker = ShapeChecker()
                shpchecker.append(ishape, dict(zip(all_argkeys, all_argvals)))
                shpchecker.check()
                oshape_eval = shpchecker.eval(oshape)

                output = all_argvals[0].reshape(*oshape_eval)                
            elif layer_type.startswith('nn.functional.'):
                tch_fn = getattr(torch.nn.functional, layer_type[14:])
                output = tch_fn(*args, **kwargs, **layer_opts)
            else:
                tch_fn = getattr(torch, layer_type)
                output = tch_fn(*args, **kwargs, **layer_opts)
        except Exception as exc:
            exc.add_note(dedent(f'''
                This error happened while making a forward call to a 
                "{layer_type}" layer.'''))
            raise exc

        return output


class MBCollapse(nn.Module):
    def __init__(self, layer, ndims):
        '''
        This module is only there to make the following input
        shape pattern possible.

        ```
        pool = nn.MaxPool2d(4)
        x = torch.randn(8, 9, 12, 16)
        y = pool(x)
        assert y.shape == (8, 9, 3, 4)
        ```

        Unfortunately, pytorch throws out an error if you
        feed a 4-d tensor to a `nn.MaxPool2d` layer.

        This module
        (1) collapses the batch dimensions into a single dimension,
        (2) feeds it to the max pool layer, and then
        (3) unflattens the batch dims to the original batch shape.
        '''
        super().__init__()
        self.ndims = ndims
        self.layer = layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim >= self.ndims, dedent(f'''
            You're feeding a {input.ndim}-dimensional input to a 
                "{self.layer}" 
            pytorch module. You need more input dimensions!
        ''')

        # Inferring the batch dimensions
        mb_dims = input.shape[:-self.ndims]
        mb_size = int(np.prod(tuple(mb_dims)))

        # Step (1): Collapseing the batch dimensions into a single dimension,
        input_valid = input.reshape(mb_size, *input.shape[-self.ndims:])
        # Step (2): Feeding it to the pytorch pooling layer
        output = self.layer(input_valid)
        # Step (3) Unflattening the batch dims to the original batch shape.
        output_valid = output.reshape(*mb_dims, *output.shape[-self.ndims:])

        return output_valid

#######################################
######### Module Maker Classes ########
#######################################

class AtomicLayerMaker:
    def __init__(self, layer_type, lgstcs_dict):
        self.layer_type = layer_type
        self.lgstcs_dict = lgstcs_dict
        self.module_families = self.get_families()

    @staticmethod
    def get_families():
        # The nn convolutional module names:
        #   `conv_modulenames == ['conv1d', ..., 'lazyconvtranspose3d']`
        conv_modulenames = [''.join(strtup) for strtup in product(['', 'lazy'],
            ['conv'], ['', 'transpose'], ['1d', '2d', '3d'])]

        # The nn normalization module names:
        #   `norm_modulenames == ['batchnorm1d', ..., 'lazyinstancenorm3d',
        #                         'layernorm']`
        norm_modulenames = [''.join(strtup) for strtup in product(['', 'lazy'],
            ['batch', 'instance'], ['norm'], ['1d', '2d', '3d'])] + ['layernorm']

        # The activation module names
        act_modulenamescap = ['ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh',
            'Hardswish', 'LeakyReLU', 'LogSigmoid', 'PReLU', 'ReLU', 'ReLU6',
            'RReLU', 'SELU', 'CELU', 'GELU', 'Sigmoid', 'SiLU', 'Mish', 'Softplus',
            'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold', 'GLU']
        act_modulenames = [actname.lower() for actname in act_modulenamescap]

        # The pooling module names
        pool_modulenames = [''.join(strtup) for strtup in product(['max', 'avg',
            'fractionalmax', 'lp', 'adaptivemax', 'adaptiveavg'], ['pool'], 
            ['1d', '2d', '3d'])]

        # The functional module names
        func_modulenames1 = list(FunctionalLayer.torch_functions)
        func_modulenames1 += [f'nn.functional.{lyrnm}' for lyrnm in FunctionalLayer.torch_nn_functionals]
        func_modulenames = list({lyrnm: None for lyrnm in func_modulenames1})

        module_families = {
            'conv': conv_modulenames,
            'cnn': ['cnn', 'dcnn'],
            'linear': ['linear', 'lazylinear'],
            'fcn': ['mlp', 'fcn'],
            'norm': norm_modulenames,
            'nrm': ['norm'],
            'pool': pool_modulenames,
            'pol': ['pool'],
            'act': act_modulenames,
            'actv': ['act'],
            'func': func_modulenames}

        return module_families

    def __call__(self, **layer_optseval):
        layer_type = self.layer_type
        lgstcs_dict = self.lgstcs_dict
        conv_modulenames = self.module_families['conv']
        norm_modulenames = self.module_families['norm']
        act_modulenames = self.module_families['act']
        pool_modulenames = self.module_families['pool']
        func_modulenames = self.module_families['func']

        mdl_shape = tuple(lgstcs_dict.get('shape', tuple()))
        mdl_size = math.prod(mdl_shape)
        is_mdlbatch = not (mdl_shape == tuple())

        nn_capmodulenames = nn.modules.__all__
        nn_lower2cap = {s.lower(): s for s in nn_capmodulenames}

        layer_kwargs = dict()
        try:
            if (not is_mdlbatch) and (layer_type in nn_lower2cap):
                layer_cnstrctr = getattr(nn, nn_lower2cap[layer_type])
                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Gathering the rest of the keyword arguments from the user-provided layer options
                for opt in list(layer_optseval.keys()):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer = layer_cnstrctr(**layer_kwargs)
            elif is_mdlbatch and (layer_type in conv_modulenames):
                '''
                1. These are basic layer types that are not deep, 
                do not have inherent activation or batch-normalizations.
                2. We will be using the more general `BatchCNN` to implement 
                these in a parallel manner.
                '''
                in_channels = layer_optseval.pop('in_channels', None)
                out_channels = layer_optseval.pop('out_channels')
                channels = [in_channels, out_channels]

                assert 'groups' not in layer_optseval, dedent('''
                    The "groups" option was specified. This option's 
                    functionality is reserved for seed parallelization, and cannot be 
                    provided by the user.''')

                # Preparing the module construction keyword arguments
                assert 'conv' in layer_type
                layer_trnspsx = 'transpose' in layer_type
                layer_nconvdims = int(layer_type[-2])
                layer_belazy = layer_type.startswith('lazy')
                if in_channels is None:
                    assert layer_belazy, dedent('''
                        The "in_channels" argument was set to None while 
                        the layer is not lazy.''')

                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict(channels=channels, n_convdims=layer_nconvdims, act='identity', 
                                    norm=None, do_actout=False, do_normout=False, norm_kwargs=None, 
                                    x_inspdims=None, transpose_conv=layer_trnspsx, mb_order=1)

                # Part II: Getting the logistics arguments
                for lgstc_opt in ('shape', 'rng', 'device', 'dtype'):
                    layer_kwargs[lgstc_opt] = lgstcs_dict[lgstc_opt]

                # Part III: Gathering the rest of the keyword arguments from the user-provided layer options
                passover_optnames = ('kernel_size', 'stride', 'padding', 'dilation', 'bias', 'padding_mode')
                for opt in set(passover_optnames).intersection(set(layer_optseval.keys())):
                    layer_kwargs[opt] = layer_optseval.pop(opt)
                layer = BatchCNN(**layer_kwargs)
            elif is_mdlbatch and (layer_type in ('cnn', 'dcnn')):
                # Preparing the module construction keyword arguments
                layer2trnspscnv = {'dcnn': True, 'cnn': False}
                layer_trnspscnv = layer2trnspscnv[layer_type]

                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict(transpose_conv=layer_trnspscnv)

                # Part II: Getting the logistics arguments
                for lgstc_opt in ('shape', 'rng', 'device', 'dtype'):
                    layer_kwargs[lgstc_opt] = lgstcs_dict[lgstc_opt]

                # Part III: Gathering the rest of the keyword arguments from the user-provided layer options
                passover_optnames = ('channels', 'n_convdims', 'kernel_size',
                                     'act', 'norm', 'do_actout', 'do_normout', 
                                     'stride', 'padding', 'dilation',
                                     'output_padding', 'padding_mode', 'norm_kwargs',
                                     'x_inspdims', 'mb_order')
                for opt in set(passover_optnames).intersection(set(layer_optseval.keys())):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer = BatchCNN(**layer_kwargs)
            elif is_mdlbatch and (layer_type in ('linear', 'lazylinear')):
                '''
                1. These are basic layer types that are not deep and activated.
                2. We will be using the more general `BatchFCN` to implement 
                these in a parallel manner.
                '''
                in_features = layer_optseval.pop('in_features')
                out_features = layer_optseval.pop('out_features')
                dims = [in_features, out_features]

                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict(dims=dims, act='identity', do_actout=False,
                                    mb_order=1, norm=None, do_normout=False)

                # Part II: Getting the logistics arguments
                for lgstc_opt in ('shape', 'rng', 'device', 'dtype'):
                    layer_kwargs[lgstc_opt] = lgstcs_dict[lgstc_opt]

                # Part III: Gathering the rest of the keyword arguments from the user-provided layer options
                layer_kwargs['bias'] = layer_optseval.pop('bias', True)

                layer = BatchFCN(**layer_kwargs)
            elif is_mdlbatch and (layer_type in ('mlp', 'fcn')):
                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Getting the logistics arguments
                for lgstc_opt in ('shape', 'rng', 'device', 'dtype'):
                    layer_kwargs[lgstc_opt] = lgstcs_dict[lgstc_opt]

                # Part III: Gathering the rest of the keyword arguments from the user-provided layer options
                passover_optnames = ('dims', 'act', 'bias', 'do_actout', 'do_normout', 'mb_order', 'norm')
                for opt in set(passover_optnames).intersection(set(layer_optseval.keys())):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer = BatchFCN(**layer_kwargs)
            elif is_mdlbatch and (layer_type in norm_modulenames):
                if 'batch' in layer_type:
                    layer_normtype = 'batch'
                elif 'instance' in layer_type:
                    layer_normtype = 'instance'
                elif 'layer' in layer_type:
                    layer_normtype = 'layer'
                else:
                    msg = f'''
                        The normalization type ('batch' or 'instance' or 'layer') 
                        could not be infered from the layer type:
                            layer_type = {layer_type}.'''
                    raise ValueError(dedent(msg))

                if layer_normtype == 'layer':
                    normalized_shape = layer_optseval.pop('normalized_shape')
                    n_channels, *x_spdims = normalized_shape
                    n_spdims = len(x_spdims)

                    norm_kwargs = dict()
                    for opt in ['eps', 'elementwise_affine', 'bias']:
                        if opt in layer_optseval:
                            norm_kwargs[opt] = layer_optseval.pop(opt)
                else:
                    assert layer_normtype in ('batch', 'instance')
                    n_spdims = int(layer_type[-2])
                    n_channels = layer_optseval.pop('num_features')
                    x_spdims = None

                    norm_kwargs = dict()
                    for opt in ['eps', 'momentum', 'affine', 'track_running_stats']:
                        if opt in layer_optseval:
                            norm_kwargs[opt] = layer_optseval.pop(opt)

                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict(norm=layer_normtype, n_spdims=n_spdims,
                                    n_channels=n_channels, x_spdims=x_spdims, norm_kwargs=norm_kwargs,
                                    mb_order=1)

                # Part II: Getting the logistics arguments
                for lgstc_opt in ('shape', 'device', 'dtype'):
                    layer_kwargs[lgstc_opt] = lgstcs_dict[lgstc_opt]

                layer = BatchNorm(**layer_kwargs)
            elif is_mdlbatch and (layer_type == 'norm'):
                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Getting the logistics arguments
                for lgstc_opt in ('shape', 'device', 'dtype'):
                    layer_kwargs[lgstc_opt] = lgstcs_dict[lgstc_opt]

                # Part III: Gathering the rest of the keyword arguments from the user-provided layer options
                passover_optnames = ('norm', 'n_spdims', 'n_channels', 'x_spdims', 'norm_kwargs', 'mb_order')
                for opt in set(passover_optnames).intersection(set(layer_optseval.keys())):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer = BatchNorm(**layer_kwargs)
            elif is_mdlbatch and (layer_type in act_modulenames):
                act_modulemakers = {'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid,
                                    'identity': nn.Identity, 'silu':nn.SiLU, 'relu': nn.ReLU,
                                    'leakyrelu': nn.LeakyReLU}

                layer_typecap = nn_lower2cap[layer_type]
                layer_cnstrctr = getattr(nn, layer_typecap)
                #####################################
                ######## Layers Construction ########
                #####################################
                layer_cnstrctr = act_modulemakers[layer_type]

                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Gathering the rest of the keyword arguments from the user-provided layer options
                for opt in list(layer_optseval.keys()):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer = layer_cnstrctr(**layer_kwargs)
            elif is_mdlbatch and (layer_type == 'act'):
                layer_acttype = layer_optseval.pop('act')
                #####################################
                ######## Layers Construction ########
                #####################################
                layer_typecap = nn_lower2cap[layer_acttype]
                layer_cnstrctr = getattr(nn, layer_typecap)

                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Gathering the rest of the keyword arguments from the user-provided layer options
                for opt in list(layer_optseval.keys()):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer = layer_cnstrctr(**layer_kwargs)
            elif is_mdlbatch and (layer_type in pool_modulenames):
                pool_capnames = [''.join(strtup) for strtup in product(
                    ['Max', 'Avg', 'FractionalMax', 'LP', 'AdaptiveMax',
                     'AdaptiveAvg'], ['Pool'], ['1d', '2d', '3d'])]

                pool_modulemakers = {name.lower(): getattr(nn, name, NNVersionError(name))
                                     for name in pool_capnames}

                pool_dim = int(layer_type[-2])
                assert pool_dim in (1, 2, 3)

                #####################################
                ######## Layers Construction ########
                #####################################
                layer_cnstrctr = pool_modulemakers[layer_type]

                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Gathering the rest of the keyword arguments from the user-provided layer options
                for opt in list(layer_optseval.keys()):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer_ = layer_cnstrctr(**layer_kwargs)
                layer = MBCollapse(layer_, pool_dim)
            elif is_mdlbatch and (layer_type == 'pool'):
                pool_typescap = ['Max', 'Avg', 'FractionalMax', 'LP',
                                 'AdaptiveMax', 'AdaptiveAvg']
                pool_types = [s.lower() for s in pool_typescap]

                pool_type = layer_optseval.pop('type')
                assert pool_type in pool_types

                pool_dim = layer_optseval.pop('n_pooldim')
                assert pool_dim in (1, 2, 3)

                pool_typecap = pool_typescap[pool_types.index(pool_type)]
                nn_poollayername = f'{pool_typecap}Pool{pool_dim}d'

                #####################################
                ######## Layers Construction ########
                #####################################
                layer_cnstrctr = getattr(nn, nn_poollayername)

                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Gathering the rest of the keyword arguments from the user-provided layer options
                for opt in list(layer_optseval.keys()):
                    layer_kwargs[opt] = layer_optseval.pop(opt)

                layer_ = layer_cnstrctr(**layer_kwargs)
                layer = MBCollapse(layer_, pool_dim)
            elif is_mdlbatch and (layer_type in func_modulenames):
                #####################################
                ######## Layers Construction ########
                #####################################
                # Part I: The basic and fixed keyword arguments to the constructor
                layer_kwargs = dict()

                # Part II: Gathering the rest of the keyword arguments from the user-provided layer options
                for opt in list(layer_optseval.keys()):
                    layer_kwargs[opt] = layer_optseval.pop(opt)
                layer = FunctionalLayer(layer_type, layer_kwargs)
            else:
                layer_kwargs = dict()
                msg = f'''
                    The "{layer_type}" layer is not a well-defined type for 
                    atomic construction.'''
                raise ValueError(dedent(msg))
        except Exception as exc:
            layer_kwargs_str = pprint.pformat(layer_kwargs).replace("\n", "\n" + (" " * 20))
            exc.add_note(dedent(f'''
                This error happend when creating a "{layer_type}" layer with 
                the following options:
                    layer_kwargs = {layer_kwargs_str}
                '''))
            raise exc

        # Making sure no options were left unused
        if len(layer_optseval) > 0:
            raise ValueError(dedent(f'''
                After creating a "{layer_type}" layer, Some options were left unused. Make 
                sure you do not specify these, or specify them in a proper manner: 
                    unused options = {layer_optseval}'''))

        return layer

    @classmethod
    def make_encyclopedia(cls, lgstcs_dict):
        module_families = cls.get_families()

        encyclopedia = odict()
        for fam_name, layer_types in module_families.items():
            layer_makers = odict()
            for layer_type in layer_types:
                layer_makers[layer_type] = cls(layer_type, lgstcs_dict)
            encyclopedia[fam_name] = layer_makers

        return encyclopedia


class ConfigModuleMaker:
    def __init__(self, module_name, module_config, lgstcs_dict,
                 module_encyclopedia):
        self.module_name = module_name
        self.module_config = module_config
        self.lgstcs_dict = lgstcs_dict
        self.module_encyclopedia = module_encyclopedia

    def __call__(self, **hparams):
        module_name = self.module_name
        # Taking a shallow copy
        module_config = odict(self.module_config)
        lgstcs_dict = self.lgstcs_dict
        module_encyclopedia = self.module_encyclopedia

        if len(hparams) > 0:
            assert 'hparams' in module_config, dedent(f'''
                When creating a "{module_name}" module, some hyper-parameters 
                were provided for construction, but the module did not expect 
                any kind of hyper-params in its definition.
                    hparams = {hparams}
                    module_config = {module_config}''')

            hparams_config = module_config['hparams']
            assert isinstance(hparams_config, dict), dedent(f'''
                When creating a "{module_name}" module, the hyper-parameters 
                must be provided in keyword style (i.e., a dictionary with 
                the hyper-parameter keys and values). However, hparams had 
                a type of "{type(hparams)}":
                    hparams = {hparams}''')

        for key, val in hparams.items():
            assert key in hparams_config, dedent(f'''
                When creating a "{module_name}" module, the hyper-parameters 
                "{key}" was passed, but this module does not expect this 
                hyper-parameter in its definition":
                    hparams = {hparams}
                    module_config = {module_config}''')
            hparams_config[key] = val

        # Creating an instance of
        config_module = ConfigModule(module_name=module_name, module_config=module_config,
                                     lgstcs_dict=lgstcs_dict, module_encyclopedia=module_encyclopedia, pop_opts=True)

        return config_module


class RepeatLayerMaker:
    def __init__(self, layer_type, module_encyclopedia):
        assert layer_type in ('repeat', 'dense')
        self.layer_type = layer_type
        self.module_encyclopedia = module_encyclopedia
        self.module_families = self.get_families()

    @staticmethod
    def get_families():
        return {'repeat': ['repeat', 'dense']}

    def __call__(self, **layer_optseval):
        layer_type = self.layer_type
        module_encyclopedia = self.module_encyclopedia

        repeat_module = 'RepeatLayer'
        assert repeat_module in ('RepeatLayer', 'DenseLayer')

        if (layer_type == 'repeat') and (repeat_module == 'RepeatLayer'):
            n_reps = layer_optseval.pop('n_reps')
            module = layer_optseval.pop('module')
            hpdefs = layer_optseval.pop('hpdefs')
            hparams = layer_optseval
            layer_optseval = tuple()

            layer = RepeatLayer(n_reps=n_reps, module=module, hparams=hparams,
                                hpdefs=hpdefs, module_encyclopedia=module_encyclopedia)
        elif (layer_type == 'repeat') and (repeat_module == 'DenseLayer'):
            n_reps = layer_optseval.pop('n_reps')
            module = layer_optseval.pop('module')
            hpdefs = layer_optseval.pop('hpdefs')
            hparams = layer_optseval
            layer_optseval = tuple()

            layer = DenseLayer(n_reps=n_reps, n_hist=1, catdim=None,
                               channel=None, module=module, hparams=hparams, hpdefs=hpdefs,
                               module_encyclopedia=module_encyclopedia)
        elif layer_type == 'dense':
            n_reps = layer_optseval.pop('n_reps')
            n_hist = layer_optseval.pop('n_hist', n_reps)
            catdim = layer_optseval.pop('catdim')
            channel = layer_optseval.pop('channel')
            module = layer_optseval.pop('module')
            hpdefs = layer_optseval.pop('hpdefs')
            hparams = layer_optseval
            layer_optseval = tuple()

            layer = DenseLayer(n_reps=n_reps, n_hist=n_hist, catdim=catdim,
                               channel=channel, module=module, hparams=hparams, hpdefs=hpdefs,
                               module_encyclopedia=module_encyclopedia)
        else:
            msg = f'''
                The "{layer_type}" layer is not a well-defined type for 
                atomic construction.'''
            raise ValueError(dedent(msg))

        # Making sure no options were left unused
        if len(layer_optseval) > 0:
            raise ValueError(dedent(f'''
                Some options were left unused. Make sure you do not specify these, 
                or specify them in a proper manner: 
                    unused options = {layer_optseval}'''))

        return layer

    @classmethod
    def grow_encyclopedia(cls, module_encyclopedia):
        module_families = cls.get_families()

        for fam_name, layer_types in module_families.items():
            layer_makers = odict()
            for layer_type in layer_types:
                layer_makers[layer_type] = cls(layer_type, module_encyclopedia)
            module_encyclopedia[fam_name] = layer_makers

#######################################
######### User-Level Functions ########
#######################################

def make_nn(name, hparams, config, n_seeds, rng, device, dtype):
    '''
    Takes a module name (`name`) and hyper-parameters (`hparams`)
    along with the module definitions dictionary (`config`), and
    creates an instance of the neural network.

    Args:
        name (str): The instantiated module name

        hparams (dict): The instantiated module hyper-parameters

        config (dict): The module definitions dictionary. Must be
            a deep dict rather than a hierarchical one.

        n_seeds (int): The number of seeds / parallelized models.

        rng (BatchRNG): The batch random number generator.

        device (torch.device): The pytorch device for the model
            to be housed in.

        device (torch.device): The pytorch float data type.

    Returns:
        net (nn.Module): an instantiated neural model ready
            to be trained.
    '''
    hparams = dict() if hparams is None else hparams

    lgstcs_dict = dict(shape=(n_seeds,), device=device,
                       dtype=dtype, rng=rng)

    module_encyclopedia = AtomicLayerMaker.make_encyclopedia(lgstcs_dict)
    RepeatLayerMaker.grow_encyclopedia(module_encyclopedia)

    config_hie = deep2hie(config, dictcls=odict, sep="/")
    config_deep = hie2deep(config_hie, dictcls=odict, sep="/")

    all_modulemakers = {module_name: module_maker
                        for module_fam, module_makerdict in module_encyclopedia.items()
                        for module_name, module_maker in module_makerdict.items()}

    module_encyclopedia['config'] = dict()
    for module_name, module_config in config_deep.items():
        '''
        Example:
          module_name = 'slcnn'

          module_config = yaml.safe_load("""
            hparams:
              k: null
              n_inpch: null
              n_outch: null
              padding: same
              stride: 1
            layers:
              01_conv:
                bias: true
                in_channels: n_inpch
                kernel_size: k
                out_channels: n_outch
                padding: padding
                padding_mode: zeros
                stride: stride
                type: conv1d
              02_bn:
                num_features: n_outch
                type: batchnorm1d
              03_act:
                type: relu """)
        '''
        assert module_name not in all_modulemakers, dedent(f'''
            The module name "{module_name}" is already used as a basic 
            module name. Please specify a non-general name for the module.''')
        module_maker = ConfigModuleMaker(module_name, module_config, lgstcs_dict,
                                         module_encyclopedia)
        module_encyclopedia['config'][module_name] = module_maker

    all_modulemakers = {module_name: module_maker
                        for module_fam, module_makerdict in module_encyclopedia.items()
                        for module_name, module_maker in module_makerdict.items()}

    assert name in all_modulemakers, dedent(f'''
        The module "{module_name}" was not defined in the configs.
        
        * The following are the defined configurable module names:
            config_modules = {list(module_encyclopedia['config'].keys())}

        * The following are all the defined module names:
            all_modules = {list(all_modulemakers.keys())}
        ''')

    module_maker = all_modulemakers[name]

    net = module_maker(**hparams)

    return net

#######################################
######## Unit Testing Functions #######
#######################################

def get_testrands(n_rands, dstr, device, dtype):
    np_random = np.random.RandomState(seed=12345)
    balanced_32bit_seed = np_random.randint(
        0, 2**31-1, dtype=np.int32)
    rng = torch.Generator(tch_device)
    rng.manual_seed(int(balanced_32bit_seed))
    rvslst = [torch.empty(nrand, device=device, dtype=dtype)
        for nrand in n_rands]
    if dstr == 'normal':
        rvslst = [x.normal_(generator=rng) for x in rvslst]
    elif dstr == 'uniform':
        rvslst = [x.uniform_(generator=rng) for x in rvslst]
    else:
        raise ValueError(f'undefined dstr={dstr}')
    rvs = torch.cat(rvslst, dim=0)
    return rvs


def get_testbatchrands(n_rands, dstr, n_cache, n_seeds, device, dtype):
    rng_seeds = np.arange(0, n_seeds*1000, 1000)
    rng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=device, dtype=dtype,
        unif_cache_cols=n_cache,
        norm_cache_cols=n_cache)
    rng.seed(rng_seeds)

    if dstr == 'normal':
        rvslst = [rng.randn(n_seeds, n) for n in n_rands]
    elif dstr == 'uniform':
        rvslst = [rng.rand(n_seeds, n) for n in n_rands]
    else:
        raise ValueError(f'undefined dstr={dstr}')
    rvs = torch.cat(rvslst, dim=1)
    return rvs


###############################################################################
###################### Batch Optimizer annd LR Scheduler ######################
###############################################################################

class BatchParamGrouper:
    def __init__(self, params: list | dict | tuple, shape: tuple):
        """
        This class:
            (1) gets a `params` argument that you would have passed 
                to a `torch.optim.*` instance, and 
            (2) splits all of the nn.parameters along their first 
                dimension into n_seeds parts and detaches them.

        The `params` argument can be one of five structures:
            (1) list[torch.tensor]
            (2) dict[str: torch.tensor]
            (3) {'params': list[torch.tensor], **group_hparams}
            (4) list[{'params': list[torch.tensor], **group_hparams}]
            (5) list[{'params': dict[str: torch.tensor], **group_hparams}]
        """
        assert len(shape) == 1, 'non-singleton shapes are not supported yet'
        (n_seeds,) = shape

        self.attach_queue = []

        # Splitting the parameters along the seed
        params_singletype = self.is_single(params)
        params_grouptype = self.is_group(params)
        if params_singletype in ('list', 'dict'):
            n_grps = 1
            optim_params = self.get_optimparams(params, None, n_seeds)
        elif params_grouptype in ('list', 'dict'):
            # Example:
            #   params = {'params': [tnsr_1, tnsr2, ..., tnsr_k], **group_hparams}
            #   params_sngltn = [tnsr_1, tnsr2, ..., tnsr_k]
            if params_grouptype == 'list':
                params_lst = params
            elif params_grouptype == 'dict':
                params_lst = [params]
            else:
                raise ValueError(f'undefined params_grouptype = {params_grouptype}')
            n_grps = len(params_lst)
            optim_params = []
            for params_ in params_lst:
                params_cp = dict(params_)
                params_sngltn = params_cp.pop('params')
                optim_params += self.get_optimparams(params_sngltn, params_cp, n_seeds)
        else:
            assert params_singletype == 'none'
            assert params_grouptype == 'none'
            raise ValueError(dedent("""
                The `params` argument can be one of five structures:
                    (1) list[torch.tensor]
                    (2) dict[str: torch.tensor]
                    (3) {'params': list[torch.tensor], **group_hparams}
                    (4) list[{'params': list[torch.tensor], **group_hparams}]
                    (5) list[{'params': dict[str: torch.tensor], **group_hparams}]
            
            """) + dedent(f"""
                The given `params` does not match any of those structures:
                    params = {params}
            """))

        # Example: 
        #   optim_params = [{'params': [tnsr_a1[0],         tnsr_a2[0],         ..., tnsr_ak[0]        ]},
        #                   {'params': [tnsr_a1[1],         tnsr_a2[1],         ..., tnsr_ak[1]        ]},
        #                   ...,
        #                   {'params': [tnsr_a1[n_seeds-1], tnsr_a2[n_seeds-1], ..., tnsr_ak[n_seeds-1]]},
        #
        #                   {'params': [tnsr_b1[0],         tnsr_b2[0],         ..., tnsr_bk[0]        ]},
        #                   {'params': [tnsr_b1[1],         tnsr_b2[1],         ..., tnsr_bk[1]        ]},
        #                   ...,
        #                   {'params': [tnsr_b1[n_seeds-1], tnsr_b2[n_seeds-1], ..., tnsr_bk[n_seeds-1]]}
        assert len(optim_params) == n_grps * n_seeds
        assert all(isinstance(val, dict) for val in optim_params)
        assert all('params' in val for val in optim_params)
        
        optim_params1 = np.array(optim_params)
        assert optim_params1.shape == (n_grps * n_seeds,)
        optim_params2 = optim_params1.reshape(n_grps, n_seeds)
        assert optim_params2.shape == (n_grps, n_seeds)
        optim_params3 = optim_params2.T
        assert optim_params3.shape == (n_seeds, n_grps)
        optim_params4 = optim_params3.ravel()
        assert optim_params4.shape == (n_seeds * n_grps,)
        self.param_groups = optim_params4.tolist()
        self.param_groups_arr = optim_params3
        self.n_seeds = n_seeds
        self.n_grps = n_grps
    
    def get_optimparams(self, params: list | tuple | dict, hparams: dict | None, n_seeds: int) -> list:
        params_singletype = BatchParamGrouper.is_single(params)
        hparams = hparams if hparams is not None else dict()
        assert isinstance(hparams, dict)
        if params_singletype == 'list':
            # Example:
            #   params = [tnsr_1, tnsr_2, ..., tnsr_k]
            split_params = [self.split_param(i, param, n_seeds) for i, param in enumerate(params)]
            # Example: 
            #   split_params = [[tnsr_1[0], tnsr_1[1], ..., tnsr_1[n_seeds-1]],
            #                   [tnsr_2[0], tnsr_2[1], ..., tnsr_2[n_seeds-1]],
            #                   ...,
            #                   [tnsr_k[0], tnsr_k[1], ..., tnsr_k[n_seeds-1]],]
            optim_params = [{'params': [tnsr_lst[i] for tnsr_lst in split_params], **hparams} for i in range(n_seeds)]
            # Example: 
            #   optim_params = [{'params': [tnsr_1[0],         tnsr_2[0],         ..., tnsr_k[0]        ]},
            #                   {'params': [tnsr_1[1],         tnsr_2[1],         ..., tnsr_k[1]        ]},
            #                   ...,
            #                   {'params': [tnsr_1[n_seeds-1], tnsr_2[n_seeds-1], ..., tnsr_k[n_seeds-1]]}]
        elif params_singletype == 'dict':
            # Example:
            #   params = {'name_1': tnsr_1, 'name_2': tnsr_2, ..., 'name_k': tnsr_k}
            split_params = [self.split_param(name, param, n_seeds) for name, param in params.items()]
            # Example: 
            #   split_params = [[tnsr_1[0], tnsr_1[1], ..., tnsr_1[n_seeds-1]],
            #                   [tnsr_2[0], tnsr_2[1], ..., tnsr_2[n_seeds-1]],
            #                   ...,
            #                   [tnsr_k[0], tnsr_k[1], ..., tnsr_k[n_seeds-1]],]
            optim_params = [{'params': [tnsr_lst[i] for tnsr_lst in split_params], **hparams} for i in range(n_seeds)]
            # Example: 
            #   optim_params = [{'params': [tnsr_1[0],         tnsr_2[0],         ..., tnsr_k[0]        ]},
            #                   {'params': [tnsr_1[1],         tnsr_2[1],         ..., tnsr_k[1]        ]},
            #                   ...,
            #                   {'params': [tnsr_1[n_seeds-1], tnsr_2[n_seeds-1], ..., tnsr_k[n_seeds-1]]}]
        else:
            raise ValueError(f'''
                The `params` argument to this method can be one of two structures:
                    (1) list[torch.tensor]
                    (2) dict[str: torch.tensor]
                
                However, I received:
                    params = {params}''')
        
        assert len(optim_params) == n_seeds
        
        return optim_params

    def split_param(self, name: str, param: torch.tensor, n_seeds: int):
        assert isinstance(name, str | int)
        assert isinstance(n_seeds, int)
        assert n_seeds > 0
        assert torch.is_tensor(param)
        assert param.shape[0] % n_seeds == 0
        
        # if isinstance(lr, float):
        #     lr_lst = [lr] * n_seeds
        # elif isinstance(lr, np.ndarray):
        #     lr_lst = lr.tolist()
        # elif isinstance(lr, (list, tuple)):
        #     lr_lst = list(lr)
        # elif torch.is_tensor(lr):
        #     lr_lst = lr.detach().cpu().numpy().tolist()
        # else:
        #     raise ValueError(f'undefined lr = {lr}')
        # assert len(lr_lst) == n_seeds

        if param.shape[0] == n_seeds:
            psplit = [param[i].detach() for i in range(n_seeds)]
        else:
            n_pgrps = param.shape[0] // n_seeds
            param_rshp = param.reshape(n_seeds, n_pgrps, *param.shape[1:])
            psplit = [param_rshp[i].detach() for i in range(n_seeds)]
        
        self.attach_queue.append((param, psplit))
        return psplit
    
    @staticmethod
    def is_single(params):
        is_list = isinstance(params, list | tuple) 
        if is_list:
            is_list = is_list and all(torch.is_tensor(val) for val in params)
        
        is_dict = isinstance(params, dict) 
        if is_dict:
            is_dict = is_dict and (all(isinstance(key, str) for key in params))
        if is_dict:
            is_dict = is_dict and all(torch.is_tensor(val) for val in params.values())
        
        if is_list:
            paramstype = 'list'
        elif is_dict:
            paramstype = 'dict'
        else:
            paramstype = 'none'
        return paramstype
    
    @staticmethod
    def is_group(params):
        is_list = isinstance(params, list | tuple) 
        if is_list:
            is_list = is_list and all(isinstance(val, dict) for val in params) 
        if is_list:
            is_list = is_list and all('params' in val for val in params) 
        if is_list:
            is_list = is_list and all(BatchParamGrouper.is_single(val['params']) in ('list', 'dict') for val in params)
        
        is_dict = isinstance(params, dict)
        if is_dict:
            is_dict = is_dict and ('params' in params)
        if is_dict:
            is_dict = is_dict and (BatchParamGrouper.is_single(params['params']) in ('list', 'dict'))
        
        if is_list:
            paramstype = 'list'
        elif is_dict:
            paramstype = 'dict'
        else:
            paramstype = 'none'
        return paramstype
        
    def attach_grads(self):
        n_seeds = self.n_seeds
        attach_queue = self.attach_queue
        for param, psplit in attach_queue:
            pgrad = param.grad
            if pgrad is None:
                pgrad_rshp = [None] * n_seeds
            elif pgrad.shape[0] == n_seeds:
                pgrad_rshp = pgrad
            else:
                n_pgrps = pgrad.shape[0] // n_seeds
                pgrad_rshp = pgrad.reshape(n_seeds, n_pgrps, *pgrad.shape[1:])
            for i in range(n_seeds):
                psplit[i].grad = pgrad_rshp[i]

class BatchLRScheduler:
    def __init__(self, kind: str, hparams: dict, optimizer: any, n_seeds: int, n_grps: int):
        # Finding the lr scheduler constructor
        schdtyplst = torch.optim.lr_scheduler.__all__
        schd_makers1 = {schdtyp:         getattr(torch.optim.lr_scheduler, schdtyp) for schdtyp in schdtyplst}
        schd_makers2 = {schdtyp.lower(): getattr(torch.optim.lr_scheduler, schdtyp) for schdtyp in schdtyplst}
        schd_makers = {**schd_makers1, **schd_makers2}
        assert kind in schd_makers, f'undefined lr scheduler type {kind}'
        schd_maker = schd_makers[kind]

        opt_maker = type(optimizer)
        param_groups = optimizer.param_groups

        assert len(param_groups) % n_seeds == 0
        n_grps = len(param_groups) // n_seeds
        param_groups2d = np.array(param_groups).reshape(n_seeds, n_grps)
        assert param_groups2d.shape == (n_seeds, n_grps)
        # Making fake and lite optimizers for the scheduler
        schedulers = []
        for seed_paramgrps in param_groups2d.tolist():
            # Example:
            #   seed_paramgrps = [{'params': [tnsr_a1[0],         tnsr_a2[0],         ..., tnsr_ak[0]        ]},
            #                     {'params': [tnsr_b1[0],         tnsr_b2[0],         ..., tnsr_bk[0]        ]}],
            assert len(seed_paramgrps) == n_grps
            param_grpfakes = []
            for param_grp in seed_paramgrps:
                assert isinstance(param_grp, dict)
                assert 'params' in param_grp
                hps_grp = dict(param_grp)
                hps_grp.pop('params')
                fake_tnsr = nn.Parameter(torch.zeros(1))
                param_grpfake = {'params': [fake_tnsr], **hps_grp} 
                param_grpfakes.append(param_grpfake)
            assert len(param_grpfakes) == n_grps
                                
            opt_fake = opt_maker(params=param_grpfakes)
            schdlr = schd_maker(opt_fake, **hparams)
            schedulers.append(schdlr)
        assert len(schedulers) == n_seeds

        schdtyp2args = {'LambdaLR': ('epoch',),
                        'MultiplicativeLR': ('epoch',),
                        'StepLR': ('epoch',),
                        'MultiStepLR': ('epoch',),
                        'ConstantLR': ('epoch',),
                        'LinearLR': ('epoch',),
                        'ExponentialLR': ('epoch',),
                        'SequentialLR': (),
                        'CosineAnnealingLR': ('epoch',),
                        'ChainedScheduler': (),
                        'ReduceLROnPlateau': ('metrics',),
                        'CyclicLR': ('epoch',),
                        'CosineAnnealingWarmRestarts': ('epoch',),
                        'OneCycleLR': ('epoch',),
                        'PolynomialLR': ('epoch',),
                        'LRScheduler': ('epoch',)}
        schdtyp2args = {**schdtyp2args, 
            **{key.lower(): val for key, val in schdtyp2args.items()}}
        step_args = schdtyp2args[kind]

        self.optimizer = optimizer
        self.schedulers = schedulers
        self.n_seeds = n_seeds
        self.n_grps = n_grps
        self.step_args = step_args
    
    def step(self, metrics=None, epoch=None):
        schedulers = self.schedulers
        param_groups = self.optimizer.param_groups
        n_seeds = self.n_seeds
        n_grps = self.n_grps
        step_args = self.step_args
        if metrics is not None:
            assert len(metrics) == n_seeds

        schdlr_lrs_lst = []
        for i_seed, schdlr in enumerate(schedulers):
            step_kws = dict()
            if 'metrics' in step_args:
                assert metrics is not None
                step_kws['metrics'] = metrics[i_seed]
            if 'epoch' in step_args:
                assert isinstance(epoch, int)
                step_kws['epoch'] = epoch
            schdlr.step(**step_kws)
            i_schdlr_lrs = schdlr.get_last_lr()
            assert isinstance(i_schdlr_lrs, list)
            assert len(i_schdlr_lrs) == n_grps
            assert all(isinstance(val, float) for val in i_schdlr_lrs)
            schdlr_lrs_lst.append(i_schdlr_lrs)
        schdlr_lrs2d = np.array(schdlr_lrs_lst, dtype=np.float64)
        assert schdlr_lrs2d.shape == (n_seeds, n_grps)
        schdlr_lrs1d = schdlr_lrs2d.reshape(n_seeds * n_grps).tolist()
        assert len(schdlr_lrs1d) == (n_seeds * n_grps)

        for i_grp in range(n_seeds * n_grps):
            param_group = param_groups[i_grp]
            new_lr = float(schdlr_lrs1d[i_grp])
            param_group["lr"] = new_lr
    
    def get_last_lr(self):
        schdlr_lrs_lst = []
        for i_seed, schdlr in enumerate(self.schedulers):
            schdlr_lrs_lst += schdlr.get_last_lr()
        return schdlr_lrs_lst

if __name__ == '__main__':
    #######################################
    ##### torch_quantile Unit Testing #####
    #######################################
    print('Performing torch_quantile unit tests...')
    test_quantile()
    print('Successful torch_quantile unit tests!')
    print('-' * 40)

    #######################################
    ##### torch_quantile Unit Testing #####
    #######################################
    print('Performing get_srtalgnidxs unit tests...')
    test_srtalgnidxs()
    print('Successful get_srtalgnidxs unit tests!')
    print('-' * 40)

    #######################################
    ######## Batch RNG Unit Testing #######
    #######################################

    from scipy import stats
    tch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tch_dtype = torch.float64
    # Checking the identity and the normality/uniformity of the torch.generator random variables
    for dstr in ['uniform', 'normal']:
        rvs1 = get_testrands([200, 6450, 6450, 400], dstr, tch_device, tch_dtype)
        rvs2 = get_testrands([900, 6100, 6100, 400], dstr, tch_device, tch_dtype)

        dstrstr = f'{dstr}'.capitalize() + ' ' * (7 - len(str(dstr)))
        cdf = {'normal': stats.norm.cdf, 'uniform': stats.uniform.cdf}[dstr]
        rvs_np = rvs1.detach().cpu().numpy().ravel()
        # Performing the Kolmogrov test to see if the RVs truly fit the theoretical CDF
        pval1 = stats.kstest(rvs_np, cdf).pvalue
        # Performing the Kolmogrov test to see if the KS-test can detect non-iid samples
        pval2 = stats.kstest((rvs_np[None, :] * np.ones((10, 1))).ravel(), cdf).pvalue

        eqprcnt = (rvs1 == rvs2).float().mean().item() * 100
        print(f'For the {dstrstr} distribution, the RVs were {eqprcnt:6.2f}% identical,', end=' ')
        print(f'KS-test p-value was {pval1:0.3f} (higher-better) ', end=' ')
        print(f'and the correlated p-value was {pval2:0.4f} (lower-better).')
    
    # Checking the same for the BatchRNG class
    for n_seeds, dstr, n_cache in product([1, 10], ['uniform', 'normal'], [0, 1, 10, 100, 1000, 10000, 100000]):
        rvs1 = get_testbatchrands([200, 6450, 6450, 400], dstr, n_cache, n_seeds, tch_device, tch_dtype)
        rvs2 = get_testbatchrands([900, 6100, 6100, 400], dstr, n_cache, n_seeds, tch_device, tch_dtype)
        eqprcnt = (rvs1 == rvs2).float().mean().item() * 100

        dstrstr = f'{dstr}'.capitalize() + ' ' * (7 - len(str(dstr)))
        ncachstr = f'{n_cache}'.capitalize() + ' ' * (6 - len(str(n_cache)))
        seedsstr = f'{n_seeds}' + ' ' * (2 - len(str(n_seeds)))

        cdf = {'normal': stats.norm.cdf, 'uniform': stats.uniform.cdf}[dstr]
        rvs_np = rvs1.detach().cpu().numpy().ravel()
        # Performing the Kolmogrov test to see if the RVs truly fit the theoretical CDF
        pval1 = stats.kstest(rvs_np, cdf).pvalue
        # Performing the Kolmogrov test to see if the KS-test can detect non-iid samples
        pval2 = stats.kstest((rvs_np[None, :] * np.ones((10, 1))).ravel(), cdf).pvalue
        
        print(f'{dstrstr}, {ncachstr} columns, {seedsstr} seeds -> {eqprcnt:06.2f}% identity, ' + 
            f'{pval1:0.3f} p-value (higher-better), {pval2:0.4f} correlated p-value (lower-better).')