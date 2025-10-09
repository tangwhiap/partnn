import math
import numpy as np
import torch
###########################################################
###### The `dlnpsi` Psi Downward Recurrence Function ######
###########################################################

def get_dlnpsi(z, n_max, device, cdtype):
    """
    Downward recurrence for the logarithmic derivative of psi (psi'/psi).

    This function is a tensor implementation of the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L30

    Parameters
    ----------
    z: (torch.tensor) The input z array with any shape.

    n_max: (int) The number of iterations for convergence.

    device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    Returns
    -------
    eta: (torch.tensor) The computed eta values.
        `assert eta.shape == (*z.shape, n_max)`
    """
    z_shape = z.shape
    assert z.shape == (*z_shape,)

    eta_p20 = torch.zeros(*z_shape, n_max + 20, device=device, dtype=cdtype)
    assert eta_p20.shape == (*z_shape, n_max + 20)

    for n_now in range(n_max + 18, 0, -1):
        term1 = (n_now + 1) / z
        assert term1.shape == (*z_shape,)

        term2_inv = term1 + eta_p20[..., n_now + 1]
        assert term2_inv.shape == (*z_shape,)

        eta_p20[..., n_now] = term1 - 1.0 / term2_inv

    eta = eta_p20[..., :n_max]
    assert eta.shape == (*z_shape, n_max)

    return eta

def get_dlnpsirefnp(z_np, n_max):
    """
    The numpy version of the `get_dlnpsi` function. The code is mainly adopted from
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L30

    Parameters
    ----------
    z_np: (float | np.float) The input scalar z.

    n_max: (int) The number of iterations for convergence.

    Returns
    -------
    eta: (torch.tensor) The computed eta values.
        `assert eta.shape == (n_max,)`
    """
    eta = np.zeros((n_max + 20), dtype='cdouble')
    for n in range(n_max + 20 - 2, 0, -1):
        eta[n] = (n + 1) / z_np - 1 / ((n + 1) / z_np + eta[n + 1])
    return eta[:n_max]

def test_dlnpsizeta(func_tch, func_npref, func_name, verbose=False):
    """
    This is a unit test for the torch and numpy version of the `dlnpsi` function.
    """
    n_max, n_partset, n_part = 120, 10, 100
    np_random = np.random.RandomState(12345)

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    z_tstrealnp = np_random.randn(n_partset, n_part)
    assert z_tstrealnp.shape == (n_partset, n_part)
    z_tstimagnp = np_random.randn(n_partset, n_part)
    assert z_tstimagnp.shape == (n_partset, n_part)
    z_tstnp = z_tstrealnp + 1j * z_tstimagnp
    assert z_tstnp.shape == (n_partset, n_part)
    z_tst = torch.tensor(z_tstnp, device=tch_device, dtype=tch_cdtype)
    assert z_tst.shape == (n_partset, n_part)

    if verbose:
        print(f'Testing the {func_name} routine with ' +
              f'{n_partset*n_part} particles and n_max={n_max}:')

    eta_pred = func_tch(z_tst, n_max, tch_device, tch_cdtype)
    assert eta_pred.shape == (n_partset, n_part, n_max)

    eta_refnp = np.array([func_npref(z.item(), n_max) for z in z_tstnp.ravel()]
        ).reshape(n_partset, n_part, n_max)
    assert eta_refnp.shape == (n_partset, n_part, n_max)

    eta_ref = torch.from_numpy(eta_refnp).to(device=tch_device, dtype=tch_cdtype)
    assert eta_ref.shape == (n_partset, n_part, n_max)

    pnorm_err = 2
    eta_abserr = (eta_pred - eta_ref).reshape(n_partset, n_part, n_max).abs().pow(
        pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
    assert eta_abserr.shape == (n_partset, n_part)

    eta_relerr = 0.5 * eta_abserr / (
        eta_pred.reshape(n_partset, n_part, n_max).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
        eta_ref.reshape(n_partset, n_part, n_max).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
    eta_relerr[eta_abserr == 0.0] = 0.0
    assert eta_relerr.shape == (n_partset, n_part)

    eta_abserrmax = eta_abserr.max().item()
    eta_abserrmean = eta_abserr.mean().item()
    eta_relerrmax = eta_relerr.max().item()
    eta_relerrmean = eta_relerr.mean().item()

    if verbose:
        print(f'  eta       : Max Abs Error={eta_abserrmax:0.03g}, ' +
              f'Mean Abs Error={eta_abserrmean:0.03g}, Max Rel Error={eta_relerrmax:0.03g}, ' +
              f'Mean Rel Error={eta_relerrmean:0.03g}')

    assert eta_abserrmax < 1e-8
    assert eta_abserrmean < 1e-10

###########################################################
###### The `dlnzeta` Zeta Upward Recurrence Function ######
###########################################################

def get_dlnzeta(z, n_max, device, cdtype):
    """
    The upward recurrence for the logarithmic derivative of zeta (zeta'/zeta).

    This function is a tensor implementation of the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L38

    Parameters
    ----------
    z: (torch.tensor) The input z array with any shape.

    n_max: (int) The number of iterations for convergence.

    device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    Returns
    -------
    eta: (torch.tensor) The computed eta values.
        `assert eta.shape == (*z.shape, n_max)`
    """
    z_shape = z.shape
    assert z.shape == (*z_shape,)

    eta = torch.zeros(*z_shape, n_max, device=device, dtype=cdtype)
    assert eta.shape == (*z_shape, n_max)

    eta[..., 0] = -1j

    for n_now in range(1, n_max):
        term1 = n_now / z
        assert term1.shape == (*z_shape,)

        term2_inv = term1 - eta[..., n_now - 1]
        assert term2_inv.shape == (*z_shape,)

        eta[..., n_now] =  - term1 + 1.0 / term2_inv

    return eta

def get_dlnzetarefnp(z, n_max):
    """
    The upward recurrence for the logarithmic derivative of zeta (zeta'/zeta).

    This function is mainly adopted from the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L38

    Parameters
    ----------
    z_np: (float | np.float) The input scalar z.

    n_max: (int) The number of iterations for convergence.

    Returns
    -------
    eta: (torch.tensor) The computed eta values.
        `assert eta.shape == (n_max,)`
    """
    eta = np.zeros((n_max), dtype='cdouble')
    eta[0] = -1j
    for n in range(1, n_max):
        eta[n] = - n / z + 1 / (n / z - eta[n-1])
    return eta

###########################################################
###### The `tup` T Ratio Upward Recurrence Function #######
###########################################################

def get_tup(z, dp, dz, device, dtype, cdtype):
    """
    The upward recurrence for the ratio (`T = psi / zeta`).

    This function is a tensor implementation of the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L47

    Parameters
    ----------
    z: (torch.tensor) The input z array with any shape.
        `assert z.shape == (n_partset, n_part)`

    dp: (torch.tensor) The input dp.
        `assert dp.shape == (n_partset, n_part, n_max)`

    dz: (torch.tensor) The input dz.
        `assert dz.shape == (n_partset, n_part, n_max)`

    device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    Returns
    -------
    T: (torch.tensor) The computed T values.
        `assert T.shape == (n_partset, n_part, n_max)`
    """
    *z_shape, n_max = dp.shape
    assert z.shape == (*z_shape,)
    assert dp.shape == (*z_shape, n_max)
    assert dz.shape == (*z_shape, n_max)
    oz_shape = tuple(1 for _ in z_shape)

    # Creating an n-range
    n_arange1 = torch.arange(2, n_max, device=device, dtype=dtype)
    assert n_arange1.shape == (n_max - 2,)

    n_arange = n_arange1.reshape(*oz_shape, n_max - 2)
    assert n_arange.shape == (*oz_shape, n_max - 2)

    # Dividing n by z
    ndivz = n_arange / z.unsqueeze(-1)
    assert ndivz.shape == (*z_shape, n_max - 2)

    ###########################################################
    ##################### Preparing Alpha #####################
    ###########################################################
    alpha = torch.empty(*z_shape, n_max, device=device, dtype=cdtype)
    assert alpha.shape == (*z_shape, n_max)

    # The first column should be one to facilitate a delayed cumprod application.
    alpha[..., 0] = 1.0

    # There is a special formula for the initial column.
    alpha[..., 1] = 0.5 + (0.5 * (z * 2j).exp() * (z + 1j) / (z - 1j))

    # The rest of the formula
    alpha[..., 2:] = (ndivz - dp[..., 1:-1]) / (ndivz - dz[..., 1:-1])

    ###########################################################
    ####################### Preparing T #######################
    ###########################################################
    T = alpha.cumprod(dim=-1)
    assert T.shape == (*z_shape, n_max)

    T[..., 0] = 0

    return T

def get_tuprefnp(z_np, dp_np, dz_np):
    """
    The upward recurrence for the ratio (`T = psi / zeta`).

    This function is mainly adopted from the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L47

    Parameters
    ----------
    z_np: (float | np.float) The input scalar z.

    dp_np: (np.ndarray) The input dp.
        `assert dp_np.shape == (n_max,)`

    dz_np: (np.ndarray) The input dz.
        `assert dz_np.shape == (n_max,)`

    Returns
    -------
    T: (np.ndarray) The computed eta values.
        `assert T.shape == (n_max,)`
    """
    T = np.zeros((len(dp_np)), dtype='cdouble')
    T[1] = 0.5 * np.exp(2 * 1j * z_np) * (z_np + 1j) / (z_np - 1j) + 0.5
    for n in range(2, len(dp_np)):
        T[n] = T[n-1] * (n / z_np - dp_np[n-1]) / (n / z_np - dz_np[n-1])
    return T

def toss_cmplxrand(np_random, shape, tch_device, tch_cdtype):
    """
    This is a utility function for tossing a bunch of complex random
    numbers from a numpy RNG and sending them to pytorch.
    """
    z_tstrealnp = np_random.randn(*shape)
    assert z_tstrealnp.shape == shape
    z_tstimagnp = np_random.randn(*shape)
    assert z_tstimagnp.shape == shape
    z_tstnp = z_tstrealnp + 1j * z_tstimagnp
    assert z_tstnp.shape == shape
    z_tst = torch.tensor(z_tstnp, device=tch_device, dtype=tch_cdtype)
    assert z_tst.shape == shape
    return z_tst, z_tstnp

def test_tup(verbose=False):
    """
    This is a unit test for the torch and numpy version of the `dlnpsi` function.
    """
    n_max, n_partset, n_part = 120, 10, 100
    np_random = np.random.RandomState(12345)

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    z_tst, z_tstnp = toss_cmplxrand(np_random, (n_partset, n_part),
                                    tch_device, tch_cdtype)
    assert z_tst.shape == (n_partset, n_part)
    assert z_tstnp.shape == (n_partset, n_part)

    dp_tst, dp_tstnp = toss_cmplxrand(np_random, (n_partset, n_part, n_max),
                                      tch_device, tch_cdtype)
    assert dp_tst.shape == (n_partset, n_part, n_max)
    assert dp_tstnp.shape == (n_partset, n_part, n_max)

    dz_tst, dz_tstnp = toss_cmplxrand(np_random, (n_partset, n_part, n_max),
                                      tch_device, tch_cdtype)
    assert dz_tst.shape == (n_partset, n_part, n_max)
    assert dz_tstnp.shape == (n_partset, n_part, n_max)

    if verbose:
        print('Testing the tup routine with ' +
              f'{n_partset*n_part} particles and n_max={n_max}:')

    t_pred = get_tup(z_tst, dp_tst, dz_tst, tch_device, tch_dtype, tch_cdtype)
    assert t_pred.shape == (n_partset, n_part, n_max)

    t_refnp = np.array([get_tuprefnp(z_tstnp[i, j], dp_tstnp[i, j], dz_tstnp[i, j])
        for i in range(n_partset) for j in range(n_part)]
        ).reshape(n_partset, n_part, n_max)
    assert t_refnp.shape == (n_partset, n_part, n_max)

    t_ref = torch.from_numpy(t_refnp).to(device=tch_device, dtype=tch_cdtype)
    assert t_ref.shape == (n_partset, n_part, n_max)

    pnorm_err = 2
    t_abserr = (t_pred - t_ref).reshape(n_partset, n_part, n_max).abs().pow(
        pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
    assert t_abserr.shape == (n_partset, n_part)

    t_relerr = 0.5 * t_abserr / (
        t_pred.reshape(n_partset, n_part, n_max).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
        t_ref.reshape(n_partset, n_part, n_max).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
    t_relerr[t_abserr == 0.0] = 0.0
    assert t_relerr.shape == (n_partset, n_part)

    t_abserrmax = t_abserr.max().item()
    t_abserrmean = t_abserr.mean().item()
    t_relerrmax = t_relerr.max().item()
    t_relerrmean = t_relerr.mean().item()

    if verbose:
        print(f'  T         : Max Abs Error={t_abserrmax:0.03g}, ' +
              f'Mean Abs Error={t_abserrmean:0.03g}, Max Rel Error={t_relerrmax:0.03g}, ' +
              f'Mean Rel Error={t_relerrmean:0.03g}')

    assert t_abserrmax < 1e-8
    assert t_abserrmean < 1e-10

###########################################################
#### The `qqg` Q efficiencies and Asymmetry Parameter #####
###########################################################
def get_qqg(a, b, x, device, dtype):
    """
    Calculating the extinction and scattering efficiencies and
    asymmetry parameter from Mie coefficients.

    This function is a tensor implementation of the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L56

    Parameters
    ----------
    a: (torch.tensor) The input a array.
        `assert a.shape == (n_partset, n_part, n_max)`

    b: (torch.tensor) The input b array.
        `assert b.shape == (n_partset, n_part, n_max)`

    x: (torch.tensor) The input shell size parameter.
        `assert x.shape == (n_partset, n_part)`

    device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    Returns
    -------
    qe: (torch.tensor) The extinction Q efficiencies.
        `assert qe.shape == (n_partset, n_part)`

    qs: (torch.tensor) The scattering Q efficiencies.
        `assert qs.shape == (n_partset, n_part)`

    g: (torch.tensor) The g asymmetry parameter.
        `assert g.shape == (n_partset, n_part)`
    """
    *x_shape, n_max = a.shape
    assert a.shape == (*x_shape, n_max)
    assert b.shape == (*x_shape, n_max)
    assert x.shape == (*x_shape,)
    ox_shape = tuple(1 for _ in x_shape)

    n_arange1 = torch.arange(1, n_max, device=device, dtype=dtype)
    assert n_arange1.shape == (n_max - 1,)

    n_arange = n_arange1.reshape(*ox_shape, n_max - 1)
    assert n_arange.shape == (*ox_shape, n_max - 1)

    apb_real = (a.real + b.real)
    assert apb_real.shape == (*x_shape, n_max)

    x_square = x.square()
    assert x_square.shape == (*x_shape,)

    a2pb2_real = (a.real.square() + b.real.square() + a.imag.square() + b.imag.square())
    assert a2pb2_real.shape == (*x_shape, n_max)

    qe = 2 * ((2 * n_arange + 1) * apb_real[..., :-1]).sum(dim=-1) / x_square
    assert qe.shape == (*x_shape,)

    qs = 2 * ((2 * n_arange + 1) * a2pb2_real[..., :-1]).sum(dim=-1) / x_square
    assert qs.shape == (*x_shape,)

    g_term1 = ((n_arange + 2) / (1 + 1 / n_arange)) * (a[..., :-1] * a[..., 1:].conj() + 
        b[..., :-1] * b[..., 1:].conj()).real
    assert g_term1.shape == (*x_shape, n_max - 1)

    g_term2 = ((2 + 1 / n_arange) / (n_arange + 1)) * (a * b.conj()).real[..., :-1]
    assert g_term2.shape == (*x_shape, n_max - 1)

    g = 4 * (g_term1 + g_term2).sum(dim=-1) / (qs * x_square)
    assert g.shape == (*x_shape,)

    return qe, qs, g

def get_qqgnpref(a_np, b_np, x_np):
    """
    Calculating the extinction and scattering efficiencies and
    asymmetry parameter from Mie coefficients.

    This function is a mainly adopted from the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L56

    Parameters
    ----------
    a_np: (np.ndarray) The input a array.
        `assert a.shape == (n_max,)`

    b_np: (np.ndarray) The input b array.
        `assert b.shape == (n_max,)`

    x_np: (np.float) The input shell size parameter.

    Returns
    -------
    qe: (np.float) The extinction Q efficiencies.

    qs: (np.float) The scattering Q efficiencies.

    g: (np.float) The g asymmetry parameter.
    """
    n = np.arange(1, len(a_np))
    qe = 2 * np.sum( (2*n+1) * (a_np+b_np).real[:-1] ) / (x_np**2)
    qs = 2 * np.sum( (2*n+1) * (a_np.real**2 + a_np.imag**2 + b_np.real**2 + b_np.imag**2)[:-1]) / (x_np**2)
    g = 4 * np.sum(((n+2) / (1+1/n)) * (a_np[:-1] * np.conj(a_np[1:]) + b_np[:-1] * np.conj(b_np[1:])).real +
        ((2+1/n) / (n+1)) * (a_np * np.conj(b_np)).real[:-1]) / (qs * x_np**2)
    return qe, qs, g

def test_qqg(verbose=False):
    """
    This is a unit test for the torch and numpy version of the `qqg` function.
    """
    n_max, n_partset, n_part = 120, 10, 100
    np_random = np.random.RandomState(12345)

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    x_tstnp = np_random.randn(n_partset, n_part)
    assert x_tstnp.shape == (n_partset, n_part)
    x_tst = torch.from_numpy(x_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert x_tst.shape == (n_partset, n_part)

    a_tst, a_tstnp = toss_cmplxrand(np_random, (n_partset, n_part, n_max),
                                    tch_device, tch_cdtype)
    assert a_tst.shape == (n_partset, n_part, n_max)
    assert a_tstnp.shape == (n_partset, n_part, n_max)

    b_tst, b_tstnp = toss_cmplxrand(np_random, (n_partset, n_part, n_max),
                                    tch_device, tch_cdtype)
    assert b_tst.shape == (n_partset, n_part, n_max)
    assert b_tstnp.shape == (n_partset, n_part, n_max)

    vars_spec = {
        'qe': {'dim': tuple(), 'dtype': tch_dtype},
        'qs': {'dim': tuple(), 'dtype': tch_dtype},
        'g': {'dim': tuple(), 'dtype': tch_dtype}}

    vars_pred = get_qqg(a_tst, b_tst, x_tst, tch_device, tch_dtype)

    if verbose:
        print('Testing the qqg routine with ' +
              f'{n_partset*n_part} particles and n_max={n_max}:')

    for vname, vpred in zip(vars_spec, vars_pred):
        vinfo = vars_spec[vname]
        vdim = vinfo['dim']
        assert vpred.shape == (n_partset, n_part, *vdim)
        assert torch.is_tensor(vpred)
        vinfo['pred'] = vpred
        vinfo['reflst'] = []

    for i in range(n_partset):
        for j in range(n_part):
            vars_refij = get_qqgnpref(a_tstnp[i, j], b_tstnp[i, j], x_tstnp[i, j])

            for vname, vrefij in zip(vars_spec, vars_refij):
                vinfo = vars_spec[vname]
                vdim = vinfo['dim']
                assert vrefij.shape == (*vdim,)
                vinfo['reflst'].append(vrefij)

    for vname, vinfo in vars_spec.items():
        vdim = vinfo['dim']
        vdtype = vinfo['dtype']
        vpred = vinfo['pred']

        vrefnp = np.array(vinfo['reflst']).reshape(n_partset, n_part, *vdim)
        assert vrefnp.shape == (n_partset, n_part, *vdim)

        vref = torch.from_numpy(vrefnp).to(device=tch_device, dtype=vdtype)
        assert vref.shape == (n_partset, n_part, *vdim)
        vinfo['ref'] = vref

        pnorm_err = 2

        v_abserr = (vpred - vref).reshape(n_partset, n_part, math.prod(vdim)).abs().pow(
            pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
        assert v_abserr.shape == (n_partset, n_part)

        v_relerr = 2.0 * v_abserr / (
            vpred.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
            vref.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
        v_relerr[v_abserr == 0.0] = 0.0
        assert v_relerr.shape == (n_partset, n_part)

        v_abserrmax = v_abserr.max().item()
        v_abserrmean = v_abserr.mean().item()
        v_relerrmax = v_relerr.max().item()
        v_relerrmean = v_relerr.mean().item()

        vinfo['abserrmax'] = v_abserrmax
        vinfo['abserrmean'] = v_abserrmean
        vinfo['relerrmax'] = v_relerrmax
        vinfo['relerrmean'] = v_relerrmean

        if verbose:
            print(f'  {vname + " " * (10-len(vname))}: Max Abs Error={v_abserrmax:0.03g}, ' +
                  f'Mean Abs Error={v_abserrmean:0.03g}, Max Rel Error={v_relerrmax:0.03g}, ' +
                  f'Mean Rel Error={v_relerrmean:0.03g}')

    for vname, vinfo in vars_spec.items():
        v_relerrmax = vinfo['relerrmax']
        assert v_relerrmax < 1e-8, f'{vname} Max Rel Error = {v_relerrmax:0.03g}'
        v_relerrmean = vinfo['relerrmean']
        assert v_relerrmean < 1e-10, f'{vname} Mean Rel Error = {v_abserrmean:0.03g}'

###########################################################
####### The `sphere` Mie Calculations for a Sphere ########
###########################################################

def get_sphere(m, x, n_max, device, dtype, cdtype):
    """
    Calculating the Mie efficeincies for a sphere model.

    This function is a tensor implementation from the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L66

    Parameters
    ----------
    m: (torch.tensor) The complex refraction coefficient.
        `assert m.shape == (n_partset, n_part)`

    x: (torch.tensor) The particle size parameter at the specific wave-length.
        `assert x.shape == (n_partset, n_part)`

    n_max: (int) The number of iterations for convergence.

    device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    Returns
    -------
    qe: (torch.tensor) The extinction Q efficiencies.
        `assert qe.shape == (n_partset, n_part)`

    qs: (torch.tensor) The scattering Q efficiencies.
        `assert qs.shape == (n_partset, n_part)`

    g: (torch.tensor) The g asymmetry parameter.
        `assert g.shape == (n_partset, n_part)`
    """
    x_shape = x.shape
    assert x.shape == x_shape
    assert m.shape == x_shape

    # Ensuring `m` follows the right sign convention in TAMie
    m_sgnd = m.real - 1j * m.imag.abs()
    assert m_sgnd.shape == x_shape
    m_unsqz = m_sgnd.unsqueeze(-1)
    assert m_unsqz.shape == (*x_shape, 1)

    dpx = get_dlnpsi(x, n_max, device, cdtype)
    assert dpx.shape == (*x_shape, n_max)
    dpmx = get_dlnpsi(m_sgnd * x, n_max, device, cdtype)
    assert dpmx.shape == (*x_shape, n_max)
    dzx = get_dlnzeta(x, n_max, device, cdtype)
    assert dzx.shape == (*x_shape, n_max)
    T = get_tup(x, dpx, dzx, device, dtype, cdtype)
    assert T.shape == (*x_shape, n_max)
    an = (T * (m_unsqz * dpx - dpmx) / (m_unsqz * dzx - dpmx))[..., 1:]
    assert an.shape == (*x_shape, n_max-1)
    bn = (T * (dpx - m_unsqz * dpmx) / (dzx - m_unsqz * dpmx))[..., 1:]
    assert an.shape == (*x_shape, n_max-1)

    return get_qqg(an, bn, x, device, dtype)

def get_sphererefnp(m, x, n_max):
    """
    Calculating the Mie efficeincies for a sphere model.

    This function is adopted from the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L66

    Parameters
    ----------
    m: (np.complex) The complex refraction coefficient.

    x: (float) The particle size parameter at the specific wave-length.

    n_max: (int | None) The number of iterations for convergence.

    Returns
    -------
    qe: (np.float) The extinction Q efficiencies.

    qs: (np.float) The scattering Q efficiencies.

    g: (np.float) The g asymmetry parameter.
    """

    m = m.real - 1j * np.abs(m.imag)
    if n_max is None:
        nmax = int(m.real*x + 4.3*(m.real*x)**(1/3) + 3)
    else:
        nmax = n_max
    dpx = get_dlnpsirefnp(x, nmax)
    dpmx = get_dlnpsirefnp(m*x, nmax)
    dzx = get_dlnzetarefnp(x, nmax)
    T = get_tuprefnp(x,dpx,dzx)
    an = (T*(m*dpx-dpmx)/(m*dzx-dpmx))[1:]
    bn = (T*(dpx-m*dpmx)/(dzx-m*dpmx))[1:]
    return get_qqgnpref(an,bn,x)

def test_sphere(verbose=False):
    """
    This is a unit test for the torch and numpy version of the `qqg` function.
    """
    n_max, n_partset, n_part = 120, 10, 100
    np_random = np.random.RandomState(12345)

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    x_tstnp = np_random.rand(n_partset, n_part)
    assert x_tstnp.shape == (n_partset, n_part)
    x_tst = torch.from_numpy(x_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert x_tst.shape == (n_partset, n_part)

    m_tst, m_tstnp = toss_cmplxrand(np_random, (n_partset, n_part),
                                    tch_device, tch_cdtype)
    assert m_tst.shape == (n_partset, n_part)
    assert m_tstnp.shape == (n_partset, n_part)

    vars_spec = {
        'qe': {'dim': tuple(), 'dtype': tch_dtype},
        'qs': {'dim': tuple(), 'dtype': tch_dtype},
        'g': {'dim': tuple(), 'dtype': tch_dtype}}

    vars_pred = get_sphere(m_tst, x_tst, n_max, tch_device, tch_dtype, tch_cdtype)

    if verbose:
        print('Testing the sphere routine with ' +
              f'{n_partset*n_part} particles and n_max={n_max}:')

    for vname, vpred in zip(vars_spec, vars_pred):
        vinfo = vars_spec[vname]
        vdim = vinfo['dim']
        assert vpred.shape == (n_partset, n_part, *vdim)
        assert torch.is_tensor(vpred)
        vinfo['pred'] = vpred
        vinfo['reflst'] = []

    for i in range(n_partset):
        for j in range(n_part):
            vars_refij = get_sphererefnp(m_tstnp[i, j], x_tstnp[i, j], n_max)

            for vname, vrefij in zip(vars_spec, vars_refij):
                vinfo = vars_spec[vname]
                vdim = vinfo['dim']
                assert vrefij.shape == (*vdim,)
                vinfo['reflst'].append(vrefij)

    for vname, vinfo in vars_spec.items():
        vdim = vinfo['dim']
        vdtype = vinfo['dtype']
        vpred = vinfo['pred']

        vrefnp = np.array(vinfo['reflst']).reshape(n_partset, n_part, *vdim)
        assert vrefnp.shape == (n_partset, n_part, *vdim)

        vref = torch.from_numpy(vrefnp).to(device=tch_device, dtype=vdtype)
        assert vref.shape == (n_partset, n_part, *vdim)
        vinfo['ref'] = vref

        pnorm_err = 2

        v_abserr = (vpred - vref).reshape(n_partset, n_part, math.prod(vdim)).abs().pow(
            pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
        assert v_abserr.shape == (n_partset, n_part)

        v_relerr = 2.0 * v_abserr / (
            vpred.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
            vref.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
        v_relerr[v_abserr == 0.0] = 0.0
        assert v_relerr.shape == (n_partset, n_part)

        v_abserrmax = v_abserr.max().item()
        v_abserrmean = v_abserr.mean().item()
        v_relerrmax = v_relerr.max().item()
        v_relerrmean = v_relerr.mean().item()

        vinfo['abserrmax'] = v_abserrmax
        vinfo['abserrmean'] = v_abserrmean
        vinfo['relerrmax'] = v_relerrmax
        vinfo['relerrmean'] = v_relerrmean

        if verbose:
            print(f'  {vname + " " * (10-len(vname))}: Max Abs Error={v_abserrmax:0.03g}, ' +
                  f'Mean Abs Error={v_abserrmean:0.03g}, Max Rel Error={v_relerrmax:0.03g}, ' +
                  f'Mean Rel Error={v_relerrmean:0.03g}')

    for vname, vinfo in vars_spec.items():
        v_abserrmax = vinfo['abserrmax']
        assert v_abserrmax < 1e-8, f'{vname} Max Abs Error = {v_abserrmax:0.03g}'
        v_abserrmean = vinfo['abserrmean']
        assert v_abserrmean < 1e-10, f'{vname} Mean Abs Error = {v_abserrmean:0.03g}'

###########################################################
########## The Non-Spherical `coreshell` Routine ##########
###########################################################

def get_coreshellnonsphr(mc, ms, xc, xs, n_max, device, dtype, cdtype, full=False):
    """
    Calculating the Mie efficiencies for a core-shell particle model. No spherical
    approximations are implemented in this function. A more comprehensive super-set
    of this function which handles the edge cases better can be found at the `get_coreshell`
    function.

    This function is the tensor implementation of the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L79

    Parameters
    ----------
    mc: (torch.tensor) The complex refraction index for the core.
        `assert mc.shape == (n_partset, n_part)`

    ms: (torch.tensor) The complex refraction index for the shell.
        `assert ms.shape == (n_partset, n_part)`

    xc: (float) The particle core size parameter at the specific wave-length
        ($2\pi r_{core} / \lambda$).
        `assert xc.shape == (n_partset, n_part)`

    xs: (float) The particle shell size parameter at the specific wave-length
        ($2\pi r_{shell} / \lambda$).
        `assert xs.shape == (n_partset, n_part)`

    n_max: (int | None) The number of iterations for convergence.

    device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    full: (bool) Whether to only return the qqg pair, or the entire
        set of calculated arrays as the output.

    Returns
    -------
    Assuming `full=False`:

        qe: (torch.tensor) The extinction Q efficiencies.
            `assert qe.shape == (n_partset, n_part)`

        qs: (torch.tensor) The scattering Q efficiencies.
            `assert qs.shape == (n_partset, n_part)`

        g: (torch.tensor) The g asymmetry parameter.
            `assert g.shape == (n_partset, n_part)`

    For a full list of output variables with `full=True`, see
    the implementation.
    """
    x_shape = xc.shape
    assert mc.shape == x_shape
    assert ms.shape == x_shape
    assert xc.shape == x_shape
    assert xs.shape == x_shape
    ox_shape = tuple(1 for _ in x_shape)

    n_arange1 = torch.arange(n_max, device=device, dtype=dtype)
    assert n_arange1.shape == (n_max,)
    n_arange = n_arange1.reshape(*ox_shape, n_max)
    assert n_arange.shape == (*ox_shape, n_max)

    # Ensure index of refraction uses the correct sign convention
    mc = mc.real - 1j * mc.imag.abs()
    assert mc.shape == x_shape
    ms = ms.real - 1j * ms.imag.abs()
    assert ms.shape == x_shape

    ms_3d, xs_3d, mc_3d, xc_3d = ms.unsqueeze(-1), xs.unsqueeze(-1), mc.unsqueeze(-1), xc.unsqueeze(-1)
    assert ms_3d.shape == (*x_shape, 1)
    assert xs_3d.shape == (*x_shape, 1)
    assert mc_3d.shape == (*x_shape, 1)
    assert xc_3d.shape == (*x_shape, 1)

    # Logarithmic derivatives psi'/psi and zeta'/zeta by recurrence:
    # Naming convention: d = log derivative, Z = zeta, P = psi, sc = m_s*x_c etc., 's' alone = x_s
    dPss = get_dlnpsi(ms * xs, n_max, device, cdtype)
    assert dPss.shape == (*x_shape, n_max)
    dPs = get_dlnpsi(xs + 1j * 0, n_max, device, cdtype)
    assert dPs.shape == (*x_shape, n_max)
    dPcc = get_dlnpsi(mc * xc, n_max, device, cdtype)
    assert dPcc.shape == (*x_shape, n_max)
    dPsc = get_dlnpsi(ms * xc, n_max, device, cdtype)
    assert dPsc.shape == (*x_shape, n_max)

    dZss = get_dlnzeta(ms * xs, n_max, device, cdtype)
    assert dZss.shape == (*x_shape, n_max)
    dZs = get_dlnzeta(xs + 1j * 0, n_max, device, cdtype)
    assert dZs.shape == (*x_shape, n_max)
    dZsc = get_dlnzeta(ms * xc, n_max, device, cdtype)
    assert dZsc.shape == (*x_shape, n_max)

    # Upward recurrence for zeta(msxc)*psi(msxc), zeta(msxs)*psi(msxc), psi(msxc)/psi(msxs), and psi(xs)/zeta(xs):
    PsoZs = get_tup(xs, dPs, dZs, device, dtype, cdtype)
    assert PsoZs.shape == (*x_shape, n_max)

    alpha_PscoPss = (dPss + n_arange / (ms_3d * xs_3d)) / (dPsc + n_arange / (ms_3d * xc_3d))
    assert alpha_PscoPss.shape == (*x_shape, n_max)
    alpha_PscoPss[..., 0] = ((torch.exp(-1j * ms * (xc + xs)) - torch.exp(1j * ms * (xc - xs)))
                             / (torch.exp(-2 * 1j * ms * xs) - 1))
    PscoPss = alpha_PscoPss.cumprod(dim=-1)
    assert PscoPss.shape == (*x_shape, n_max)

    alpha_ZscPsc = 1.0 / ((dZsc + n_arange / (ms_3d * xc_3d)) * (dPsc + n_arange / (ms_3d * xc_3d)))
    assert alpha_ZscPsc.shape == (*x_shape, n_max)
    alpha_ZscPsc[..., 0] = (1.0 - torch.exp(-2 * 1j * ms * xc)) / 2.0
    ZscPsc = alpha_ZscPsc.cumprod(dim=-1)
    assert ZscPsc.shape == (*x_shape, n_max)

    alpha_ZssPsc = 1.0 / ((dZss + n_arange / (ms_3d * xs_3d)) * (dPsc + n_arange / (ms_3d * xc_3d)))
    assert alpha_ZssPsc.shape == (*x_shape, n_max)
    alpha_ZssPsc[..., 0] = -0.5 * (torch.exp(-1j * ms * (xs + xc)) - torch.exp(-1j * ms * (xs - xc)))
    ZssPsc = alpha_ZssPsc.cumprod(dim=-1)
    assert ZssPsc.shape == (*x_shape, n_max)

    # Mie coefficients
    U = mc_3d * dPsc - ms_3d * dPcc
    assert U.shape == (*x_shape, n_max)
    V = ms_3d * dPsc - mc_3d * dPcc
    assert V.shape == (*x_shape, n_max)
    W = -1j * (PscoPss * ZssPsc - ZscPsc)
    assert W.shape == (*x_shape, n_max)
    an = (PsoZs * ((dPss - ms_3d * dPs) * (mc_3d + U * W) -         U * PscoPss**2)
          / ((dPss - ms_3d * dZs) * (mc_3d + U * W) -         U * PscoPss**2))[..., 1:]
    assert an.shape == (*x_shape, n_max - 1)
    bn = (PsoZs * ((ms_3d * dPss - dPs) * (ms_3d + V * W) - ms_3d * V * PscoPss**2)
          / ((ms_3d * dPss - dZs) * (ms_3d + V * W) - ms_3d * V * PscoPss**2))[..., 1:]
    assert bn.shape == (*x_shape, n_max - 1)

    qe, qs, g = get_qqg(an, bn, xs, device, dtype)
    output = (qe, qs, g, dPss, dPs, dPcc, dPsc, dZss, dZs, dZsc,
        PsoZs, ZscPsc, ZssPsc, PscoPss, U, V, W, an, bn) if full else (qe, qs, g)

    return output

def get_coreshellrefnp(mc_np, ms_np, xc_np, xs_np, n_max, void_qe=0.0,
                       void_qs=0.0, void_g=0.0, use_sphere=True, full=False):
    """
    Calculating the Mie efficeincies for a core-shell particle model.

    This function is adopted from the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L79

    Parameters
    ----------
    mc_np: (np.complex) The complex refraction index for the core.

    ms_np: (np.complex) The complex refraction index for the shell.

    xc_np: (float) The particle core size parameter at the specific wave-length
        ($2\pi r_{core} / \lambda$).

    xs_np: (float) The particle core size parameter at the specific wave-length
        ($2\pi r_{shell} / \lambda$).

    n_max: (int | None) The number of iterations for convergence.

    void_qe: (float) The extinction Q efficiency for zero-sized particles.
        This is only relevant when `use_sphere=True`.

    void_qs: (float) The scattering Q efficiency for zero-sized particles.
        This is only relevant when `use_sphere=True`.

    void_qs: (float) The g asymmetry parameter for zero-sized particles.
        This is only relevant when `use_sphere=True`.

    use_sphere: (bool) Whether to use the the `sphere` function when the
        particle can be approximated as a sphere.

    full: (bool) Whether to only return the qqg pair, or the entire
        set of calculated arrays as the output. Setting `full=True` can
        only be made possible when `use_sphere=False`.

    Returns
    -------
    Assuming `full=False`:

        qe: (np.float) The extinction Q efficiency.

        qs: (np.float) The scattering Q efficiency.

        g: (np.float) The g asymmetry parameter.

    For a full list of output variables with `full=True`, see
    the implementation.
    """

    assert (not use_sphere) or (not full)

    # Check if the particle is empty
    if use_sphere and (xs_np <= 0.0):
        return np.array(void_qe), np.array(void_qs), np.array(void_g)
    # Rayleigh's approximation was added to accommodate overly small particles
    if use_sphere and (xs_np < 0.001):
        f = (xc_np / xs_np) ** 3
        m = mc_np * f + ms_np * (1 - f)
        mratio = (m * m - 1) / (m * m + 2)
        qs = (8 / 3) * (xs_np ** 4) * (np.abs(mratio) ** 2)
        qa = 4 * xs_np * mratio.imag * (1 + (4 / 3) * (xs_np ** 3) * mratio.imag)
        qe = qs + qa
        g = np.array(0.0)
        return qe, qs, g
    # Check if the sphere function can be used instead (type 1)
    if use_sphere and ((mc_np == ms_np) or (xc_np / xs_np < 0.01)):
        return get_sphererefnp(ms_np, xs_np, n_max)
    # Check if the sphere function can be used instead (type 1)
    if use_sphere and (xc_np / xs_np > 0.99):
        return get_sphererefnp(mc_np, xs_np, n_max)

    # Ensure index of refraction uses the correct sign convention
    mc_np = mc_np.real - 1j * np.abs(mc_np.imag)
    ms_np = ms_np.real - 1j * np.abs(ms_np.imag)

    # Logarithmic derivatives psi'/psi and zeta'/zeta by recurrence:
    # Naming convention: d = log derivative, Z = zeta, P = psi, sc = m_s*x_c etc., 's' alone = x_s
    if n_max is None:
        n_max = int(ms_np.real * xs_np + 4.3 * (ms_np.real * xs_np)**(1/3) + 3)

    [dPss,dPs,dPcc,dPsc] =  [get_dlnpsirefnp(arg,n_max) for arg in [ms_np * xs_np, xs_np + 1j * 0, mc_np * xc_np, ms_np * xc_np]]
    [dZss,dZs,     dZsc] = [get_dlnzetarefnp(arg,n_max) for arg in [ms_np * xs_np, xs_np + 1j * 0,                ms_np * xc_np]]

    # Upward recurrence for zeta(msxc)*psi(msxc), zeta(msxs)*psi(msxc), psi(msxc)/psi(msxs), and psi(xs)/zeta(xs):
    PsoZs = get_tuprefnp(xs_np, dPs, dZs)
    ZscPsc, ZssPsc, PscoPss = [np.zeros((n_max), dtype='cdouble') for _ in range(3)]
    ZscPsc[0]  = (1 - np.exp(-2 * 1j * ms_np * xc_np)) / 2
    ZssPsc[0]  = -0.5 * (np.exp(-1j * ms_np * (xs_np + xc_np)) - np.exp(-1j * ms_np * (xs_np - xc_np)))
    PscoPss[0] = (np.exp(-1j * ms_np * (xc_np + xs_np)) - np.exp(1j * ms_np * (xc_np - xs_np)))/(np.exp(-2 * 1j * ms_np * xs_np) - 1)
    for n in range(1, n_max):
        PscoPss[n] = PscoPss[n-1] * (dPss[n] + n / (ms_np * xs_np)) / (dPsc[n] + n / (ms_np * xc_np))
        ZscPsc[n]  = ZscPsc[n-1] / ((dZsc[n] + n / (ms_np * xc_np)) * (dPsc[n] + n / (ms_np * xc_np)))
        ZssPsc[n]  = ZssPsc[n-1] / ((dZss[n] + n / (ms_np * xs_np)) * (dPsc[n] + n / (ms_np * xc_np)))

    # Mie coefficients
    U = mc_np * dPsc - ms_np * dPcc
    V = ms_np * dPsc - mc_np * dPcc
    W = -1j * (PscoPss * ZssPsc - ZscPsc)
    an = (PsoZs * ((dPss - ms_np * dPs) * (mc_np + U * W) - U * PscoPss**2) /
          ((dPss - ms_np * dZs) * (mc_np + U * W) - U * PscoPss**2))[1:]
    bn = (PsoZs * ((ms_np * dPss - dPs) * (ms_np + V * W) - ms_np * V * PscoPss**2) /
          ((ms_np * dPss - dZs) * (ms_np + V * W) - ms_np * V * PscoPss**2))[1:]

    qe, qs, g = get_qqgnpref(an, bn, xs_np)
    output = (qe, qs, g, dPss, dPs, dPcc, dPsc, dZss, dZs, dZsc,
        PsoZs, ZscPsc, ZssPsc, PscoPss, U, V, W, an, bn) if full else (qe, qs, g)
    return output

def test_coreshellnonsphr(verbose=False):
    """
    This is a unit test for the torch and numpy version of the `qqg` function.
    """
    n_max, n_partset, n_part = 120, 10, 100
    np_random = np.random.RandomState(12345)

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    xs_tstnp = 10.0 ** (0.9 * np_random.randn(n_partset, n_part))
    assert xs_tstnp.shape == (n_partset, n_part)
    xs_tst = torch.from_numpy(xs_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert xs_tst.shape == (n_partset, n_part)

    xc_tstnp = xs_tstnp * np.minimum(1.0, np.exp(-1.30 + 0.84 * np_random.randn(n_partset, n_part)))
    assert xc_tstnp.shape == (n_partset, n_part)
    xc_tst = torch.from_numpy(xc_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert xc_tst.shape == (n_partset, n_part)

    mc_tstrealnp = 1.55 + 0.4 * np_random.rand(n_partset, n_part)
    assert mc_tstrealnp.shape == (n_partset, n_part)
    mc_tstimagnp = 0.00 + 0.8 * np_random.rand(n_partset, n_part)
    assert mc_tstimagnp.shape == (n_partset, n_part)
    mc_tstnp = mc_tstrealnp + 1j * mc_tstimagnp
    assert mc_tstnp.shape == (n_partset, n_part)
    mc_tst = torch.from_numpy(mc_tstnp).to(device=tch_device, dtype=tch_cdtype)
    assert mc_tst.shape == (n_partset, n_part)

    ms_tstrealnp = 1.32 + 0.18 * np_random.rand(n_partset, n_part)
    assert ms_tstrealnp.shape == (n_partset, n_part)
    ms_tstimagnp = 0.00 + 0.001 * np_random.rand(n_partset, n_part)
    assert ms_tstimagnp.shape == (n_partset, n_part)
    ms_tstnp = ms_tstrealnp + 1j * ms_tstimagnp
    assert ms_tstnp.shape == (n_partset, n_part)
    ms_tst = torch.from_numpy(ms_tstnp).to(device=tch_device, dtype=tch_cdtype)
    assert ms_tst.shape == (n_partset, n_part)

    vars_spec = {
        'qe': {'dim': tuple(), 'dtype': tch_dtype},
        'qs': {'dim': tuple(), 'dtype': tch_dtype},
        'g': {'dim': tuple(), 'dtype': tch_dtype},
        'dPss': {'dim': (n_max,), 'dtype': tch_cdtype},
        'dPs': {'dim': (n_max,), 'dtype': tch_cdtype},
        'dPcc': {'dim': (n_max,), 'dtype': tch_cdtype},
        'dPsc': {'dim': (n_max,), 'dtype': tch_cdtype},
        'dZss': {'dim': (n_max,), 'dtype': tch_cdtype},
        'dZs': {'dim': (n_max,), 'dtype': tch_cdtype},
        'dZsc': {'dim': (n_max,), 'dtype': tch_cdtype},
        'PsoZs': {'dim': (n_max,), 'dtype': tch_cdtype},
        'ZscPsc': {'dim': (n_max,), 'dtype': tch_cdtype},
        'ZssPsc': {'dim': (n_max,), 'dtype': tch_cdtype},
        'PscoPss': {'dim': (n_max,), 'dtype': tch_cdtype},
        'U': {'dim': (n_max,), 'dtype': tch_cdtype},
        'V': {'dim': (n_max,), 'dtype': tch_cdtype},
        'W': {'dim': (n_max,), 'dtype': tch_cdtype},
        'an': {'dim': (n_max - 1,), 'dtype': tch_cdtype},
        'bn': {'dim': (n_max - 1,), 'dtype': tch_cdtype}}

    if verbose:
        print('Testing the non-spherical core-shell routine with ' +
              f'{n_partset*n_part} particles and n_max={n_max}:')

    vars_pred = get_coreshellnonsphr(mc_tst, ms_tst, xc_tst, xs_tst,
        n_max, tch_device, tch_dtype, tch_cdtype, full=True)

    for vname, vpred in zip(vars_spec, vars_pred):
        vinfo = vars_spec[vname]
        vdim = vinfo['dim']
        assert vpred.shape == (n_partset, n_part, *vdim)
        assert torch.is_tensor(vpred)
        vinfo['pred'] = vpred
        vinfo['reflst'] = []

    for i in range(n_partset):
        for j in range(n_part):
            vars_refij = get_coreshellrefnp(mc_tstnp[i, j], ms_tstnp[i, j],
                xc_tstnp[i, j], xs_tstnp[i, j], n_max, use_sphere=False, full=True)

            for vname, vrefij in zip(vars_spec, vars_refij):
                vinfo = vars_spec[vname]
                vdim = vinfo['dim']
                assert vrefij.shape == (*vdim,)
                vinfo['reflst'].append(vrefij)

    for vname, vinfo in vars_spec.items():
        vdim = vinfo['dim']
        vdtype = vinfo['dtype']
        vpred = vinfo['pred']

        vrefnp = np.array(vinfo['reflst']).reshape(n_partset, n_part, *vdim)
        assert vrefnp.shape == (n_partset, n_part, *vdim)

        vref = torch.from_numpy(vrefnp).to(device=tch_device, dtype=vdtype)
        assert vref.shape == (n_partset, n_part, *vdim)
        vinfo['ref'] = vref

        pnorm_err = 2

        v_abserr = (vpred - vref).reshape(n_partset, n_part, math.prod(vdim)).abs().pow(
            pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
        assert v_abserr.shape == (n_partset, n_part)

        v_relerr = 2.0 * v_abserr / (
            vpred.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
            vref.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
        v_relerr[v_abserr == 0.0] = 0.0
        assert v_relerr.shape == (n_partset, n_part)

        v_abserrmax = v_abserr.max().item()
        v_abserrmean = v_abserr.mean().item()
        v_relerrmax = v_relerr.max().item()
        v_relerrmean = v_relerr.mean().item()

        vinfo['abserrmax'] = v_abserrmax
        vinfo['abserrmean'] = v_abserrmean
        vinfo['relerrmax'] = v_relerrmax
        vinfo['relerrmean'] = v_relerrmean

        if verbose:
            print(f'  {vname + " " * (10-len(vname))}: Max Abs Error={v_abserrmax:0.03g}, ' +
                  f'Mean Abs Error={v_abserrmean:0.03g}, Max Rel Error={v_relerrmax:0.03g}, ' +
                  f'Mean Rel Error={v_relerrmean:0.03g}')

    for vname, vinfo in vars_spec.items():
        v_abserrmax = vinfo['abserrmax']
        assert v_abserrmax < 1e-8, f'{vname} Max Abs Error = {v_abserrmax:0.03g}'
        v_abserrmean = vinfo['abserrmean']
        assert v_abserrmean < 1e-10, f'{vname} Mean Abs Error = {v_abserrmean:0.03g}'

###########################################################
############# The General `coreshell` Routine #############
###########################################################

def get_coreshell(mc, ms, xc, xs, n_max, device, dtype, cdtype,
                  void_qe=0.0, void_qs=0.0, void_g=0.0):
    """
    Calculating the Mie efficeincies for a core-shell particle model. This is a more general
    version of the `get_coreshellnonsphr` function, which handles the edge cases better:

        1. When the core to shell size ratio is less than 1%, the core-shell model becomes
            numerically unstable. In this case, the spherical routine for the shell is a good
            approximation.

        2. When the core to shell size ratio is less than 99%, the core-shell model becomes
            numerically unstable. In this case, the spherical routine for the core is a good
            approximation.

        3. When shell size is zero, some constants need to be reported.

    This function is the tensor implementation of the following:
      https://github.com/pnnl/NEURALMIE/blob/0c1512a9a93185928e30e5837d0fdae4b06cc1ea/TAMie.py#L79

    Parameters
    ----------
    mc: (torch.tensor) The complex refraction index for the core.
        `assert mc.shape == (n_partset, n_part)`

    ms: (torch.tensor) The complex refraction index for the shell.
        `assert ms.shape == (n_partset, n_part)`

    xc: (float) The particle core size parameter at the specific wave-length
        ($2\pi r_{core} / \lambda$).
        `assert xc.shape == (n_partset, n_part)`

    xs: (float) The particle shell size parameter at the specific wave-length
        ($2\pi r_{shell} / \lambda$).
        `assert xs.shape == (n_partset, n_part)`

    n_max: (int | None) The number of iterations for convergence.

    device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    void_qe: (float) The extinction Q efficiency for zero-sized particles.

    void_qs: (float) The scattering Q efficiency for zero-sized particles.

    void_qs: (float) The g asymmetry parameter for zero-sized particles.

    Returns
    -------
    Assuming `full=False`:

        qe: (torch.tensor) The extinction Q efficiencies.
            `assert qe.shape == (n_partset, n_part)`

        qs: (torch.tensor) The scattering Q efficiencies.
            `assert qs.shape == (n_partset, n_part)`

        g: (torch.tensor) The g asymmetry parameter.
            `assert g.shape == (n_partset, n_part)`

    For a full list of output variables with `full=True`, see
    the implementation.
    """
    x_shape = xc.shape
    assert mc.shape == x_shape
    assert ms.shape == x_shape
    assert xc.shape == x_shape
    assert xs.shape == x_shape

    # Reshaping the arrays
    n_part = math.prod(x_shape)
    mc, ms, xc, xs = mc.ravel(), ms.ravel(), xc.ravel(), xs.ravel()
    assert mc.shape == (n_part,)
    assert ms.shape == (n_part,)
    assert xc.shape == (n_part,)
    assert xs.shape == (n_part,)

    # The index of the zero-sized particles. They should be assigned an input constant.
    i_void = (xs <= 0)
    assert i_void.shape == (n_part,)
    i_rmng = i_void.logical_not()
    assert i_rmng.shape == (n_part,)

    # The index for particles that are too small
    i_rayleigh = (xs < 0.001).logical_and(i_rmng)
    assert i_rayleigh.shape == (n_part,)
    i_rmng = i_rmng.logical_and(i_rayleigh.logical_not())
    assert i_rmng.shape == (n_part,)

    # The index for particles that have negligible core sizes.
    i_sphr1 = (mc == ms).logical_or((xc / xs) < 0.01).logical_and(i_rmng)
    assert i_sphr1.shape == (n_part,)
    i_rmng = i_rmng.logical_and(i_sphr1.logical_not())
    assert i_rmng.shape == (n_part,)

    # The index for particles that are almost all core.
    i_sphr2 = ((xc / xs) > 0.99).logical_and(i_rmng)
    assert i_sphr2.shape == (n_part,)
    i_rmng = i_rmng.logical_and(i_sphr2.logical_not())
    assert i_rmng.shape == (n_part,)

    # The rest of the particles follow the core-shell model reasonably well.
    i_coreshl = i_rmng
    assert i_coreshl.shape == (n_part,)

    # Initializing empty array for the extinction q efficiency output.
    qe = torch.empty(n_part, device=device, dtype=dtype)
    assert qe.shape == (n_part,)

    # Initializing empty arrays for the scattering q efficiency output.
    qs = torch.empty(n_part, device=device, dtype=dtype)
    assert qs.shape == (n_part,)

    # Initializing empty arrays for the g asymmetry parameter output.
    g = torch.empty(n_part, device=device, dtype=dtype)
    assert g.shape == (n_part,)

    # Handling the zero-sized particles case.
    if i_void.any():
        qe[i_void] = void_qe
        qs[i_void] = void_qs
        g[i_void] = void_g

    # Handling the core-negligible particles case.
    if i_sphr1.any():
        qe_sphr1, qs_sphr1, g_sphr1 = get_sphere(ms[i_sphr1], xs[i_sphr1], n_max,
            device, dtype, cdtype)
        qe[i_sphr1] = qe_sphr1
        qs[i_sphr1] = qs_sphr1
        g[i_sphr1] = g_sphr1

    # Handling the almost all-core particles case.
    if i_sphr2.any():
        qe_sphr2, qs_sphr2, g_sphr2 = get_sphere(mc[i_sphr2], xs[i_sphr2], n_max,
            device, dtype, cdtype)
        qe[i_sphr2] = qe_sphr2
        qs[i_sphr2] = qs_sphr2
        g[i_sphr2] = g_sphr2
    
    # Handling the very small particle cases.
    if i_rayleigh.any():
        n_rayl = i_rayleigh.sum().item()
        # print(f'  ** Note **: Found {n_rayl} particles qualifying for Rayleigh approximations.')

        mc_rayl, ms_rayl, xc_rayl, xs_rayl = mc[i_rayleigh], ms[i_rayleigh], xc[i_rayleigh], xs[i_rayleigh]
        assert mc_rayl.shape == (n_rayl,)
        assert ms_rayl.shape == (n_rayl,)
        assert xc_rayl.shape == (n_rayl,)
        assert xs_rayl.shape == (n_rayl,)

        xc2xs_rayl = (xc_rayl / xs_rayl).pow(3.0)
        assert xc2xs_rayl.shape == (n_rayl,)
        
        # some weighted average of the core and shell refractive indices
        mcs_rayl = mc_rayl * xc2xs_rayl + ms_rayl * (1.0 - xc2xs_rayl)
        assert mcs_rayl.shape == (n_rayl,)

        # The $m^2$ value in Rayleigh's formula
        mcs_sqrayl = mcs_rayl.square()
        assert mcs_sqrayl.shape == (n_rayl,)

        # `mratio_rayl` is basically $\frac{m^2 - 1}{m^2 + 2}$
        mratio_rayl = (mcs_sqrayl - 1.0) / (mcs_sqrayl + 2.0)
        assert mratio_rayl.shape == (n_rayl,)

        # The Rayleigh scattering efficiency formula: 
        #   $Q_s = \frac{8}{3} x^4 |\frac{m^2 - 1}{m^2 + 2}|^2$
        qs_rayleigh = (8 / 3) * xs_rayl.pow(4) * mratio_rayl.abs().square()
        assert qs_rayleigh.shape == (n_rayl,)

        # The Rayleigh absorption efficiency formula: 
        #   $Q_a = 4 x Im{\frac{m^2 - 1}{m^2 + 2}} (1 + \frac{4x^3}{3} Im{\frac{m^2 - 1}{m^2 + 2}})$
        qa_rayleigh = 4 * xs_rayl * mratio_rayl.imag * (1 + (4 / 3) * xs_rayl.pow(3) * mratio_rayl.imag)
        assert qa_rayleigh.shape == (n_rayl,)

        # The extinction efficiency is the sum of scattering and absorption efficiencies
        qe_rayleigh = qs_rayleigh + qa_rayleigh
        assert qe_rayleigh.shape == (n_rayl,)

        # In In Rayleigh scattering, the "g asymmetry parameter" is essentially always close to zero, 
        # indicating that the scattered light is distributed nearly equally in all directions due to 
        # the small size of the scattering particles compared to the wavelength of light
        g_rayleigh = 0.0

        qe[i_rayleigh] = qe_rayleigh
        qs[i_rayleigh] = qs_rayleigh
        g[i_rayleigh] = g_rayleigh

    # Handling the almost all-core particles case.
    group_nmax = False
    need_coreshl = i_coreshl.any() > 0
    if need_coreshl and (not group_nmax):
        qe_coreshl, qs_coreshl, g_coreshl = get_coreshellnonsphr(mc[i_coreshl], ms[i_coreshl],
            xc[i_coreshl], xs[i_coreshl], n_max, device, dtype, cdtype, full=False)
        qe[i_coreshl] = qe_coreshl
        qs[i_coreshl] = qs_coreshl
        g[i_coreshl] = g_coreshl
    elif need_coreshl and group_nmax:
        # In this setting we try to divide the particles based on their appropriate
        # `n_max` values into 100 bins, and then run each bin with its `n_max`.
        # The premise is that this may reduce the amount of unnecessary computation
        # by using a large `n_max` for all particles. However, in my initial tests,
        # the speed-ups were not significantly better, and I abandoned this idea.
        i_coreshl2 = torch.arange(n_part, device=device, dtype=torch.long)[i_coreshl]
        mc2, ms2, xc2, xs2 = mc[i_coreshl], ms[i_coreshl], xc[i_coreshl], xs[i_coreshl]
        n_part2, = ms2.shape

        n_maxbch = (ms2.real * xs2 + 4.3 * (ms2.real * xs2).pow(1.0 / 3.0) + 3).to(dtype=torch.long)
        assert n_maxbch.shape == (n_part2,)

        # Capping up `n_max` at the user-provided level
        if n_max is not None:
            n_maxbch[n_maxbch > n_max] = n_max

        n_maxbchas = n_maxbch.argsort()
        i_stends = np.linspace(0, n_part2, max(101, n_part2 + 1)).astype(int).tolist()
        for ii1, ii2 in zip(i_stends[:-1], i_stends[1:]):
            i_mb = n_maxbchas[ii1: ii2]
            n_max3 = n_maxbch[i_mb[-1]]
            assert (n_maxbch[i_mb] <= n_max3).all()
            qe3, qs3, g3 = get_coreshellnonsphr(mc2[i_mb], ms2[i_mb], xc2[i_mb],
                xs2[i_mb], n_max3, device, dtype, cdtype, full=False)

            i_mb2 = i_coreshl2[i_mb]
            qe[i_mb2] = qe3
            qs[i_mb2] = qs3
            g[i_mb2] = g3

    qe, qs, g = qe.reshape(*x_shape), qs.reshape(*x_shape), g.reshape(*x_shape)

    assert not qe.isnan().any()
    assert not qs.isnan().any()
    assert not g.isnan().any()

    return qe, qs, g

def test_coreshell(verbose=False):
    """
    This is a unit test for the torch and numpy version of the `coreshell` function.
    """
    n_max, n_partset, n_part = 120, 10, 100
    np_random = np.random.RandomState(12345)

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    xs_tstnp = 10.0 ** (2.0 * np_random.randn(n_partset, n_part))
    xs_tstnp[xs_tstnp < (10 ** (2.0 * -2.3))] = 0.0
    assert xs_tstnp.shape == (n_partset, n_part)
    xs_tst = torch.from_numpy(xs_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert xs_tst.shape == (n_partset, n_part)

    xc_tstnp = xs_tstnp * np.minimum(1.1, np.exp(-2.60 + 1.68 * np_random.randn(n_partset, n_part)))
    assert xc_tstnp.shape == (n_partset, n_part)
    xc_tst = torch.from_numpy(xc_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert xc_tst.shape == (n_partset, n_part)

    mc_tstrealnp = 1.55 + 0.4 * np_random.rand(n_partset, n_part)
    assert mc_tstrealnp.shape == (n_partset, n_part)
    mc_tstimagnp = 0.00 + 0.8 * np_random.rand(n_partset, n_part)
    assert mc_tstimagnp.shape == (n_partset, n_part)
    mc_tstnp = mc_tstrealnp + 1j * mc_tstimagnp
    assert mc_tstnp.shape == (n_partset, n_part)
    mc_tst = torch.from_numpy(mc_tstnp).to(device=tch_device, dtype=tch_cdtype)
    assert mc_tst.shape == (n_partset, n_part)

    ms_tstrealnp = 1.32 + 0.18 * np_random.rand(n_partset, n_part)
    assert ms_tstrealnp.shape == (n_partset, n_part)
    ms_tstimagnp = 0.00 + 0.001 * np_random.rand(n_partset, n_part)
    assert ms_tstimagnp.shape == (n_partset, n_part)
    ms_tstnp = ms_tstrealnp + 1j * ms_tstimagnp
    assert ms_tstnp.shape == (n_partset, n_part)
    ms_tst = torch.from_numpy(ms_tstnp).to(device=tch_device, dtype=tch_cdtype)
    assert ms_tst.shape == (n_partset, n_part)

    vars_spec = {
        'qe': {'dim': tuple(), 'dtype': tch_dtype},
        'qs': {'dim': tuple(), 'dtype': tch_dtype},
        'g': {'dim': tuple(), 'dtype': tch_dtype}}

    if verbose:
        print('Testing the fully-fledged core-shell routine with ' +
              f'{n_partset*n_part} particles and n_max={n_max}:')

    vars_pred = get_coreshell(mc_tst, ms_tst, xc_tst, xs_tst,
        n_max, tch_device, tch_dtype, tch_cdtype)

    for vname, vpred in zip(vars_spec, vars_pred):
        vinfo = vars_spec[vname]
        vdim = vinfo['dim']
        assert vpred.shape == (n_partset, n_part, *vdim)
        assert torch.is_tensor(vpred)
        vinfo['pred'] = vpred
        vinfo['reflst'] = []

    for i in range(n_partset):
        for j in range(n_part):
            vars_refij = get_coreshellrefnp(mc_tstnp[i, j], ms_tstnp[i, j],
                xc_tstnp[i, j], xs_tstnp[i, j], n_max, use_sphere=True, full=False)

            for vname, vrefij in zip(vars_spec, vars_refij):
                vinfo = vars_spec[vname]
                vdim = vinfo['dim']
                assert vrefij.shape == (*vdim,)
                vinfo['reflst'].append(vrefij)

    for vname, vinfo in vars_spec.items():
        vdim = vinfo['dim']
        vdtype = vinfo['dtype']
        vpred = vinfo['pred']

        vrefnp = np.array(vinfo['reflst']).reshape(n_partset, n_part, *vdim)
        assert vrefnp.shape == (n_partset, n_part, *vdim)

        vref = torch.from_numpy(vrefnp).to(device=tch_device, dtype=vdtype)
        assert vref.shape == (n_partset, n_part, *vdim)
        vinfo['ref'] = vref

        pnorm_err = 2

        v_abserr = (vpred - vref).reshape(n_partset, n_part, math.prod(vdim)).abs().pow(
            pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
        assert v_abserr.shape == (n_partset, n_part)

        v_relerr = 2.0 * v_abserr / (
            vpred.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
            vref.reshape(n_partset, n_part, math.prod(vdim)).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
        v_relerr[v_abserr == 0.0] = 0.0
        assert v_relerr.shape == (n_partset, n_part)

        v_abserrmax = v_abserr.max().item()
        v_abserrmean = v_abserr.mean().item()
        v_relerrmax = v_relerr.max().item()
        v_relerrmean = v_relerr.mean().item()

        vinfo['abserrmax'] = v_abserrmax
        vinfo['abserrmean'] = v_abserrmean
        vinfo['relerrmax'] = v_relerrmax
        vinfo['relerrmean'] = v_relerrmean

        if verbose:
            print(f'  {vname + " " * (10-len(vname))}: Max Abs Error={v_abserrmax:0.03g}, ' +
                  f'Mean Abs Error={v_abserrmean:0.03g}, Max Rel Error={v_relerrmax:0.03g}, ' +
                  f'Mean Rel Error={v_relerrmean:0.03g}')

    for vname, vinfo in vars_spec.items():
        v_abserrmean = vinfo['abserrmax']
        assert v_abserrmax < 1e-8, f'{vname} Max Abs Error = {v_abserrmax:0.03g}'
        v_abserrmean = vinfo['abserrmean']
        assert v_abserrmean < 1e-10, f'{vname} Mean Abs Error = {v_abserrmean:0.03g}'
    
    # Finding the refernce and predicted absorption efficiency values
    qe_pred = vars_spec['qe']['pred']
    assert qe_pred.shape == (n_partset, n_part)
    qs_pred = vars_spec['qs']['pred']
    assert qs_pred.shape == (n_partset, n_part)
    qa_pred = qe_pred - qs_pred
    assert qa_pred.shape == (n_partset, n_part)

    qe_ref = vars_spec['qe']['ref']
    assert qe_ref.shape == (n_partset, n_part)
    qs_ref = vars_spec['qs']['ref']
    assert qs_ref.shape == (n_partset, n_part)
    qa_ref = qe_ref - qs_ref
    assert qa_ref.shape == (n_partset, n_part)

###########################################################
############### Optional Unit Test Functions ##############
###########################################################

def test_smallnmax(verbose=False):
    """
    This is a unit test for the torch version of the `coreshell` function. Here we want to see how a 
    small `n_max` impacts the results for large particle sizes.

    This function generates some synthetic data. If you have real data, see the similar `test_smallnmaxdata` function
    """
    n_max, n_partset, n_part = 120, 10, 100
    np_random = np.random.RandomState(12345)

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    xs_tstnp = 10.0 ** (2.3 + 0.15 * np_random.randn(n_partset, n_part))
    assert xs_tstnp.shape == (n_partset, n_part)
    xs_tst = torch.from_numpy(xs_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert xs_tst.shape == (n_partset, n_part)

    xc_tstnp = xs_tstnp * np.minimum(1.0, 10.0 ** (-1.2 + 0.5 * np_random.randn(n_partset, n_part)))
    assert xc_tstnp.shape == (n_partset, n_part)
    xc_tst = torch.from_numpy(xc_tstnp).to(device=tch_device, dtype=tch_dtype)
    assert xc_tst.shape == (n_partset, n_part)

    mc_rand = np_random.rand(n_partset, n_part) > 0.96
    mc_tstrealnp = 1.55 + (0.29 * mc_rand)
    assert mc_tstrealnp.shape == (n_partset, n_part)
    mc_tstimagnp = 0.00 + (0.8 * mc_rand)
    assert mc_tstimagnp.shape == (n_partset, n_part)
    mc_tstnp = mc_tstrealnp + 1j * mc_tstimagnp
    assert mc_tstnp.shape == (n_partset, n_part)
    mc_tst = torch.from_numpy(mc_tstnp).to(device=tch_device, dtype=tch_cdtype)
    assert mc_tst.shape == (n_partset, n_part)

    ms_tstrealnp = 1.32 + 0.18 * np_random.rand(n_partset, n_part)
    assert ms_tstrealnp.shape == (n_partset, n_part)
    ms_tstimagnp = 0.00 + 0.001 * np_random.rand(n_partset, n_part)
    assert ms_tstimagnp.shape == (n_partset, n_part)
    ms_tstnp = ms_tstrealnp + 1j * ms_tstimagnp
    assert ms_tstnp.shape == (n_partset, n_part)
    ms_tst = torch.from_numpy(ms_tstnp).to(device=tch_device, dtype=tch_cdtype)
    assert ms_tst.shape == (n_partset, n_part)

    n_maxinfrd = (ms_tst.real * xs_tst + 4.3 * (ms_tst.real * xs_tst).pow(1.0 / 3.0) + 3).to(dtype=torch.long)
    assert n_maxinfrd.shape == (n_partset, n_part)

    if verbose:
        print(f'Testing the core-shell routine on synthetic data with {n_partset*n_part} particles ' +
            f'and n_max={n_max} where {n_maxinfrd.min().item()} <= inferred n_max <= {n_maxinfrd.max().item()}:')

    # Running the small `n_max` results
    qe_lrg1, qs_lrg1, g_lrg1 = get_coreshell(mc_tst,  ms_tst, xc_tst, xs_tst, 120, 
        tch_device, tch_dtype, tch_cdtype, void_qe=0.0, void_qs=0.0, void_g=0.0)

    # Running the large `n_max` results
    qe_lrg2, qs_lrg2, g_lrg2 = get_coreshell(mc_tst,  ms_tst, xc_tst, xs_tst, 1200, 
        tch_device, tch_dtype, tch_cdtype, void_qe=0.0, void_qs=0.0, void_g=0.0)

    # Comparing the runs
    for vname, vpred, vref in [('qe', qe_lrg1, qe_lrg2), ('qs', qs_lrg1, qs_lrg2), ('g', g_lrg1, g_lrg2)]:
        pnorm_err = 2
        v_abserr = (vpred - vref).reshape(n_partset, n_part, 1).abs().pow(
            pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
        assert v_abserr.shape == (n_partset, n_part)

        v_relerr = 2 * v_abserr / (
            vpred.reshape(n_partset, n_part, 1).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
            vref.reshape(n_partset, n_part, 1).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
        v_relerr[v_abserr == 0.0] = 0.0
        assert v_relerr.shape == (n_partset, n_part)

        v_abserrmax = v_abserr.max().item()
        v_abserrmean = v_abserr.mean().item()
        v_relerrmax = v_relerr.max().item()
        v_relerrmean = v_relerr.mean().item()

        if verbose:
            print(f'  {vname + " " * (10-len(vname))}: Max Abs Error={v_abserrmax:0.03g}, ' +
                    f'Mean Abs Error={v_abserrmean:0.03g}, Max Rel Error={v_relerrmax:0.03g}, ' +
                    f'Mean Rel Error={v_relerrmean:0.03g}')

def test_smallnmaxdata(refr_prtcore, refr_prtshll, x_prtcore, x_prtshll, verbose=False):
    """
    This is a unit test for the torch version of the `coreshell` function. Here we want to see how a 
    small `n_max` impacts the results for large particle sizes.

    This function relies on real data provided in the input. See the `test_smallnmax` for the same 
    test with synthetic data.
    """

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64
    tch_cdtype = torch.complex128

    n_maxinfrd = (refr_prtshll.real * x_prtshll + 4.3 * 
        (refr_prtshll.real * x_prtshll).pow(1.0 / 3.0) + 3).to(dtype=torch.long).ravel()
    i_lrgprt, = torch.where(n_maxinfrd > 120)
    n_partlrg1, = i_lrgprt.shape

    ii1 = np.linspace(0, n_partlrg1 - 1, 1000).astype(int).tolist()
    i_lrgprt = i_lrgprt[n_maxinfrd[i_lrgprt].argsort()[ii1]]
    n_partlrg, = i_lrgprt.shape

    mc_lrg = refr_prtcore.ravel()[i_lrgprt]
    assert mc_lrg.shape == (n_partlrg,)
    ms_lrg = refr_prtshll.ravel()[i_lrgprt]
    assert ms_lrg.shape == (n_partlrg,)
    xc_lrg = x_prtcore.ravel()[i_lrgprt]
    assert xc_lrg.shape == (n_partlrg,)
    xs_lrg = x_prtshll.ravel()[i_lrgprt]
    assert xs_lrg.shape == (n_partlrg,)

    n_max = 120
    n_max1, n_max2 = n_maxinfrd[i_lrgprt[0]].item(), n_maxinfrd[i_lrgprt[-1]].item()
    if verbose:
        print(f'Testing the core-shell routine with {n_partlrg} real particles and n_max={n_max}:')
        print(f'  * {100 * n_partlrg1 / refr_prtcore.numel():.2f}' + '%' + 
            ' of the input population have inferred n_max larger than 120.')
        print(f'  * In other words, {n_partlrg1} particles have inferred n_max larger than 120.')
        print(f'  * The inferred n_max for those particles is between {n_max1} and {n_max2}.')

    # Running the small `n_max` results
    qe_lrg1, qs_lrg1, g_lrg1 = get_coreshell(mc_lrg,  ms_lrg, xc_lrg, xs_lrg, n_max, 
        tch_device, tch_dtype, tch_cdtype, void_qe=0.0, void_qs=0.0, void_g=0.0)

    # Running the large `n_max` results
    qe_lrg2, qs_lrg2, g_lrg2 = get_coreshell(mc_lrg,  ms_lrg, xc_lrg, xs_lrg, n_max * 10, 
        tch_device, tch_dtype, tch_cdtype, void_qe=0.0, void_qs=0.0, void_g=0.0)

    for vname, vpred, vref in [('qe', qe_lrg1, qe_lrg2), ('qs', qs_lrg1, qs_lrg2), ('g', g_lrg1, g_lrg2)]:
        pnorm_err = 2
        v_abserr = (vpred - vref).reshape(n_partlrg, 1).abs().pow(
            pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err)
        assert v_abserr.shape == (n_partlrg,)

        v_relerr = 2 * v_abserr / (
            vpred.reshape(n_partlrg, 1).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err) +
            vref.reshape(n_partlrg, 1).abs().pow(pnorm_err).sum(dim=-1).pow(1.0 / pnorm_err))
        v_relerr[v_abserr == 0.0] = 0.0
        assert v_relerr.shape == (n_partlrg,)

        v_abserrmax = v_abserr.max().item()
        v_abserrmean = v_abserr.mean().item()
        v_relerrmax = v_relerr.max().item()
        v_relerrmean = v_relerr.mean().item()

        if verbose:
            print(f'  {vname + " " * (10-len(vname))}: Max Abs Error={v_abserrmax:0.03g}, ' +
                    f'Mean Abs Error={v_abserrmean:0.03g}, Max Rel Error={v_relerrmax:0.03g}, ' +
                    f'Mean Rel Error={v_relerrmean:0.03g}')

if __name__ == '__main__':
    ###############################################################################################
    ########################## Running Unit Tests on All of the Functions #########################
    ###############################################################################################

    # Unit-testing the `dlnpsi` function
    test_dlnpsizeta(get_dlnpsi, get_dlnpsirefnp, func_name='dlnpsi', verbose=True)

    # Unit-testing the `dlnzeta` function
    test_dlnpsizeta(get_dlnzeta, get_dlnzetarefnp, func_name='dlnzeta', verbose=True)

    # Unit-testing the `tup` function
    test_tup(verbose=True)

    # Unit-testing the `qqg` function
    test_qqg(verbose=True)

    # Unit-testing the `sphere` function
    test_sphere(verbose=True)

    # Unit-testing the `core-shell` function without any `sphere` approximations
    test_coreshellnonsphr(verbose=True)

    # Unit-testing the general `core-shell` function
    test_coreshell(verbose=True)

    # Unit-testing the usage of small `n_max` for large particles on synthetic data
    # test_smallnmax(verbose=True)

    # Unit-testing the usage of small `n_max` for large particles on real data
    # test_smallnmaxdata(refr_prtcore, refr_prtshll, x_prtcore, x_prtshll, verbose=True)