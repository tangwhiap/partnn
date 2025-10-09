# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import importlib
# %matplotlib inline
# mplbackend = 'agg'
# assert mplbackend in ('agg', 'retina')
# if (mplbackend == 'agg'):
#     import matplotlib
#     matplotlib.use('Agg')
# elif (mplbackend == 'retina') and (importlib.util.find_spec("matplotlib_inline") is not None):
#     import matplotlib_inline
#     matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
# elif (mplbackend == 'retina'):
#     from IPython.display import set_matplotlib_formats
#     set_matplotlib_formats('retina')
# else:
#     raise ValueError(f'undefined matplotlib backend {mplbackend}')
# plt.rcParams.update({'figure.max_open_warning': 0})
# plt.ioff();

# +
import os
import torch
import numpy as np
import json
import yaml
import time
import math
import fnmatch
import pathlib
import warnings
import datetime
import tensorboardX
from torch import nn
from os.path import isdir
from os.path import exists
from textwrap import dedent
import matplotlib.pyplot as plt
from pyinstrument import Profiler

from partnn.io_cfg import PROJPATH
from partnn.io_cfg import data_dir
from partnn.io_cfg import configs_dir
from partnn.io_cfg import results_dir
from partnn.io_cfg import storage_dir
from partnn.io_utils import DataWriter
from partnn.io_utils import eval_formula
from partnn.io_utils import preproc_cfgdict
from partnn.io_utils import hie2deep, deep2hie
from partnn.io_utils import parse_refs, get_subdict, resio
from partnn.tch_utils import BatchRNG, EMA, profmem, make_nn
from partnn.tch_utils import BatchParamGrouper, BatchLRScheduler
from partnn.aerometrics import AeroMeasures

from ruamel import yaml as ruyaml
from partnn.notebooks.n20_utils import unflatten_cfg, get_snrtspltidxs, load_histdata, Timer
from partnn.notebooks.n20_utils import make_pp, Evaluation, Visualization, DataPrepper, get_spltcfgs 
from partnn.notebooks.n20_utils import get_lbldims, load_riopp, load_rioenc, VAELabelNN, IIDLossMaker


# -

# # Loading Config

# + tags=["active-ipynb"]
# json_cfgtreeid = '02_adhoc/20_mlpcnd.yml'
# # ! rm -rf "./20_vaehist/results/20_mlpcnd_00.h5"
# # ! rm -rf "./20_vaehist/storage/20_mlpcnd_00"
#
# json_cfgpath = f'{configs_dir}/{json_cfgtreeid}'
# if json_cfgpath.endswith('.json'):
#     with open(json_cfgpath, 'r') as fp:
#         json_cfgdict = json.load(fp, object_pairs_hook=dict)
# elif json_cfgpath.endswith('.yml'):
#     with open(json_cfgpath, "r") as fp:
#         json_cfgdict = dict(ruyaml.safe_load(fp))
# else:
#     raise RuntimeError(f'unknown config extension: {json_cfgpath}')
#
# json_cfgdict['io/config_id'] = '20_mlpcnd_00'
# json_cfgdict['io/config_id'] = '20_mlpcnd_00'
# json_cfgdict['io/results_dir'] = './20_vaehist/results'
# json_cfgdict['io/storage_dir'] = './20_vaehist/storage'
# json_cfgdict['io/tch/device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Applying the looping processes
# all_cfgdicts1 = preproc_cfgdict(json_cfgdict)[:10]
# # Parsing all the "/ref"-, "/def"-, and "/pop"-ending keys
# all_cfgdicts = [parse_refs(cfgdict, trnsfrmtn='hie', pathsep=' -> ', 
#     cfg_path=json_cfgtreeid) for cfgdict in all_cfgdicts1]
# # Dropping the trailing references section
# all_refdicts = [get_subdict(cfgdict, prefix='refs', pop=True) 
#     for cfgdict in all_cfgdicts]
#
# cfg_dict_input = all_cfgdicts[9]

# + tags=["active-py"]
def main(cfg_dict_input):


# -

    ###############################################################################
    ###################### Processing the Config Dictionary #######################
    ###############################################################################
    cfg_dict = cfg_dict_input.copy()

    #########################################################
    #################### Ignored Options ####################
    #########################################################
    cfgdesc = cfg_dict.pop('desc', None)
    cfgdate = cfg_dict.pop('date', None)

    #########################################################
    ################### Mandatory Options ###################
    #########################################################
    prob_type = cfg_dict.pop('problem')
    rng_seed_list = cfg_dict.pop('rng_seed/list')

    opt_type = cfg_dict.pop('opt/type')
    n_epochs = cfg_dict.pop('opt/epoch')
    n_mb = cfg_dict.pop('opt/mb/n')
    schdlr_type = cfg_dict.pop('opt/schdlr/type', None)
    schdlr_hparams = get_subdict(cfg_dict, 'opt/schdlr', pop=True)
    mb_rndmztn = cfg_dict.pop('opt/rndmztn')
    assert mb_rndmztn in ('shuffle', 'iid', 'none')
    opt_hparams = get_subdict(cfg_dict, 'opt', pop=True)

    #########################################################
    ###################### Data Options #####################
    #########################################################
    data_cfg = get_subdict(cfg_dict, 'data', pop=False)
    data_tree = cfg_dict.pop('data/tree')
    data_chems = cfg_dict.pop('data/chems', None)
    data_binsi1 =  cfg_dict.pop('data/bins/idx/start', 0)
    data_binsi2 =  cfg_dict.pop('data/bins/idx/end', None)

    split_cfg = get_subdict(cfg_dict, 'split', pop=True)
    #########################################################
    ################## Neural Spec Options ##################
    #########################################################
    nn_mdlscfg = get_subdict(cfg_dict, 'nn/modules', pop=True)
    pp_mdlscfg = get_subdict(cfg_dict, 'pp/modules', pop=True)
    asgn_mdlscfg = get_subdict(cfg_dict, 'asgn/modules', pop=True)
    metric_mdlscfg = hie2deep(get_subdict(cfg_dict, 'metric/modules', pop=True), maxdepth=1)

    enc_type = cfg_dict.pop('nn/enc/type')
    enc_hparamsraw = get_subdict(cfg_dict, 'nn/enc', pop=True)

    dec_type = cfg_dict.pop('nn/dec/type')
    dec_hparamsraw = get_subdict(cfg_dict, 'nn/dec', pop=True)

    ltnt_dimraw = cfg_dict.pop('nn/ltnt/dim')
    ltnt_sigtnsfm = cfg_dict.pop('nn/ltnt/sig/tnsfm', 'exp')

    ppnn_type = cfg_dict.pop('nn/pp/type')
    ppnn_hparamsraw = get_subdict(cfg_dict, 'nn/pp', pop=True)

    lblpp_type = cfg_dict.pop('nn/lbl/pp/type', None)
    lblpp_hparamsraw = get_subdict(cfg_dict, 'nn/lbl/pp', pop=True)

    lblnn_type = cfg_dict.pop('nn/lbl/nn/type', None)
    lblnn_hparamsraw = get_subdict(cfg_dict, 'nn/lbl/nn', pop=True)

    #########################################################
    ################### Training Criteria ###################
    #########################################################
    w_kl = cfg_dict.pop('cri/kl/w', None)
    w_klmu = cfg_dict.pop('cri/kl/mu/w', w_kl)
    w_klsig = cfg_dict.pop('cri/kl/sig/w', w_kl)

    rcnst_cri = cfg_dict.pop('cri/rcnst/metric')
    ppcri_inpspc = cfg_dict.pop('cri/rcnst/ppinp/space', 'x')
    ppcri_rndtrp = cfg_dict.pop('cri/rcnst/ppinp/rndtrp', True)
    ppcri_type = cfg_dict.pop('cri/rcnst/pp/type')
    ppcri_hparamsraw = get_subdict(cfg_dict, 'cri/rcnst/pp', pop=True)

    w_indzy = cfg_dict.pop('cri/indzy/w', None)
    n_histindzy = cfg_dict.pop('cri/indzy/hist/n', None)
    cri_indzymtrccfg = get_subdict(cfg_dict, 'cri/indzy/mtrc', pop=True)

    #########################################################
    ################## Evaluation Profiles ##################
    #########################################################
    evalcfgs = hie2deep(get_subdict(cfg_dict, 'eval', pop=True), maxdepth=1)
    perfcfgs = hie2deep(get_subdict(cfg_dict, 'perf', pop=True), maxdepth=1)
    vizcfgs = hie2deep(get_subdict(cfg_dict, 'viz', pop=True), maxdepth=1)

    mplopts_cfg = get_subdict(cfg_dict, 'mplopts', pop=True)
    mplopts_cfg = hie2deep(mplopts_cfg, maxdepth=1)

    #########################################################
    ################# I/O Logistics Options #################
    #########################################################
    config_id = cfg_dict.pop('io/config_id')
    results_dir = cfg_dict.pop('io/results_dir')
    storage_dir = cfg_dict.pop('io/storage_dir', None)
    io_avgfrq = cfg_dict.pop('io/avg/frq')
    ioflsh_period = cfg_dict.pop('io/flush/frq')
    chkpnt_period = cfg_dict.pop('io/ckpt/frq')
    device_name = cfg_dict.pop('io/tch/device')
    dtype_name = cfg_dict.pop('io/tch/dtype')
    iomon_period = cfg_dict.pop('io/mon/frq')
    io_cmprssnlvl = cfg_dict.pop('io/cmprssn_lvl')
    eval_bs = cfg_dict.pop('io/eval/bs', None)
    eval_bsnn = cfg_dict.pop('io/eval/bs/nn', eval_bs)
    eval_bspp = cfg_dict.pop('io/eval/bs/pp', eval_bs)
    eval_enbl = cfg_dict.pop('io/eval/enbl', True)
    do_logtb_ = cfg_dict.pop('io/strg/logtb', None)
    do_profile_ = cfg_dict.pop('io/strg/profile', None)
    do_savefigs_ = cfg_dict.pop('io/strg/savefigs', None)

    # disabling the evaluation if necessary.
    evalcfgs = evalcfgs if eval_enbl else dict()
    perfcfgs = perfcfgs if eval_enbl else dict()
    vizcfgs = vizcfgs if eval_enbl else dict()

    dtnow = datetime.datetime.now().isoformat(timespec='seconds')
    cfg_tree = '/'.join(config_id.split('/')[:-1])
    cfg_name = config_id.split('/')[-1]

    # Making sure no other options are left unused.
    if len(cfg_dict) > 0:
        msg_ = 'The following settings were left unused:\n'
        for key, val in cfg_dict.items():
            msg_ += f'  {key}: {val}'
        raise RuntimeError(msg_)

    ###############################################################################
    ################ Initializing the Training Classes and Objects ################
    ###############################################################################
    # Derived options and assertions
    eval_bsnn = n_mb if eval_bsnn is None else eval_bsnn
    eval_bspp = n_mb if eval_bspp is None else eval_bspp

    assert (lblnn_type is None) == (lblpp_type is None), dedent(f'''
        A label {"pp" if (lblpp_type is not None) else "nn"} was provided, 
        but a label {"nn" if (lblpp_type is not None) else "pp"} was not 
        specified. Either both or neither of them have to be provided.''')

    is_condvae = (lblpp_type is not None)
    if is_condvae:
        assert lblnn_type is not None

    if is_condvae and (lblnn_type == 'pretrained'):
        lblnn_hparams = lblnn_hparamsraw.copy()
        lblnn_enctype = lblnn_hparams.pop('enc/type')
        assert lblnn_enctype == 'vae', f'unsupported lbl/nn/enc/type = {lblnn_enctype}'
        y_sigscale = lblnn_hparams.pop('enc/sigscale')
        y_nodename = lblnn_hparams.pop('enc/nodename')
    elif is_condvae:
        lblnn_enctype, lblnn_hparams, y_sigscale = 'plain', None, 0
    else:
        lblnn_enctype, lblnn_hparams, y_sigscale = None, None, None

    #########################################################
    ########### I/O-Related Options and Operations ##########
    #########################################################
    name2dtype = dict(float64=(torch.double, torch.complex128), 
        float32=(torch.float32, torch.complex64),
        float16=(torch.float16, torch.complex32))
    tch_device = torch.device(device_name)
    tch_dtype, tch_cdtype = name2dtype[dtype_name]

    has_storage = storage_dir is not None
    do_logtb = bool(do_logtb_) if do_logtb_ is not None else has_storage
    do_profile = bool(do_profile_) if do_profile_ is not None else has_storage
    do_savefigs = bool(do_savefigs_) if do_savefigs_ is not None else has_storage

    do_logtb = do_logtb and has_storage
    do_profile = do_profile and has_storage
    do_savefigs = do_savefigs and has_storage

    assert not(do_logtb) or (storage_dir is not None)
    assert not(do_profile) or (storage_dir is not None)
    assert not(do_savefigs) or (storage_dir is not None)

# # Data Loading

    #########################################################
    ################ Defining the VAE Problem ###############
    #########################################################
    assert prob_type == 'vaehist'

    data_path = f'{data_dir}/{data_tree}'
    assert data_path.endswith('.nc')

    data_hist = load_histdata(data_path, data_chems, data_binsi1, data_binsi2)
    
    n_snr = data_hist['n_snr']
    n_t = data_hist['n_t']
    n_chem = data_hist['n_chem']
    n_bins = data_hist['n_bins']
    n_waveref = data_hist['n_waveref']

    m_chmprthst = data_hist['m_chmprthst']
    assert m_chmprthst.shape == (n_snr, n_t, n_chem, n_bins)
    m_prthst = data_hist['m_prthst']
    assert m_prthst.shape == (n_snr, n_t, n_bins)
    n_prthst = data_hist['n_prthst']
    assert n_prthst.shape == (n_snr, n_t, n_bins)
    rho_chm = data_hist['rho_chm']
    assert rho_chm.shape == (n_chem,)
    kappa_chm = data_hist['kappa_chm']
    assert kappa_chm.shape == (n_chem,)
    len_wvref = data_hist['len_wvref']
    assert len_wvref.shape == (n_waveref,)
    refr_wvchmref = data_hist['refr_wvchmref']
    assert refr_wvchmref.shape == (n_waveref, n_chem)
    chem_species = data_hist['chem_species']
    assert len(chem_species) == n_chem
    d_histbins = data_hist['d_histbins']
    assert d_histbins.shape == (n_bins + 1,)

    #######################################
    #   Making the Data Variables Dict    #
    #######################################
    # `datanp_dict` is an input variable (str) to array (np.ndarray) mapping.
    datanp_dict = dict(
        n_prthst=n_prthst.reshape(n_snr * n_t, 1, n_bins),
        m_prthst=m_prthst.reshape(n_snr * n_t, 1, n_bins),
        m_chmprthst=m_chmprthst.reshape(n_snr * n_t, n_chem, n_bins))

    # `aero_cstinfo` contains the aerosol constants used within ccn, inp, and optical evaluation metrics
    aero_cstinfo = dict(chem_species=chem_species, n_chem=n_chem, n_bins=n_bins, n_waveref=n_waveref)
    aero_cstinfo['rho_chm'] = torch.from_numpy(rho_chm).to(device=tch_device, dtype=tch_dtype)
    aero_cstinfo['kappa_chm'] = torch.from_numpy(kappa_chm).to(device=tch_device, dtype=tch_dtype)
    aero_cstinfo['len_wvref'] = torch.from_numpy(len_wvref).to(device=tch_device, dtype=tch_dtype)
    aero_cstinfo['refr_wvchmref'] = torch.from_numpy(refr_wvchmref).to(device=tch_device, dtype=tch_cdtype)

    # `data_dict` is an input variable (str) to tensor (torch.tensor) mapping.
    data_dict = {key: torch.from_numpy(varnp).to(device=tch_device, dtype=tch_dtype)
        for key, varnp in datanp_dict.items()}

    # Preparing the data hash for later caching
    hash_data = {key: data_cfg for key in data_dict}

    # Example:
    #   data_dims == {
    #       'n_prthst': (1, n_bins),
    #       'm_prthst': (1, n_bins),
    #       'm_chmprthst': (n_chem, n_bins)
    #   }
    data_dims = {key: tuple(varnp.shape[1:]) for key, varnp in datanp_dict.items()}

    # The shape variables
    shapevars = dict(n_snr=n_snr, n_t=n_t, n_chem=n_chem, n_bins=n_bins,
        chem_species=chem_species, device=device_name, dtype=dtype_name)

    # Adding the aeromeasures pp config
    pp_mdlscfg['aeromeasures'] = {'class': AeroMeasures, 'hparams/aerocst': aero_cstinfo}

    #########################################################
    ########### Constructing the Batch RNG Object ###########
    #########################################################
    n_seeds = len(rng_seed_list)
    rng_seeds = np.array(rng_seed_list)
    rng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=1_000_000,
        norm_cache_cols=5_000_000)
    rng.seed(np.broadcast_to(rng_seeds, rng.shape))

    erng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=0,
        norm_cache_cols=0)
    erng.seed(np.broadcast_to(rng_seeds, erng.shape))

    vrng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=0,
        norm_cache_cols=0)
    vrng.seed(np.broadcast_to(rng_seeds, vrng.shape))

    mrng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=0,
        norm_cache_cols=0)
    mrng.seed(np.broadcast_to(rng_seeds, mrng.shape))

    #########################################################
    ######## Instantiating Train/Test Split Indecis #########
    #########################################################
    split_vars = split_cfg['vars']
    split_idxdict = get_snrtspltidxs(split_cfg, n_snr, n_t, n_seeds,
        rng, tch_device, pop_opts=True)
    trn_spltidxs = split_idxdict['split_idxs']
    n_trn = trn_spltidxs.shape[-1]
    n_tst = n_snr * n_t - n_trn
    assert trn_spltidxs.shape == (n_seeds, n_trn)
    tst_spltidxs = split_idxdict['negsplit_idxs']
    assert tst_spltidxs.shape == (n_seeds, n_tst)

    #######################################
    #  Constructing the NN pre-processor  #
    #######################################
    ppnn_hparams = {hparam: eval_formula(val, shapevars, catch=True) for hparam, val in ppnn_hparamsraw.items()}
    nn_pp = make_pp(ppnn_type, ppnn_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)
    nn_pp.infer(data_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
    un_dims = nn_pp.get_dims(data_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
    xn_dims = nn_pp.get_dims(data_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)

    datannpp_dims = {**data_dims, **un_dims}
    ppcri_hparams = {hparam: eval_formula(val, shapevars, catch=True) for hparam, val in ppcri_hparamsraw.items()}
    cri_pp = make_pp(ppcri_type, ppcri_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)
    cri_pp.infer(data_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
    uc_dims = cri_pp.get_dims(datannpp_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
    xc_dims = cri_pp.get_dims(datannpp_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)

    lbl_pp, ul_dims, xl_dims = None, dict(), dict()
    if is_condvae and (lblpp_type == 'pretrained'):
        fpidx_lblpp = lblpp_hparamsraw.pop('fpidx')
        resdir_lblpp = lblpp_hparamsraw.pop('resdir', None)
        rio_lbl = resio(fpidx=fpidx_lblpp, resdir=resdir_lblpp)
        lbl_ppinfo = load_riopp(rio_lbl, device_name, dtype_name, tch_device, tch_dtype, tch_cdtype)
        rng_seed_list_lblpp, lbl_pp = lbl_ppinfo.pop('rng_seed_list'), lbl_ppinfo.pop('pp')
        assert rng_seed_list == rng_seed_list_lblpp, f'loaded label pp rng seeds do not match'
        xl_dims = lbl_pp.get_dims(data_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)
        ul_dims = lbl_pp.get_dims(data_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
        assert len(lblpp_hparamsraw) == 0, f'unused options: {lblpp_hparamsraw}'
    elif is_condvae:
        lblpp_hparams = {hparam: eval_formula(val, shapevars, catch=True) for hparam, val in lblpp_hparamsraw.items()}
        lbl_pp = make_pp(lblpp_type, lblpp_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)
        lbl_pp.infer(data_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
        xl_dims = lbl_pp.get_dims(data_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)
        ul_dims = lbl_pp.get_dims(data_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)

    for x_node, x_dtupl in xl_dims.items():
        x_dtupn = xn_dims.get(x_node, x_dtupl)
        assert x_dtupl == x_dtupn
    xnl_dims = {**xn_dims, **xl_dims}

    #########################################################
    ########### Evaluation Point Sampling Options ###########
    #########################################################
    eparams = dict()
    for eid, eopts in evalcfgs.items():
        epp_cfg = dict()

        # TODO: The inferred parameters must be run on $f^-1(f(x))$ and
        # not $x$ directly for the baseline shifts and scales to be most
        # accurate in practice. That being said, this seems good enough.
        epp_type = eopts.pop('pp/type')
        epp_inpspc = eopts.pop('ppinp/space', 'x')
        epp_rndtrp = eopts.pop('ppinp/rndtrp', True)
        epp_hparamsraw = get_subdict(eopts, 'pp', pop=True)
        
        epp_hparams = {hparam: eval_formula(val, shapevars, catch=True) 
            for hparam, val in epp_hparamsraw.items()}
        epp = make_pp(epp_type, epp_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)

        epp.infer(data_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
        ue_dims = epp.get_dims(datannpp_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
        xe_dims = epp.get_dims(datannpp_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)

        # Getting the split configuration dictionary
        n_evlpnts = eopts.pop('n', None)
        e_sigscale = eopts.pop('sigscale', None)
        e_spltcfgsraw = get_subdict(eopts, 'splits', pop=True)
        e_spltcfgs = get_spltcfgs(e_spltcfgsraw, eid, n_evlpnts, e_sigscale, is_condvae, lblnn_enctype)

        # Determining the split data indecis `e_spltidxs`
        for e_split, e_spltcfg in e_spltcfgs.items():
            if e_split == 'train':
                e_spltidxs = trn_spltidxs
            elif e_split == 'test':
                e_spltidxs = tst_spltidxs
            elif e_split == 'snr':
                with torch.no_grad():
                    e_snridxlst_ = eopts.pop('snr/idxs')
                    n_esnridxlst = len(e_snridxlst_)

                    assert all(ii_snr < n_snr for ii_snr in e_snridxlst_)
                    assert all(ii_snr >= -n_snr for ii_snr in e_snridxlst_)

                    # Translating negative scenario indices
                    e_snridxlst = [ii_snr % n_snr for ii_snr in e_snridxlst_]
                    n_esplit = n_esnridxlst * n_t

                    e_splitidxs1 = torch.tensor(e_snridxlst).to(device=tch_device, dtype=torch.long)
                    assert e_splitidxs1.shape == (n_esnridxlst,)

                    ii_t = torch.arange(n_t, device=tch_device, dtype=torch.long)
                    assert ii_t.shape == (n_t,)

                    e_splitidxs2 = n_t * e_splitidxs1.reshape(n_esnridxlst, 1) + ii_t.reshape(1, n_t)
                    assert e_splitidxs2.shape == (n_esnridxlst, n_t)

                    e_splitidxs3 = e_splitidxs2.reshape(n_esplit)
                    assert e_splitidxs3.shape == (n_esplit,)

                    e_spltidxs = e_splitidxs3.reshape(1, n_esplit).expand(n_seeds, n_esplit)
                    assert e_spltidxs.shape == (n_seeds, n_esplit)
            elif e_split == 'normal':
                e_spltidxs = None
            else:
                raise ValueError(f'undefined e_split={e_split}')
            
            assert e_spltcfg['idxs'] is None
            e_spltcfg['idxs'] = e_spltidxs
            assert e_spltcfg['iidxs'] is None
            e_spltcfg['iidxs'] = None

        eparams[eid] = dict()
        eparams[eid]['pp'] = epp
        eparams[eid]['inpspc'] = epp_inpspc
        eparams[eid]['rndtrp'] = epp_rndtrp
        eparams[eid]['udims'] = ue_dims
        eparams[eid]['xdims'] = xe_dims
        eparams[eid]['frq'] = eopts.pop('frq')
        eparams[eid]['rndmztn'] = eopts.pop('rndmztn')
        eparams[eid]['store'] = get_subdict(eopts, 'store', pop=True)
        eparams[eid]['splits'] = e_spltcfgs

        assert len(eopts) == 0, f'unused eval options for {eid}: {eopts}'


    # Piping the matplotlib-related options dictionary
    for vid, vcfg in vizcfgs.items():
        mplopt_tmplts = vcfg.pop('mplopts/plan', [])
        if isinstance(mplopt_tmplts, str):
            mplopt_tmplts = [mplopt_tmplts]
        v_mplopts = dict()
        for mpo_tmplt in mplopt_tmplts:
            assert mpo_tmplt in mplopts_cfg, dedent(f'''
                The Matplotlib option template "{mpo_tmplt}" 
                visualization "{vid}". However, this template
                was never defined:
                    available templates: {mplopts_cfg.keys()}
            ''')
            v_mplopts.update(mplopts_cfg[mpo_tmplt])
        mpo_ovrrd = get_subdict(vcfg, 'mplopts', pop=True)
        mpo_ovrrd = dict() if mpo_ovrrd is None else mpo_ovrrd.copy()
        v_mplopts.update(mpo_ovrrd)
        v_mplopts['type'] = vcfg['type']
        vcfg['mplopts'] = v_mplopts
        vcfg.setdefault('splits', None)

    # Converting str values to tuples
    for prf_id in perfcfgs:
        prf_cartcfg = perfcfgs[prf_id]
        assert set(prf_cartcfg) == {'srctrg', 'asgn', 'node', 'metric', 'stat'}
        prf_cartcfg = {optn: (val,) if isinstance(val, str) else val 
            for optn, val in prf_cartcfg.items()}
        perfcfgs[prf_id] = prf_cartcfg

    # The latent dimension
    ltnt_dim = eval_formula(ltnt_dimraw, shapevars, catch=True)

    # Initializing the encoder
    enc_hparams = dict()
    for hparam, val in enc_hparamsraw.items():
        enc_hparams[hparam] = eval_formula(val, shapevars, catch=True)

    encoder = make_nn(name=enc_type, hparams=enc_hparams, config=nn_mdlscfg, 
        n_seeds=n_seeds, rng=rng, device=tch_device, dtype=tch_dtype)

    # Initializing the decoder
    dec_hparams = dict()
    for hparam, val in dec_hparamsraw.items():
        dec_hparams[hparam] = eval_formula(val, shapevars, catch=True)

    decoder = make_nn(name=dec_type, hparams=dec_hparams, config=nn_mdlscfg, 
        n_seeds=n_seeds, rng=rng, device=tch_device, dtype=tch_dtype)

    # Setting the optimizer
    model_parameters = list(encoder.parameters()) + list(decoder.parameters())

    # Getting the seed-splitted parameters
    if schdlr_type is not None:
        pgrouper = BatchParamGrouper(params=model_parameters, shape=(n_seeds,))
        optim_params = pgrouper.param_groups
    else:
        pgrouper = None
        optim_params = model_parameters

    # Instantiating the optimizer
    if opt_type == 'adam':
        opt = torch.optim.Adam(optim_params, **opt_hparams)
    elif opt_type == 'sgd':
        opt = torch.optim.SGD(optim_params, **opt_hparams)
    else:
        raise NotImplementedError(f'opt/dstr="{opt_type}" not implmntd')

    # Instantiating the learning rate scheduler
    if schdlr_type is not None:
        scheduler = BatchLRScheduler(kind=schdlr_type, hparams=schdlr_hparams, 
            optimizer=opt, n_seeds=n_seeds, n_grps=pgrouper.n_grps)
    else:
        scheduler = None

    # Initializing the label network
    lblnn, y_dims = None, dict()
    if is_condvae and (lblnn_type == 'pretrained'):
        fpidx_lblnn = lblnn_hparams.pop('fpidx')
        resdir_lblnn = lblnn_hparams.pop('resdir', None)

        rio_lblnn = resio(fpidx=fpidx_lblnn, resdir=resdir_lblnn)
        lblnn_info = load_rioenc(rio_lblnn, device_name, dtype_name, tch_device, tch_dtype)

        rng_seed_list_lblnn = lblnn_info.pop('rng_seed_list')
        assert rng_seed_list == rng_seed_list_lblnn, f'loaded label nn rng seeds do not match'

        assert lblnn_enctype == 'vae'
        lblnn = VAELabelNN(**lblnn_info, y_node=y_nodename)

        tmprng = BatchRNG(shape=(n_seeds,), lib='torch', device=tch_device, dtype=tch_dtype,
            unif_cache_cols=0, norm_cache_cols=0)
        tmprng.seed(np.broadcast_to(rng_seeds, rng.shape))
        lblnnkwsfk = dict(n_seeds=n_seeds, n_mb=n_mb, rng=tmprng, sigscale=y_sigscale)  
        y_dims = get_lbldims(lblnn, ul_dims, n_seeds, n_mb, tch_device, tch_dtype, lblnnkws=lblnnkwsfk)
        
        assert len(lblnn_hparams) == 0, f'unused options: {lblnn_hparams}'
    elif is_condvae:
        assert (lblnn_enctype, lblnn_hparams) == ('plain', None)
        lblnn_hparams = dict()
        for hparam, val in lblnn_hparamsraw.items():
            lblnn_hparams[hparam] = eval_formula(val, shapevars, catch=True)
            
        lblnn = make_nn(name=lblnn_type, hparams=lblnn_hparams, config=nn_mdlscfg,
            n_seeds=n_seeds, rng=rng, device=tch_device, dtype=tch_dtype)
        y_dims = get_lbldims(lblnn, ul_dims, n_seeds, n_mb, tch_device, tch_dtype)
        assert len(y_dims) > 0

    if w_indzy is not None:
        assert is_condvae, dedent(f'''
            An iid z and y loss weight was provided, but the 
            model is non-conditional. The iid z and y loss can 
            only make sense for conditional vae models:
                cri/indzy/w: {w_indzy}''')
        iid_lossfinder = IIDLossMaker(n_histindzy, cri_indzymtrccfg, rng, tch_device, tch_dtype)
        # If you're in absolute need of efficiency, you can set `has_indzyloss = (w_indzy != 0)`
        has_indzyloss = True
    else:
        assert n_histindzy is None, f'undefined cri/indzy/hist/n={n_histindzy}'
        assert len(cri_indzymtrccfg) == 0, dedent(f'''
            unused cri/indzy/mtrc options: {cri_indzymtrccfg}''')
        iid_lossfinder, has_indzyloss = None, False


    if results_dir is not None:
        pathlib.Path(os.sep.join([results_dir, cfg_tree])
                    ).mkdir(parents=True, exist_ok=True)

    if storage_dir is not None:
        cfgstrgpnt_dir = os.sep.join([storage_dir, cfg_tree, cfg_name])
        pathlib.Path(cfgstrgpnt_dir).mkdir(parents=True, exist_ok=True)
        strgidx = sum(isdir(f'{cfgstrgpnt_dir}/{x}') for x in os.listdir(cfgstrgpnt_dir))
        dtnow_ = dtnow[2:].replace('-', '').replace(':', '').replace('.', '')
        cfgstrg_dir = f'{cfgstrgpnt_dir}/{strgidx:02d}_{dtnow_}'
        pathlib.Path(cfgstrg_dir).mkdir(parents=True, exist_ok=True)

        yaml.Dumper.ignore_aliases = lambda *args: True
        with open(f'{cfgstrg_dir}/config.yml', 'w') as fp:
            yaml.dump(cfg_dict_input, fp, sort_keys=False, default_flow_style=None)
    else:
        cfgstrg_dir = None

    if do_savefigs:
        assert cfgstrg_dir is not None
        fig_dir = f'{cfgstrg_dir}/figures'
        pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
    else:
        fig_dir = None

    if do_logtb:
        import logging
        if 'tbwriter' in locals():
            locals()['tbwriter'].close()
        tbwriter = tensorboardX.SummaryWriter(cfgstrg_dir)
        logging.getLogger("tensorboardX.x2num").setLevel(logging.CRITICAL)
    else:
        tbwriter = None
        
    if do_profile:
        profiler = Profiler()
        profiler.start()
    else:
        profiler = None

    # Data writer construction
    hdfpth = None
    if results_dir is not None:
        hdfpth = f'{results_dir}/{cfg_tree}/{cfg_name}.h5'
    dwriter = DataWriter(flush_period=ioflsh_period*n_seeds, 
        compression_level=io_cmprssnlvl)

    # Data Prepper Construction
    dprepper = DataPrepper(n_seeds, rng_seed_list, io_avgfrq, chkpnt_period,
        iomon_period, device_name, dtype_name, dtnow, cfg_tree, cfg_name)

    # Identifying the hyper-parameter from etc config columns
    hppats = ['problem', 'opt/*', 'data/tree', 'split/*', 'nn/*', 'cri/*', 'eval/*']

    etcpats = ['desc', 'date', 'rng_seed/list', 'data/*', 'io/*', 'eval/*/viz/*/mplopts/*', 
        'nn/modules/*', 'pp/modules/*', 'mplopts/*', 'perf/*', 'metric/modules/*', 
        'asgn/modules/*', 'viz/*']

    # Making the hyper-param and etc dictionaries
    hp_dict, etc_dict = DataPrepper.make_hpetcopts(cfg_dict_input, 
        hppats, etcpats, n_seeds, device_name, dtnow)

    # Pushing the static output data once
    # TODO: store the label static nn parameters
    pp_statics = {'mdl/cri': cri_pp, 'mdl/nn': nn_pp}
    pp_statics.update({f'eval/{eid}': eopts['pp'] for eid, eopts in eparams.items()})
    if is_condvae:
        pp_statics.update({'mdl/lbl': lbl_pp})
    dprepper.push_static(hp_dict, etc_dict, cfg_dict_input, shapevars, pp_statics)

    # Evaluation tools
    trn_sttime = time.time()
    dprepper.trn_sttime = trn_sttime
    last_perfdict = dict()
    ema = EMA(gamma=0.999, gamma_sq=0.998)
    timer = Timer()

# # Training Loop

# + editable=true slideshow={"slide_type": ""}
    ###############################################################################
    ######################## Training and Evaluation Loop #########################
    ###############################################################################
    ii_trn = None
    for epoch in range(n_epochs + 1):
        timer.tic('train')

        encoder.train()
        decoder.train()

        opt.zero_grad()

        #######################################
        #  Sampling training data and indices #
        #######################################

        # First, we sample random indices to select from `trn_spltidxs`.
        iistrt, iiend = epoch * n_mb, (epoch + 1) * n_mb
        ii_mbidxs_ = torch.arange(iistrt, iiend, device=tch_device).remainder(n_trn).to(dtype=torch.long)
        assert ii_mbidxs_.shape == (n_mb,)

        renew_ii = (ii_mbidxs_ == 0).any().item()
        if renew_ii and (mb_rndmztn == 'none'):
            ii_trn_ = torch.arange(n_trn, device=tch_device, dtype=torch.long)
            assert ii_trn_.shape == (n_trn,)

            ii_trn = ii_trn_.reshape(1, n_trn).expand(n_seeds, n_trn)
            assert ii_trn.shape == (n_seeds, n_trn)
        if renew_ii and (mb_rndmztn == 'shuffle'):
            ii_trn_ = rng.rand(n_seeds, n_trn)
            assert ii_trn_.shape == (n_seeds, n_trn)

            ii_trn = ii_trn_.argsort(dim=-1).detach()
            assert ii_trn.shape == (n_seeds, n_trn)
        elif renew_ii and (mb_rndmztn == 'iid'):
            ii_trn = (rng.rand(n_seeds, n_trn) * n_trn).to(torch.long)
            assert ii_trn.shape == (n_seeds, n_trn)
        elif not renew_ii:
            assert ii_trn.shape == (n_seeds, n_trn)
        else:
            raise ValueError(f'undefined mini-batch randomization = {mb_rndmztn}')

        ii_mbidxs = ii_mbidxs_.reshape(1, n_mb).expand(n_seeds, n_mb)
        assert ii_mbidxs.shape == (n_seeds, n_mb)

        ii_mb = torch.take_along_dim(ii_trn, ii_mbidxs, dim=-1)
        assert ii_mb.shape == (n_seeds, n_mb)

        # First, we sample random indices to select from `trn_spltidxs`.
        ii_mb = (rng.rand(n_seeds, n_mb) * n_trn).to(torch.long)
        assert ii_mb.shape == (n_seeds, n_mb)

        # Next, we select those from the training split indices.
        i_mb = torch.take_along_dim(trn_spltidxs, ii_mb, dim=1)
        assert i_mb.shape == (n_seeds, n_mb)

        # Now, we collect these indices from the full data.
        xnl_mbs = dict()
        for x_node, (n_xchnls, n_xlen) in xnl_dims.items():
            x_mb_ = data_dict[x_node][i_mb.ravel()]
            assert x_mb_.shape == (n_seeds * n_mb, n_xchnls, n_xlen)
            x_mb = x_mb_.reshape(n_seeds, n_mb, n_xchnls, n_xlen)
            assert x_mb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)
            xnl_mbs[x_node] = x_mb

        # The encoder's pp input mini-batches
        xn_mbs = {x_node: xnl_mbs[x_node] for x_node in xn_dims}
        # The labeler's pp input mini-batches
        xl_mbs = {x_node: xnl_mbs[x_node] for x_node in xl_dims}

        # Next, we apply the input pre-processing transformations
        # before feeding it to the encoder.
        un_mbs, u_shaper = nn_pp.forward(xn_mbs, full=False)
        for u_node, (n_uchnls, n_ulen) in un_dims.items():
            u_mb = un_mbs[u_node]
            assert u_mb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

        #######################################
        #         Applying the Labeler        #
        #######################################
        if is_condvae:
            with torch.no_grad():
                ul_mbs, u_shaper = lbl_pp.forward(xl_mbs, full=False)
                for u_node, (n_uchnls, n_ulen) in ul_dims.items():
                    u_mb = ul_mbs[u_node]
                    assert u_mb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

                lblnnkws = dict()
                if lblnn_enctype == 'vae':
                    lblnnkws.update(n_seeds=n_seeds, n_mb=n_mb, rng=rng, sigscale=y_sigscale)
                assert lblnn_enctype in ('plain', 'vae')
            
                y_mbs = lblnn(**ul_mbs, **lblnnkws)
                assert isinstance(y_mbs, dict)
        else:
            y_mbs = dict()

        for y_node, (n_ychnls, n_ylen) in y_dims.items():
            y_mb = y_mbs[y_node]
            assert y_mb.shape == (n_seeds, n_mb, n_ychnls, n_ylen)

        #######################################
        #         Applying the Encoder        #
        #######################################
        # Encoding the pre-processed input
        u_enc_ = encoder(**un_mbs, **y_mbs)
        assert u_enc_.shape == (n_seeds, n_mb, ltnt_dim * 2)

        u_enc = u_enc_.reshape(n_seeds, n_mb, 2, ltnt_dim)
        assert u_enc.shape == (n_seeds, n_mb, 2, ltnt_dim)

        # Decoupling the mu and logsigma values out of the encoder
        mu_mb = u_enc[:, :, 0, :]
        assert mu_mb.shape == (n_seeds, n_mb, ltnt_dim)

        if ltnt_sigtnsfm == 'exp':
            logsigma_mb = u_enc[:, :, 1, :]
            assert logsigma_mb.shape == (n_seeds, n_mb, ltnt_dim)
        elif ltnt_sigtnsfm == 'explin':
            logsigtlde_mb = u_enc[:, :, 1, :]
            assert logsigtlde_mb.shape == (n_seeds, n_mb, ltnt_dim)

            logsigma_mb = torch.where(logsigtlde_mb < 0, logsigtlde_mb, 
                (logsigtlde_mb + 1).log())
            assert logsigma_mb.shape == (n_seeds, n_mb, ltnt_dim)
        else:
            raise ValueError(f'undefined ltnt_sigtnsfm={ltnt_sigtnsfm}')

        #######################################
        ##### Computing the IID Z/Y Loss ######
        #######################################
        loss_indzy, loss_indzyunit = 0, 0
        if has_indzyloss:
            loss_indzyunit = iid_lossfinder(z_mbs={'mu': mu_mb}, 
                y_mbs=y_mbs, n_seeds=n_seeds, n_mb=n_mb)
            assert loss_indzyunit.shape == (n_seeds,)
            loss_indzy = w_indzy * loss_indzyunit
            assert loss_indzy.shape == (n_seeds,)

        #######################################
        #     Computing the KL-Divergences    #
        #######################################
        # Computing the KL-Divergences to the Normal prior in the latent space
        logvar_mb = (2 * logsigma_mb)
        assert logvar_mb.shape == (n_seeds, n_mb, ltnt_dim)

        sigmasq_mb = logvar_mb.exp()
        assert sigmasq_mb.shape == (n_seeds, n_mb, ltnt_dim)

        musq_mb = mu_mb.square()
        assert musq_mb.shape == (n_seeds, n_mb, ltnt_dim)

        kl_mumb = musq_mb.sum(dim=-1) / 2.0
        assert kl_mumb.shape == (n_seeds, n_mb)

        kl_sigmb = (sigmasq_mb - 1.0 - logvar_mb).sum(dim=-1) / 2.0
        assert kl_sigmb.shape == (n_seeds, n_mb)

        loss_klmu = kl_mumb.mean(dim=-1)
        assert loss_klmu.shape == (n_seeds,)

        loss_klsig = kl_sigmb.mean(dim=-1)
        assert loss_klsig.shape == (n_seeds,)

        loss_kl = loss_klmu * w_klmu + loss_klsig * w_klsig
        assert loss_kl.shape == (n_seeds,)

        #######################################
        #         Applying the Decoder        #
        #######################################
        eps_mb = rng.randn(n_seeds, n_mb, ltnt_dim)
        assert eps_mb.shape == (n_seeds, n_mb, ltnt_dim)

        z_mb = mu_mb + eps_mb * logsigma_mb.exp()
        assert z_mb.shape == (n_seeds, n_mb, ltnt_dim)

        unhat_mbs = decoder(z=z_mb, **y_mbs)
        for u_node, (n_uchnls, n_ulen) in un_dims.items():
            uhat_mb = unhat_mbs[u_node]
            assert uhat_mb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

        xnhat_mbs = None
        if ppcri_inpspc == 'x':
            xnhat_mbs = nn_pp.inverse(unhat_mbs, u_shaper, strict=True, full=False)
            for x_node, (n_xchnls, n_xlen) in xn_dims.items():
                x_mb = xnhat_mbs[x_node]
                assert x_mb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)

        #######################################
        #   Finding the Reconstruction Loss   #
        #######################################
        # Finding the right input to the criterion pre-processor
        if (ppcri_inpspc == 'x') and (ppcri_rndtrp == True):
            # In principle, `x_tildembs` should be identical to `xn_mbs`.
            # However, since some I/O transformations may not be truly
            # one-to-one, or due to numerical errors, they can be different.
            # Since this is not the VAE's fault, this should not exhibit
            # itself in the reconstruction loss, which is why we use the
            # `x_tildembs` tensor instead of `x_pln` when computing the
            # reconstruction loss.
            x_tildembs = nn_pp.inverse(un_mbs, u_shaper, strict=True, full=False)
            for x_node, (n_xchnls, n_xlen) in xn_dims.items():
                x_mb = x_tildembs[x_node]
                assert x_mb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)
            cri_inpmbs, cri_inphatmbs = x_tildembs, xnhat_mbs
        elif (ppcri_inpspc == 'x') and (ppcri_rndtrp == False):
            cri_inpmbs, cri_inphatmbs, x_tildembs = xn_mbs, xnhat_mbs, xn_mbs
        elif (ppcri_inpspc == 'u'):
            cri_inpmbs, cri_inphatmbs, x_tildembs = un_mbs, unhat_mbs, None
        else:
            raise ValueError(f'undefined ppcri_inpspc={ppcri_inpspc}')

        # Applying the criterion pre-processor to the original
        u_cmbs, u_cshaper = cri_pp.forward(cri_inpmbs, full=False)
        for u_node, (n_uchnls, n_ulen) in uc_dims.items():
            u_mb = u_cmbs[u_node]
            assert u_mb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

        # Applying the criterion pre-processor to the reconstruction
        uhat_cmbs, uhat_cshaper = cri_pp.forward(cri_inphatmbs, full=False)
        for u_node, (n_uchnls, n_ulen) in uc_dims.items():
            uhat_cmb = uhat_cmbs[u_node]
            assert uhat_cmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

        # Finding the reconstruction loss
        u_lossrcnsts = dict()
        for u_node, (n_uchnls, n_ulen) in uc_dims.items():
            u_cmb = u_cmbs[u_node]
            assert u_cmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

            uhat_cmb = uhat_cmbs[u_node]
            assert uhat_cmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

            uhat_err = u_cmb - uhat_cmb
            assert uhat_err.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

            if rcnst_cri == 'mse':
                u_lossrcnst = 0.5 * uhat_err.square().mean(dim=[-1, -2, -3])
                assert u_lossrcnst.shape == (n_seeds,)
            elif rcnst_cri == 'mae':
                u_lossrcnst = uhat_err.abs().mean(dim=[-1, -2, -3])
                assert u_lossrcnst.shape == (n_seeds,)
            else:
                raise ValueError(f'undefined rcnst_cri={rcnst_cri}')

            u_lossrcnsts[u_node] = u_lossrcnst

        loss_rcnst = torch.stack(list(u_lossrcnsts.values()), dim=1).sum(dim=1)
        assert loss_rcnst.shape == (n_seeds,)

        loss = loss_kl + loss_rcnst + loss_indzy
        assert loss.shape == (n_seeds,)

        loss_sum = loss.sum()
        assert loss_sum.shape == tuple()
        assert not loss_sum.isnan()

        loss_sum.backward()

        # We will not update in the first epoch so that we will 
        # record the initialization statistics as well. Instead, 
        # we will update an extra epoch at the end.
        if epoch > 0:
            if pgrouper is not None:
                pgrouper.attach_grads()
            opt.step()

        if scheduler is not None:
            scheduler.step(metrics=loss, epoch=epoch)
        
        encoder.eval()
        decoder.eval()

        timer.toc()

        # Logging the loss statistics into a dictionary
        loss_dict = dict()
        with torch.no_grad():
            loss_dict['loss/net'] = loss
            loss_dict['loss/rcnst/net'] = loss_rcnst
            for u_node, u_lossrcnst in u_lossrcnsts.items():
                loss_dict[f'loss/rcnst/{u_node}'] = u_lossrcnst
            loss_dict['loss/kl/net'] = loss_kl
            loss_dict['loss/kl/mu'] = loss_klmu
            loss_dict['loss/kl/sig'] = loss_klsig
            loss_dict['ltnt/sig/l2sq'] = sigmasq_mb.mean(dim=[-1, -2])
            musq_mean = musq_mb.mean(dim=[-1, -2])
            loss_dict['ltnt/mu/l2sq'] = musq_mean
            loss_dict['ltnt/sig/l2'] = sigmasq_mb.mean(dim=-1).sqrt().mean(dim=-1)
            loss_dict['ltnt/mu/l2'] = musq_mb.mean(dim=-1).sqrt().mean(dim=-1)
            if has_indzyloss:
                loss_dict['loss/indzy/net'] = loss_indzy

            loss_dict['ulss/rcnst/net'] = loss_rcnst
            for u_node, u_lossrcnst in u_lossrcnsts.items():
                loss_dict[f'ulss/rcnst/{u_node}'] = u_lossrcnst
            loss_dict['ulss/kl/mu'] = loss_klmu
            loss_dict['ulss/kl/sig'] = loss_klsig
            if has_indzyloss:
                loss_dict['ulss/indzy/net'] = loss_indzyunit
                
        for key, tnsr in loss_dict.items():
            assert tnsr.shape == (n_seeds,)

        #######################################
        #   Logging the training statistics   #
        #######################################
        lnet_ema, _ = ema('loss/net', loss)
        lkl_ema, _ = ema('loss/kl/net', loss_kl)
        lmusq_ema, _ = ema('ltnt/mu/l2sq', musq_mean)
        lrcnst_ema, _ = ema('loss/rcnst/net', loss_rcnst)
        lrcnsts_ema = {u_node: ema(f'loss/rcnst/{u_node}', u_lossrcnst)[0]
            for u_node, u_lossrcnst in u_lossrcnsts.items()}
        if has_indzyloss:
            lindzy_ema, _ = ema('loss/indzy/net', loss_indzy)

        if (epoch % 100 == 0) and (results_dir is not None):
            print_str = f'Epoch {epoch:06d}: EMA loss={lnet_ema:.4f}, kl={lkl_ema:.4f}, musq={lmusq_ema:.4f}'
            if has_indzyloss:
                print_str += f', indzy: {lindzy_ema:.4f}'
            for u_node, u_lrcnstema in lrcnsts_ema.items():
                print_str += f', rcnst/{u_node}: {u_lrcnstema:.4f}'
            if scheduler is not None:
                lr_median = np.median(scheduler.get_last_lr())
                print_str += f', median(lr)={lr_median:.4f}'
            print(print_str, flush=True)

        if do_logtb:
            with torch.no_grad():
                for loss_name, loss_tnsr in loss_dict.items():
                    tbwriter.add_scalar(loss_name, loss_tnsr.mean(), epoch)

        #######################################
        #      Performing the Evaluations     #
        #######################################
        e_vizinputs, e_mtrcinputs = dict(), dict()
        for eid, eopts in eparams.items():
            # The evaluation frequency
            e_frq = eopts['frq']
            if (epoch % e_frq) > 0:
                continue
            # The evaluation pre-processing
            e_pp = eopts['pp']
            # The evaluation pre-processor's input type (either 'x' or 'u')
            e_ppinpspc = eopts['inpspc']
            # The nodes and their dimensions after applying the `e_pp` pre-processor
            ue_dims = eopts['udims']
            # The evaluation split configs
            e_spltcfgs = eopts['splits']

            for vid, vcfg in vizcfgs.items():
                if vcfg['eval'] == eid:
                    assert epoch % io_avgfrq == 0, dedent(f'''
                        eval/{eid} requires storage, thus 
                        "eval/{eid}/frq" % "io/avg/frq" == 0 
                        should hold, but {e_frq} % {io_avgfrq} != 0.''')

            ######### Phase I: Applying the encoder to all splits ##########
            # Whether the original x inputs should go through the nn pre-prcessor's forward/inverse round trip.
            e_pprndtrp = eopts['rndtrp']
            # The randomization of evaluation indices ('none', 'shuffle', or 'iid')
            e_rndmztn = eopts['rndmztn']
            # The nodes and their dimensions before applying the `e_pp` pre-processor
            # This should practically be the same as `x_dimes` for VAEs.
            xe_dims = eopts['xdims']

            e_encpkgs = dict()
            for e_split, e_spltcfg in e_spltcfgs.items():
                # The split data indecis across all the data
                e_splitidxs = e_spltcfg['idxs']
                # The hyper-indices over `e_splitidxs` to keep track
                # of how much we made progress through them.
                eii = e_spltcfg['iidxs']
                # The number of samples in the split for evaluation
                n_evlpnts = e_spltcfg['n']
                # The vae label networks's noise injection sigma scale
                ey_sigscale = e_spltcfg['lbl']['sigscale']

                # Applying the Encoder
                timer.tic(f'sample/{eid}/{e_split}/encode')
                *pkg_encode, renew_eii, eii = Evaluation.encode(
                    eii, e_split, e_splitidxs, e_pp, e_frq, e_rndmztn,
                    ltnt_sigtnsfm, data_dict, nn_pp, encoder,
                    e_ppinpspc, e_pprndtrp, xe_dims, ue_dims,
                    is_condvae, lblnn_enctype, ey_sigscale, 
                    lbl_pp, lblnn, xl_dims, ul_dims, xnl_dims, y_dims,
                    n_seeds, n_evlpnts, ltnt_dim, eval_bsnn, epoch,
                    xn_dims, un_dims, erng, tch_device, tch_dtype)
                timer.toc()

                # Saving the same ordering for continuing in the next epochs
                if renew_eii:
                    e_spltcfg['iidxs'] = eii

                e_encpkgs[e_split] = pkg_encode

                # Making sure these variables would not leak elsewhere
                del n_evlpnts, eii, e_splitidxs

            for e_split, e_spltcfg in e_spltcfgs.items():
                # Getting the split encoded data
                (emu_mb, esigma_mb, u_eppmbs, ey_mbs, ei_mb) = e_encpkgs[e_split]
                # The variant configs
                e_vrntcfgs = e_spltcfg['vrnts']
                # The number of samples in the split for evaluation
                n_evlpnts = e_spltcfg['n']

                eis_vrntdatas = dict()
                for e_vrnt, e_vrntcfg in e_vrntcfgs.items():
                    ######### Phase II: Compiling the latent variables of all variants #########
                    timer.tic(f'sample/{eid}/{e_split}/{e_vrnt}/zsample')
                    ltnt_pkg = Evaluation.sample_zy(
                        emu_mb, esigma_mb, ey_mbs, ei_mb, y_dims,
                        n_seeds, n_evlpnts, ltnt_dim, is_condvae, erng, e_encpkgs,
                        eid, e_split, e_vrnt, e_vrntcfg, tch_device, tch_dtype)
                    emu_mbv, esigma_mbv, ez_mbv, ey_mbvs, ei_mbv, n_rcns, vrnt_type = ltnt_pkg
                    timer.toc()

                    ############## Phase III: Applying the decoder to all splits ###############
                    # Applying the decoder to the evaluation samples
                    timer.tic(f'sample/{eid}/{e_split}/{e_vrnt}/decode')
                    if vrnt_type == 'orig':
                        assert n_rcns == 1
                        uhat_eppmbvs = {u_node: u_tnsr.unsqueeze(2)
                            for u_node, u_tnsr in u_eppmbs.items()}
                        for u_node, (n_uchnls, n_ulen) in ue_dims.items():
                            eu_mbv = uhat_eppmbvs[u_node]
                            assert eu_mbv.shape == (n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen)
                        ey_mbvs2 = ey_mbvs
                    else:
                        assert ez_mbv is not None
                        uhat_eppmbvs, eyhat_mbvs = Evaluation.decode(
                            ez_mbv, ey_mbvs, nn_pp, decoder, e_pp, e_ppinpspc, lbl_pp, lblnn,
                            is_condvae, lblnn_enctype, n_seeds, n_evlpnts, n_rcns, ltnt_dim, eval_bsnn, 
                            xn_dims, un_dims, ue_dims, xl_dims, ul_dims, y_dims)
                        ey_mbvs2 = eyhat_mbvs
                    timer.toc()

                    eis_vrntdatas[e_vrnt] = (emu_mbv, esigma_mbv, ez_mbv, ey_mbvs2, uhat_eppmbvs, ei_mbv, n_rcns)

                    if vrnt_type in ('genr', 'gaus'):
                        assert e_split == 'normal'
                        eis_vrntdatas[f'{e_vrnt}.inp'] = (None, None, None, ey_mbvs, None, None, n_rcns)

                ######## Data Assembly for Metrics and Visualization #########
                eis_mtrcinputs, eis_vizinputs = Evaluation.assemble(
                    eid, e_split, eis_vrntdatas, n_seeds, n_evlpnts,
                    ltnt_dim, ue_dims, y_dims)
                e_mtrcinputs.update(eis_mtrcinputs)
                e_vizinputs.update(eis_vizinputs)

        #################### Metric Calculations ###################
        e_perfstats, e_perfhists = Evaluation.performance(perfcfgs, e_mtrcinputs, aero_cstinfo, 
            asgn_mdlscfg, metric_mdlscfg, eparams, split_vars, n_seeds, n_t, mrng, timer, 
            tch_device, tch_dtype, tch_cdtype)

        ############# Visualization: MDS Calculation ###############       
        e_vizoutputs = Visualization.mds(e_vizinputs, eparams, vizcfgs, vrng, n_seeds,
            ltnt_dim, y_dims, epoch, timer, tch_device)

        with torch.no_grad():
            if do_logtb:
                for key, val in e_perfstats.items():
                    tbwriter.add_scalar(f'perf/{key}', val.mean(), epoch)
                
            ############ Visualizations: Part 3 (Plotting) ##############
            Visualization.draw(e_vizoutputs, vizcfgs, epoch, n_seeds,
                chem_species, d_histbins, fig_dir, timer, tbwriter)

            if do_logtb:
                tbwriter.flush()

        timer.tic('save/pump')
        if epoch == 0:
            # Just making sure there is an entry for 'save/pump' before 
            # writing the report in the `dprepper.prep` call.
            timer.toctic('save/pump')

        # `stat_dict` will become hierarchical in the `.prep()` call
        stat_dict = {'perf': e_perfstats, **loss_dict}
        nn_dict = {'mdl/enc': encoder, 'mdl/dec': decoder}
        if is_condvae and (lblnn_enctype == 'vae'): 
            nn_dict['mdl/lbl'] = lblnn.encoder
        elif is_condvae and (lblnn_enctype == 'plain'): 
            nn_dict['mdl/lbl'] = lblnn
        elif is_condvae:
            raise ValueError('undefined case')
        strg_dict = deep2hie({'var/eval/raw': e_vizinputs, 
            'var/eval/mds': e_vizoutputs})

        # Filtering out the data that should not be stored
        DataPrepper.filter_vizstrg(strg_dict, eparams, vizcfgs)
        strg_dict['var/perf/hists'] = e_perfhists

        # Prepping the data tuples to be stored to the disk
        dtups = dprepper.prep(strg_dict, stat_dict, nn_dict, timer, epoch)

        dwriter.add(data_tups=dtups, file_path=hdfpth)
        timer.toc()

# -

# # Storing the Results

    if results_dir is not None:
        print(f'Training finished in {time.time() - trn_sttime:.1f} seconds.')

    # Closing up all the data/profile/log-writers
    timer.tic('save/close')
    dwriter.close()
    timer.toc()

    if do_logtb:
        timer.tic('save/tbflush')
        tbwriter.flush()
        timer.toc()

    if do_profile:
        timer.tic('save/profile')
        profiler.stop()
        html = profiler.output_html()
        htmlpath = f'{cfgstrg_dir}/profiler.html'
        with open(htmlpath, 'w') as fp:
            fp.write(html.encode('ascii', errors='ignore').decode('ascii'))
        timer.toc()

    if results_dir is not None:
        timer.print()

    # Preparing the memory usage
    outdict = dict()
    tchmemusage = profmem()
    assert str(tch_device) in tchmemusage
    if 'cuda' in device_name:
        tch_dvcmem = torch.cuda.get_device_properties(tch_device).total_memory
    else:
        tch_dvcmem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    outdict['dvc/mem/alloc'] = tchmemusage[str(tch_device)]
    outdict['dvc/mem/total'] = tch_dvcmem

# + tags=["active-py"] vscode={"languageId": "raw"}
    return outdict

if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('-c', '--configid', action='store', type=str, required=True)
        my_parser.add_argument('-d', '--device',   action='store', type=str, required=True)
        my_parser.add_argument('-s', '--nodesize', action='store', type=int, default=1)
        my_parser.add_argument('-r', '--noderank', action='store', type=int, default=0)
        my_parser.add_argument('-i', '--rsmindex', action='store', type=str, default="0.0")
        my_parser.add_argument('-n', '--nconfigs', action='store', type=int, default=-1)
        my_parser.add_argument('-g', '--nseedmax', action='store', type=int, default=0)
        my_parser.add_argument('--dry-run', action='store_true')
        args = my_parser.parse_args()
        args_configid = args.configid
        args_device_name = args.device
        args_nodesize = args.nodesize
        args_noderank = args.noderank
        arsg_rsmindex = args.rsmindex
        args_nconfigs = args.nconfigs
        args_dryrun = args.dry_run
        args_nseedmax = args.nseedmax
    else:
        args_configid = 'lvl1/lvl2/poiss2d'
        args_device_name = 'cuda:0'
        args_nodesize = 1
        args_noderank = 0
        arsg_rsmindex = 0
        args_nseedmax = 0
        args_dryrun = True

    assert args_noderank < args_nodesize
    cfgidsplit = args_configid.split('/')
    # Example: args_configid == 'lvl1/lvl2/poiss2d'
    config_id = cfgidsplit[-1]
    # Example: config_id == 'poiss2d'
    config_tree = '/'.join(cfgidsplit[:-1])
    # Example: config_tree == 'lvl1/lvl2'

    os.makedirs(configs_dir, exist_ok=True)
    # Example: configs_dir == '.../code_bspinn/config'
    os.makedirs(results_dir, exist_ok=True)
    # Example: results_dir == '.../code_bspinn/result'
    # os.makedirs(storage_dir, exist_ok=True)
    # Example: storage_dir == '.../code_bspinn/storage'
    
    if args_dryrun:
        print('>> Running in dry-run mode', flush=True)

    cfg_path_ = f'{configs_dir}/{config_tree}/{config_id}'
    cfg_exts = [cfg_ext for cfg_ext in ['json', 'yml', 'yaml'] if exists(f'{cfg_path_}.{cfg_ext}')]
    assert len(cfg_exts) < 2, f'found multiple {cfg_exts} extensions for {cfg_path_}'
    assert len(cfg_exts) > 0, f'found no json or yaml config at {cfg_path_}'
    cfg_ext = cfg_exts[0]
    cfg_path = f'{cfg_path_}.{cfg_ext}'
    print(f'>> Reading configuration from {cfg_path}', flush=True)
    
    if cfg_ext.lower() == 'json':
        with open(cfg_path, 'r') as fp:
            json_cfgdict = json.load(fp, object_pairs_hook=dict)
    elif cfg_ext.lower() in ('yml', 'yaml'):
        with open(cfg_path, 'r') as fp:
            json_cfgdict = dict(ruyaml.safe_load(fp))
    else:
        raise RuntimeError(f'unknown config extension: {cfg_ext}')
    
    if args_dryrun:
        import tempfile
        temp_resdir = tempfile.TemporaryDirectory(dir=f'{PROJPATH}/trash')
        temp_strdir = tempfile.TemporaryDirectory(dir=f'{PROJPATH}/trash')
        print(f'>> [dry-run] Temporary results dir placed at {temp_resdir.name}')
        print(f'>> [dry-run] Temporary storage dir placed at {temp_strdir.name}')
        results_dir = temp_resdir.name
        storage_dir = temp_strdir.name
        
        dr_maxfrq = 10
        dr_opts = {'opt/epoch': dr_maxfrq, 'io/cmprssn_lvl': 0, 'io/strg/logtb': False,
            'io/strg/profile': False, 'io/strg/savefigs': False, 'io/eval/enbl': False}
        for opt in dr_opts:
            assert opt in json_cfgdict
            json_cfgdict[opt] = dr_opts[opt]
        for opt in fnmatch.filter(json_cfgdict.keys(), '*/frq'):
            if json_cfgdict[opt] > dr_maxfrq:
                json_cfgdict[opt] = dr_opts[opt] = dr_maxfrq
        for opt in fnmatch.filter(json_cfgdict.keys(), '*/viz/*/store'):
            json_cfgdict[opt] = dr_opts[opt] = False

        print(f'>> [dry-run] The following options were made overriden:', flush=True)
        for opt, val in dr_opts.items():
            print(f'>>           {opt}: {val}', flush=True)
            
    nodepstfx = f'_{args_noderank:02d}'
    # Example: nodepstfx in ('_00', '_01', ...)
    json_cfgdict['io/config_id'] = f'{config_id}{nodepstfx}'
    # Example: ans in ('poiss2d', 'poiss2d_01')
    json_cfgdict['io/results_dir'] = f'{results_dir}/{config_tree}'
    # Example: ans == '.../code_bspinn/result/lvl1/lv2/poiss2d'
    json_cfgdict['io/storage_dir'] = f'{storage_dir}/{config_tree}'
    # Example: ans == '.../code_bspinn/storage/lvl1/lv2/poiss2d'
    json_cfgdict['io/tch/device'] = args_device_name
    # Example: args_device_name == 'cuda:0'

    # Pre-processing and applying the looping processes
    all_cfgdicts = preproc_cfgdict(json_cfgdict)
        
    # Selecting this node's config dict subset
    node_cfgdicts = [config_dict for cfgidx, config_dict in enumerate(all_cfgdicts) 
        if (cfgidx % args_nodesize == args_noderank)]
    n_nodecfgs = len(node_cfgdicts)
    
    # Going over the config dicts one-by-one
    rsmidx, rsmprt = tuple(int(x) for x in arsg_rsmindex.split('.'))
    for cfgidx, config_dict1 in enumerate(node_cfgdicts):
        # if a positive `args_nconfigs` is provided, we should only run 
        # `args_nconfigs` configs starting at `rsmidx`.
        is_toonew = ((args_nconfigs >= 0) and (cfgidx >= rsmidx + args_nconfigs))
        if (cfgidx < rsmidx) or is_toonew:
            continue
        
        # Parsing all the "/ref"-, "/def"-, and "/pop"-ending keys
        config_dict = parse_refs(config_dict1, trnsfrmtn='hie', pathsep=' -> ', 
            cfg_path=f'{config_tree}/{config_id}.{cfg_ext}')
        # Dropping the trailing references section
        ref_dict = get_subdict(config_dict, prefix='refs', pop=True)
        
        # Getting a single seed run to estimate the memory usage
        tempcfg = config_dict.copy()
        tempcfg['io/results_dir'] = None
        tempcfg['io/storage_dir'] = None
        tempcfg['opt/epoch'] = 0

        if args_nseedmax <= 0:
            # Taking a single-seed run
            args_device = torch.device(args_device_name)
            if 'cuda' in args_device_name:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_peak_memory_stats(device=args_device)
            tempcfg1 = dict(tempcfg)
            tempcfg1['rng_seed/list'] = [0]
            tod1 = main(tempcfg1)
            allocmem1, totmem = tod1['dvc/mem/alloc'], tod1['dvc/mem/total']

            # Taking a double-seed run
            if 'cuda' in args_device_name:
                torch.cuda.reset_peak_memory_stats(device=args_device)
            tempcfg2 = dict(tempcfg)
            tempcfg2['rng_seed/list'] = [0, 1000]
            tod2 = main(tempcfg2)
            allocmem2, totmem = tod2['dvc/mem/alloc'], tod2['dvc/mem/total']

            # Doing a linear interpolation to see how many seeds can fit
            allocmem2 = max(allocmem2, int(np.ceil(allocmem1 * 1.001)))
            nsd_max = 1 + int(0.75 * (totmem - allocmem1) / (allocmem2 - allocmem1))
        else:
            nsd_max = args_nseedmax
        
        # Computing how many parts we must split the original config into
        cfg_seeds = config_dict['rng_seed/list']
        nprts = int(np.ceil(len(cfg_seeds) / nsd_max))
        if args_nseedmax <= 0:
            mem_bias = 2 * allocmem1 - allocmem2
            mem_slope = allocmem2 - allocmem1
            print(f'>> Config index {cfgidx} takes {mem_bias/1e6:.1f} MB + {mem_slope/1e6:.1f} ' + 
                f'MB/seed (out of {totmem/1e9:.1f} GB)', flush=True)
            print(f'>> Config index {cfgidx} must be ' + 
                f'devided into {nprts} parts.', flush=True)
        
        # Looping over each part of the config
        for iprt in range(nprts):
            if (cfgidx == rsmidx) and (iprt < rsmprt):
                continue
            print(f'>>> Started Working on config index {cfgidx}.{iprt}' + 
                  f' (out of {nprts} parts and {n_nodecfgs} configs).', flush=True)
            iprtcfgseeds = cfg_seeds[(iprt*nsd_max):((iprt+1)*nsd_max)]
            iprtcfgdict = config_dict.copy()
            iprtcfgdict['rng_seed/list'] = iprtcfgseeds
            main(iprtcfgdict)
            print('-' * 40, flush=True)
        print(f'>> Finished working on config index {cfgidx} ' + 
              f'(out of {n_nodecfgs} configs).', flush=True)
        print('='*80, flush=True)
        
    if args_dryrun:
        print(f'>> [dry-run] Cleaning up {temp_resdir.name}')
        temp_resdir.cleanup()
        print(f'>> [dry-run] Cleaning up {temp_strdir.name}')
        temp_strdir.cleanup()


# + active=""
#
