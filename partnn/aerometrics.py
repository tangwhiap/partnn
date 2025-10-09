import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from textwrap import dedent
from collections import defaultdict

from partnn.tch_utils import odict
from partnn.tamie import get_coreshell
from partnn.io_utils import hie2deep, deep2hie, get_subdict

#########################################################
####### Averaging and Subseting Utility Functions #######
#########################################################

def get_chemsubidx(chem_inc, chem_exc, chem_species, n_chem):
    """
    Returns the index values for a subset of the chemical species. The chemical 
    subset can be specified using the `chem_inc` and `chem_exc` arguments.

    Parameters
    ----------
    chem_species: (list) The chemical species names. This will be used to detect 
        and remove the water (H2O) chemical.

    chem_inc: (list | str | None) The list of chemicals to include in the
        diameter calculation. Set `None` to disable this argument.

    chem_exc: (list | str | None) The list of chemicals to exclude in the
        diameter calculation. Set `None` to disable this argument.

    n_chem: (int) The total number of chemical species including water.

    Returns
    -------
    i_subchms: (list) The selected chemical species indecis.
        `assert len(i_subchms) == n_chemsub`

    n_chemsub: (int) The number of selected chemical species.
    """
    assert len(chem_species) == n_chem
    assert all(isinstance(chem_name, str) for chem_name in chem_species)

    ###########################################################
    ######## Including/Excluding the Chemical Species #########
    ###########################################################
    # The included chemical subset index
    if (chem_inc is not None) and (chem_exc is not None):
        raise ValueError(dedent(f'''
            Both `chem_inc` and `chem_exc` cannot be specified together:
                chem_inc = {chem_inc}
                chem_exc = {chem_exc}'''))
    elif (chem_inc is not None) and (chem_exc is None):
        chem_inc_lst = [chem_inc] if isinstance(chem_inc, str) else chem_inc
        assert isinstance(chem_inc_lst, list)
        assert all(isinstance(chem_name, str) for chem_name in chem_inc_lst)

        i_subchms = [i for i, chem in enumerate(chem_species) 
            if chem in chem_inc_lst]
        n_chemsub = len(i_subchms)
    elif (chem_inc is None) and (chem_exc is not None):
        chem_exc_lst = [chem_exc] if isinstance(chem_exc, str) else chem_exc
        assert isinstance(chem_exc_lst, list)
        assert all(isinstance(chem_name, str) for chem_name in chem_exc_lst)

        i_subchms = [i for i, chem in enumerate(chem_species) 
            if chem not in chem_exc_lst]
        n_chemsub = len(i_subchms)
    else:
        i_subchms = list(range(n_chem))
        n_chemsub = n_chem

    return i_subchms, n_chemsub

def get_vwakappa(v_chmprt, kappa_chm, n_chem, void_kappa, i_chmsub=None):
    """
    This function computes the "Volume Weighted Average" of some kappa coefficients. These 
    kappa coefficients are defined per chemical species, and could be related to anything.

    Parameters
    ----------
    v_chmprtsub: (torch.tensor) The speciated volume of the (subset) 
        chemicals within the particles.
        `assert v_chmprtsub.shape == (n_partset, n_part, n_wave, n_chemsub)`

    v_prtsub: (torch.tensor) The total volume of the chemical subset 
        within the particles. 
        This is essentially a summation of `v_chmprtsub` over the last 
        dimension, and is only passed to save on computational costs.
        `assert v_prtsub.shape == (n_partset, n_part, n_wave)`

    kappa_chmsub: (torch.tensor) The Kappa of all of the chemical species.
        `assert kappa_chmsub.shape == (1, 1, 1, n_chemsub)`

    n_chemsub: (int) The number of chemical species.

    void_kappa: (float) The kappa value for particles with zero volume.

    Returns
    -------
    kappa_prt: (torch.tensor) The volume-weighted average particle kappas.
        `assert kappa_prt.shape == (n_partset, n_part, n_wave)`
    """
    assert not kappa_chm.isnan().any()
    assert not v_chmprt.isnan().any()
    prt_shp = v_chmprt.shape[:-1]
    assert v_chmprt.ndim == kappa_chm.ndim
    vkap_shape = torch.broadcast_shapes(v_chmprt.shape, kappa_chm.shape)[:-1]

    ###########################################################
    ################ Subsetting the Chemicals #################
    ###########################################################
    if i_chmsub is None:
        n_chemsub, v_chmprtsub, kappa_chmsub = n_chem, v_chmprt, kappa_chm
    else:
        assert isinstance(i_chmsub, list)
        assert all(isinstance(ii, int) for ii in i_chmsub)
        assert all(ii < n_chem for ii in i_chmsub)
        assert all(ii >= 0 for ii in i_chmsub)
        n_chemsub = len(i_chmsub)
        
        # The speciated volumes of the chemical subset
        v_chmprtsub = v_chmprt[..., i_chmsub]
        assert v_chmprtsub.shape == (*prt_shp, n_chemsub)

        # The kappa values for the subset of chemical species
        kappa_chmsub = kappa_chm[..., i_chmsub]
        assert kappa_chmsub.shape[-1] == n_chemsub

    ###########################################################
    ############## Computing the Particle Kappa ###############
    ###########################################################
    assert kappa_chmsub.shape[-1] == n_chemsub
    assert (v_chmprtsub >= 0).all()

    v_prtsub = v_chmprtsub.sum(dim=-1, keepdims=True)
    assert v_prtsub.shape == (*prt_shp, 1)

    # The volume share of the chemical species within each particle
    v_nrmchmprtsub = v_chmprtsub / v_prtsub
    assert v_nrmchmprtsub.shape == (*prt_shp, n_chemsub)

    # The particle kappa values, assuming that they're not empty or pure-water
    kappa_prt1 = (v_nrmchmprtsub * kappa_chmsub).sum(axis=-1).to(dtype=kappa_chmsub.dtype)
    assert kappa_prt1.shape == vkap_shape

    # Making sure that zero volume particles get a void kappa assigned to them
    kappa_prt = torch.where(v_prtsub.squeeze(-1) > 0, kappa_prt1, void_kappa)
    assert kappa_prt.shape == vkap_shape

    # Other than the empty particles, the rest should have valid kappa values
    assert not kappa_prt.isnan().any()

    return kappa_prt

#########################################################
####### Histogram to Population Utility Functions #######
#########################################################

@torch.no_grad()
def clean_hist(m_chmprthst, n_prthst, n_seeds, n_mb, n_chem, n_bins):
    """    
    This function makes sure the histogram data is "cleaned". In other words, the 
    following properties must hold:

        1. No negative mass or counts.

        2. Zero mass should only be seen if and only if the count is zero.

    The cleaning process does the following steps in order:

        1. The counts are made non-negative by replacing negative counts with zero.

        2. The masses are replaced with zero when the counts are zero.

        3. Negative mass values are replaced with zero.

        4. When there is zero mass in a particle, the count is also replaced with zero.
    
    Parameters
    ----------
    m_chmprthst: (torch.tensor) The speciated mass of the chemicals within each diameter bin.
        `assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)`

    n_prthst: (torch.tensor) The total particle count within each diameter bin.
        `assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)`

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_bins: (int) The number of diameter bins.

    Returns
    -------

    m_chmprthst: (torch.tensor) The cleaned particle speciated masses.
        `assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)`
    
    n_prthst: (torch.tensor) The cleaned particle counts.
        `assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)`
    """
    
    n_prthst = n_prthst.detach().clone()
    assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)
    m_chmprthst = m_chmprthst.detach().clone()
    assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)

    # Stage 1: Making sure the particle counts are always non-negative
    n_prthst[n_prthst < 0] = 0

    # Stage 2: Zero mass when the count is zero.
    m_chmprthst = torch.where(n_prthst > 0, m_chmprthst, 0.0)

    # Stage 3: Making sure the mass data is always positive
    m_chmprthst[m_chmprthst < 0] = 0

    # Stage 4: Making sure there are zero particles when there is no mass in 
    # the particle!
    m_prthst = m_chmprthst.sum(dim=-2, keepdims=True)
    assert m_prthst.shape == (n_seeds, n_mb, 1, n_bins)

    n_prthst[m_prthst == 0.0] = 0.0

    return m_chmprthst, n_prthst

@torch.no_grad()
def clean_prtcls(m_prtchm, n_prt, n_seeds, n_mb, n_part, n_chem):
    """    
    This function makes sure the particle data is "cleaned". In other words, the 
    following properties must hold:

        1. No negative mass or counts.

        2. Zero mass should only be seen if and only if the count is zero.

    The cleaning process does the following steps in order:

        1. The counts are made non-negative by replacing negative counts with zero.

        2. The masses are replaced with zero when the counts are zero.

        3. Negative mass values are replaced with zero.

        4. When there is zero mass in a particle, the count is also replaced with zero.
    
    Parameters
    ----------
    m_prtchm: (torch.tensor) The speciated mass of the chemicals within each diameter bin.
        `assert m_prtchm.shape == (n_seeds, n_mb, n_part, n_chem)`

    n_prt: (torch.tensor) The total particle count within each diameter bin.
        `assert n_prt.shape == (n_seeds, n_mb, n_part, 1)`

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_part: (int) The number of particles in each population.

    n_chem: (int) The total number of chemical species including water.

    Returns
    -------

    m_prtchm: (torch.tensor) The cleaned particle speciated masses.
        `assert m_prtchm.shape == (n_seeds, n_mb, n_part, n_chem)`
    
    n_prt: (torch.tensor) The cleaned particle counts.
        `assert n_prt.shape == (n_seeds, n_mb, n_part, 1)`
    """
    
    n_prt = n_prt.detach().clone()
    assert n_prt.shape == (n_seeds, n_mb, n_part, 1)
    m_prtchm = m_prtchm.detach().clone()
    assert m_prtchm.shape == (n_seeds, n_mb, n_part, n_chem)

    # Stage 1: Making sure the particle counts are always non-negative
    n_prt[n_prt < 0] = 0

    # Stage 2: Zero mass when the count is zero.
    m_prtchm = torch.where(n_prt > 0, m_prtchm, 0.0)

    # Stage 3: Making sure the mass data is always positive
    m_prtchm[m_prtchm < 0] = 0

    # Stage 4: Making sure there are zero particles when there is no mass in 
    # the particle!
    m_prt = m_prtchm.sum(dim=-1, keepdims=True)
    assert m_prt.shape == (n_seeds, n_mb, n_part, 1)

    n_prt[m_prt == 0.0] = 0.0

    return m_prtchm, n_prt

@torch.no_grad()
def cnvrt_hist2prtcls(m_chmprthst, n_prthst, n_seeds, n_mb, n_chem, n_bins):
    """
    This function cleans the data and converts aerosol histograms into approximated 
    particles, where each bin is converted the a single particle with the same count 
    and total mass. 
    
    The cleaning process does the following in order:

        1. The counts are made non-negative by replacing negative counts with zero.

        2. The masses are replaced with zero when the counts are zero.

        3. Negative mass values are replaced with zero.

        4. When there is zero mass in a particle, the count is also replaced with zero.
    
    Parameters
    ----------
    m_chmprthst: (torch.tensor) The speciated mass of the chemicals within each diameter bin.
        `assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)`

    n_prthst: (torch.tensor) The total particle count within each diameter bin.
        `assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)`

    n_partset: (int) The number of particle sets.

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_bins: (int) The number of diameter bins.

    Returns
    -------

    m_chmprt: (torch.tensor) The approximated and cleaned particle speciated masses.
        `assert m_chmprt.shape == (n_partset, n_part, n_chem)`
    
    n_prt: (torch.tensor) The approximated and cleaned particle counts.
        `assert n_prt.shape == (n_partset, n_part)`
    
    n_partset: (int) The number of particle sets.
    
    n_part: (int) The number of particles within each particle set.
    """
    assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)
    assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)

    # Cleaning the data using the `clean_hist`
    m_chmprthst, n_prthst = clean_hist(m_chmprthst, n_prthst, n_seeds, n_mb, n_chem, n_bins)

    # We will have to convert the histograms with "approximate" particle sets
    n_partset, n_part = n_seeds * n_mb, n_bins

    # The particle counts
    n_prt = n_prthst.flatten(0, 2)
    assert n_prt.shape == (n_partset, n_part)

    # The particle speciated masses
    m_chmprtnet = m_chmprthst.flatten(0, 1).transpose(-1, -2)
    assert m_chmprtnet.shape == (n_seeds * n_mb, n_bins, n_chem)

    # We have to divide the total mass of the particles by the number of particles in the bin
    m_chmprt = torch.where(n_prt.unsqueeze(-1) > 0, 
        m_chmprtnet / n_prt.unsqueeze(-1), 0.0)
    assert m_chmprt.shape == (n_partset, n_part, n_chem)

    return m_chmprt, n_prt, n_partset, n_part

#########################################################
######### CCN Spectrum Error Utility Functions ##########
#########################################################

@torch.no_grad()
def calc_crh(m_chmprt, n_prt, tmprtr, rho_chm, kappa_chm, chem_species, 
    n_partset, n_part, n_chem, void_kappa=0.0, n_rfiters=100, e_rfiters=1e-12, 
    strict=False):
    """
    This function computes the critical relative humidity, critical diameter, 
    dry and wet diameter, and Kappa coefficient for each particle.

    Parameters
    ----------
    m_chmprt: (torch.tensor) The speciated mass of the chemicals within each particle.
        `assert m_chmprt.shape == (n_partset, n_part, n_chem)`

    n_prt: (torch.tensor) The particle counts array.
        `assert n_prt.shape == (n_partset, n_part)`

    tmprtr: (torch.tensor) The temperature of each particle set.
        `assert tmprtr.shape == (n_partset,)`

    rho_chm: (torch.tensor) The physical density of the chemical species.
        `assert rho_chm.shape == (n_chem,)`

    kappa_chm: (torch.tensor) The Kappa of the chemical species.
        `assert kappa_chm.shape == (n_chem,)`

    chem_species: (list) The chemical species names. This will be used to detect 
        and remove the water (H2O) chemical.

    n_partset: (int) The number of particle sets.

    n_part: (int) The number of distinct particles within each particle set.

    n_chem: (int) The total number of chemical species including water.

    void_kappa: (float) The kappa value for empty particles.

    n_rfiters: (int) The number of Newtonian root finding iterations to compute 
        the critical diameters.

    e_rfiters: (float) The relative update size stopping threshold for the newton 
        root finder.

    Returns
    -------
    crh_info: (dict) A dictionary with the following keys:

        d_prtdry: (torch.tensor) The dry diameter of the particles.
            `assert d_prtdry.shape == (n_partset, n_part)`

        d_prt: (torch.tensor) The (wet) diameter of the particles.
            `assert d_prt.shape == (n_partset, n_part)`

        kappa_prt: (torch.tensor) The Kappa of the particles.
            `assert kappa_prt.shape == (n_partset, n_part)`
        
        d_prtcrt: (torch.tensor) The critical diameter of the particles.
            `assert kappa_prt.shape == (n_partset, n_part)`

        crh_prt: (torch.tensor) The critical relative humidity of the particles.
            `assert crh_prt.shape == (n_partset, n_part)`
    """
    try:
        tch_device = m_chmprt.device
        tch_dtype = m_chmprt.dtype

        #############################
        #### Input Sanity Checks ####
        #############################
        assert not m_chmprt.isnan().any()
        assert (m_chmprt >= 0).all()
        assert not n_prt.isnan().any()
        assert (n_prt >= 0).all()
        is_nzero = (n_prt == 0)
        is_mzero = (m_chmprt.sum(-1) == 0)
        assert (is_nzero == is_mzero).all()

        assert m_chmprt.shape == (n_partset, n_part, n_chem)
        assert n_prt.shape == (n_partset, n_part)
        assert tmprtr.shape == (n_partset,)
        assert rho_chm.shape == (n_chem,)
        assert kappa_chm.shape == (n_chem,)

        ###########################################################
        ########## Excluding the Water Chemical Species ###########
        ###########################################################
        # The water chemical index
        i_drychms = [i for i, chem in enumerate(chem_species) if chem != 'H2O']
        n_chemdry = n_chem - 1

        m_chmprtdry = m_chmprt[:, :, i_drychms]
        assert m_chmprtdry.shape == (n_partset, n_part, n_chemdry)

        m_prtdry = m_chmprtdry.sum(dim=-1)
        assert m_prtdry.shape == (n_partset, n_part)

        kappa_chmdry = kappa_chm[i_drychms]
        assert kappa_chmdry.shape == (n_chemdry,)

        rho_chmdry = rho_chm[i_drychms]
        assert rho_chmdry.shape == (n_chemdry,)

        ###########################################################
        ########## Computing the Particle Dry Diameters ###########
        ###########################################################
        # The volume of each chemical species within the particles
        v_chmprtdry = m_chmprtdry / rho_chmdry.reshape(1, 1, n_chemdry)
        assert v_chmprtdry.shape == (n_partset, n_part, n_chemdry)

        # The dry volume of each particle
        v_prtdry = v_chmprtdry.sum(axis=-1)
        assert v_prtdry.shape == (n_partset, n_part)

        # The dry diameter of each particle
        d_prtdry = (6 * v_prtdry / np.pi) ** (1 / 3)
        assert d_prtdry.shape == (n_partset, n_part)

        ###########################################################
        ########## Computing the Particle Wet Diameters ###########
        ###########################################################
        # This section is fully optional, and the wet diameters are 
        # not needed for CRH calculations. 

        # The volume of each chemical species within the particles
        v_chmprt = m_chmprt / rho_chm.reshape(1, 1, n_chem)
        assert v_chmprt.shape == (n_partset, n_part, n_chem)

        # The wet volume of each particle
        v_prt = v_chmprt.sum(axis=-1)
        assert v_prt.shape == (n_partset, n_part)

        # The wet diameter of each particle
        d_prt = (6 * v_prt / np.pi) ** (1 / 3)
        assert d_prt.shape == (n_partset, n_part)

        ###########################################################
        ############## Computing the Particle Kappa ###############
        ###########################################################
        # The volume share of the chemical species within each particle
        v_nrmchmprtdry = v_chmprtdry / v_prtdry.unsqueeze(-1)
        assert v_nrmchmprtdry.shape == (n_partset, n_part, n_chemdry)

        # The particle kappa values, assuming that they're not empty or pure-water
        kappa_prt1 = (v_nrmchmprtdry * kappa_chmdry.reshape(1, 1, n_chemdry)).sum(axis=-1)
        assert kappa_prt1.shape == (n_partset, n_part)

        # Making sure that empty particles get a zero kappa assigned to them
        kappa_prt2 = torch.where(n_prt.squeeze(-1) > 0, kappa_prt1, void_kappa)
        assert kappa_prt2.shape == (n_partset, n_part)
        
        # Making sure that pure water particles get a zero kappa assigned to them
        kappa_prt = torch.where(m_prtdry > 0, kappa_prt2, void_kappa)
        assert kappa_prt.shape == (n_partset, n_part)
        
        # Other than the empty particles, the rest should have valid kappa values
        assert not kappa_prt.isnan().any()

        ###########################################################
        ############# Computing the Critical Diameter #############
        ###########################################################
        # Some physical constants for computing the `A` value
        water_surf_eng = 0.073e0
        water_molec_weight = 18e-3
        univ_gas_const = 8.314472e0
        water_density = 1e3

        # Computing the `A` parameter (for condensation calculations) based on the following:
        #   https://github.com/compdyn/partmc/blob/2341ef410d6f49f3169b8461b5fa8c89dbd3c7a2/tool/old_python_interface/partmc.py#L333
        A_num = (4.0 * water_surf_eng * water_molec_weight)
        A_denom = (univ_gas_const * water_density * tmprtr.unsqueeze(-1))
        A_partset = A_num / A_denom
        assert A_partset.shape == (n_partset, 1)

        # The polynomial coefficients for computing the critical diameter
        c4 = - 3.0 * (d_prtdry ** 3) * kappa_prt / A_partset
        assert c4.shape == (n_partset, n_part)
        c3 = - (d_prtdry ** 3) * (2.0 - kappa_prt)
        assert c3.shape == (n_partset, n_part)
        c0 = (d_prtdry ** 6) * (1.0 - kappa_prt)
        assert c0.shape == (n_partset, n_part)

        #############################
        # Newton's Root Finding Alg #
        #############################
        # Having `c0 == 0` can be a bit problematic for Newton; at around zero, `delta_y` can become 
        # undefined. For this reason the following sanity check must hold true.
        # assert torch.logical_or(c0 > 0, d_prtdry == 0).all()

        # Using a smart initialization from
        #   https://github.com/compdyn/partmc/blob/2341ef410d6f49f3169b8461b5fa8c89dbd3c7a2/src/aero_particle.F90#L850-L895
        # In particular, we will use the following initialization:
        #   `d = max(sqrt(-4d0 / 3d0 * c4), (-c3)**(1d0/3d0))`
        d_prtcrt1 = torch.pow((-4.0 / 3.0) * c4, 0.5)
        d_prtcrt2 = torch.pow(-c3, 1.0 / 3.0)
        d_prtcrt = torch.where(d_prtcrt1 > d_prtcrt2, d_prtcrt1, d_prtcrt2)
        assert d_prtcrt.shape == (n_partset, n_part)

        is_kappaprttiny = kappa_prt < 1e-30
        assert is_kappaprttiny.shape == (n_partset, n_part)

        # The newton iteration rule
        for i_rfiter in range(n_rfiters):
            y_prt = (d_prtcrt ** 6) + c4 * (d_prtcrt ** 4) + c3 * (d_prtcrt ** 3) + c0
            assert y_prt.shape == (n_partset, n_part)

            dy_prt = 6 * (d_prtcrt ** 5) + 4 * c4 * (d_prtcrt ** 3) + 3 * c3 * (d_prtcrt ** 2)
            assert dy_prt.shape == (n_partset, n_part)

            delta_y = y_prt / dy_prt
            assert delta_y.shape == (n_partset, n_part)
    
            # The following convergence rule was adopted from the PartMC's fortran code:
            #   https://github.com/compdyn/partmc/blob/2341ef410d6f49f3169b8461b5fa8c89dbd3c7a2/src/aero_particle.F90#L885
            deltay_rel = (delta_y / d_prtcrt).abs()
            assert deltay_rel.shape == (n_partset, n_part)
            
            # The following convergence rule was adopted from the PartMC's fortran code:
            #   https://github.com/compdyn/partmc/blob/2341ef410d6f49f3169b8461b5fa8c89dbd3c7a2/src/aero_particle.F90#L885
            has_converged1 = deltay_rel < e_rfiters
            assert has_converged1.shape == (n_partset, n_part)

            d_prtcrt = torch.where(has_converged1, d_prtcrt, d_prtcrt - delta_y)
            assert d_prtcrt.shape == (n_partset, n_part)

            # Let's see if there is any practical reason to keep iterating.
            has_converged2 = has_converged1.detach().clone()
            has_converged2[is_kappaprttiny] = True
            assert has_converged2.shape == (n_partset, n_part)
            
            if has_converged2.all():
                break

        # This exception is adopted from the following line
        #   https://github.com/compdyn/partmc/blob/2341ef410d6f49f3169b8461b5fa8c89dbd3c7a2/src/aero_particle.F90#L868
        d_prtcrt = torch.where(kappa_prt < 1e-30, d_prtdry, d_prtcrt)
        assert d_prtcrt.shape == (n_partset, n_part)

        #############################
        ## Solution Sanity Checks ###
        #############################
        # Only empty particles with zero dry diameter should have a NaN value for the critical diameter
        d_prtcrt = torch.where(d_prtdry > 0, d_prtcrt, 0.0)
        assert d_prtcrt.shape == (n_partset, n_part)
        assert not d_prtcrt.isnan().any()
        assert ((d_prtcrt + 1e-12) >= d_prtdry).all(), "critical diameter Newton loop converged to invalid solution"

        # Making sure that the Newton convergence has been met for all particles
        if strict:
            assert has_converged2.all(), "critical diameter Newton loop failed to converge"

        ###########################################################
        ########## Computing Critical Relative Humidity ###########
        ###########################################################
        # There is a fraction in the CRH computation
        crh_prtnum1 = (d_prtcrt ** 3) - (d_prtdry ** 3)
        assert crh_prtnum1.shape == (n_partset, n_part)
        crh_prtdenom = (d_prtcrt ** 3) - (d_prtdry ** 3) * (1 - kappa_prt)
        assert crh_prtdenom.shape == (n_partset, n_part)

        # We will follow the approximation made at PartMC:
        #   https://github.com/compdyn/partmc/blob/2341ef410d6f49f3169b8461b5fa8c89dbd3c7a2/tool/old_python_interface/partmc.py#L1003
        # Note that `kappa_prt < 1e-30` can be a reasonable detector for "empty" particles.
        crh_prtfrac = torch.where(kappa_prt < 1e-30, 1.0, crh_prtnum1 / crh_prtdenom)
        assert crh_prtfrac.shape == (n_partset, n_part)
        assert not crh_prtfrac.isnan().any()

        crh_prtnum2 = (A_partset / d_prtcrt).exp()
        assert crh_prtnum2.shape == (n_partset, n_part)

        crh_prt = crh_prtfrac * crh_prtnum2
        assert crh_prt.shape == (n_partset, n_part)

        crh_prt[crh_prtnum2.isinf()] = float('inf')
        assert not crh_prt.isnan().any()
        assert (crh_prt >= 1.0).all()

        crh_info = dict(d_prtdry=d_prtdry, d_prt=d_prt, kappa_prt=kappa_prt, 
            d_prtcrt=d_prtcrt, crh_prt=crh_prt)
    except Exception as exc:
        globals().update(locals())
        raise exc

    return crh_info

@torch.no_grad()
def calc_ccncdf(crh_prt, n_prt, n_partset, n_part, eps_histmin, eps_histmax, 
    n_epshist, tch_device, tch_dtype):
    """
    This function produces CCN fraction against epsilon. This also can be seen 
    as a CDF of the log-epsilon values.

    Specifically, this function is where most of the `torch.searchsorted` calls 
    are made, and the particle counts are used as weights to compute the empirical 
    critical relative humidity CDF functions.

    The input is a set of particles; both the particle counts and speciated masses 
    are necessary.
    
    Parameters
    ----------
    m_chmprt: (torch.tensor) The critical relative humidity of each particle.
        `assert m_chmprt.shape == (n_partset, n_part, n_chem)`

    n_prt: (torch.tensor) The particle counts array.
        `assert n_prt.shape == (n_partset, n_part)`

    tmprtr: (torch.tensor) The temperature of each particle set.
        `assert tmprtr.shape == (n_partset,)`

    n_partset: (int) The number of particle sets.

    n_part: (int) The number of distinct particles within each particle set.

    eps_histmin: (float) the minimum epsilon value for the CDF interval.

    eps_histmax: (float) the maximum epsilon value for the CDF interval.

    n_epshist: (int) the number of bins to divide the log-epsilon interval into.

    Returns
    -------
    ccncdf_info: (dict) a dictionary with the following keys:

        * eps_histbins: The epsilon values. 
            `assert eps_histbins.shape == (n_epshist,)`

        * ccn_cdf: The CCN CDF values.
            `assert ccn_cdf.shape == (n_partset, n_epshist)`
    """
    assert crh_prt.shape == (n_partset, n_part)
    assert n_prt.shape == (n_partset, n_part)

    # Computing the CRH epsilon
    eps_prt = crh_prt - 1
    assert eps_prt.shape == (n_partset, n_part)

    # Fiding the sorting index of epsilon within each particle set
    ias_eps = eps_prt.argsort(axis=-1)
    assert ias_eps.shape == (n_partset, n_part)

    # Sorting the particle epsilon values
    eps_prtsrtd = torch.take_along_dim(eps_prt, ias_eps, dim=-1)
    assert eps_prtsrtd.shape == (n_partset, n_part)

    # Re-ordering the particle counts
    n_prtsrtd = torch.take_along_dim(n_prt, ias_eps, dim=-1)
    assert n_prtsrtd.shape == (n_partset, n_part)

    # Getting the cumulative sum of the particle counts
    n_prtsrtdcs = n_prtsrtd.cumsum(dim=-1)
    assert n_prtsrtdcs.shape == (n_partset, n_part)

    # The log-epsilon interval
    eps_histbins = torch.linspace(math.log(eps_histmin), math.log(eps_histmax), 
        n_epshist, device=tch_device, dtype=tch_dtype).exp()
    assert eps_histbins.shape == (n_epshist,)

    # Expanding the epsilon bins to avoid complaints from `torch.searchsorted`
    eps_histbins2 = eps_histbins.reshape(1, n_epshist).expand(n_partset, n_epshist).contiguous()
    assert eps_histbins2.shape == (n_partset, n_epshist)

    # Finding the cut off index inside the sorted particles
    i_srchsrtdprt1 = torch.searchsorted(eps_prtsrtd, eps_histbins2, right=True)
    assert i_srchsrtdprt1.shape == (n_partset, n_epshist)

    # Minor adjustments to make sure we follow the CDF definition (i.e., $F(x)=Pr(X<x)$) strictly
    i_srchsrtdprt2 = torch.where(i_srchsrtdprt1 > 0, i_srchsrtdprt1 - 1, 0)
    assert i_srchsrtdprt2.shape == (n_partset, n_epshist)

    # The empirical CDF with respect to epsilon
    ccn_cdf = torch.take_along_dim(n_prtsrtdcs, i_srchsrtdprt2, dim=-1) / n_prtsrtdcs[:, -1:]
    assert ccn_cdf.shape == (n_partset, n_epshist)

    ccncdf_info = dict(eps_histbins=eps_histbins, ccn_cdf=ccn_cdf)
    return ccncdf_info

@torch.no_grad()
def get_apprxccncdf(m_input, n_input, eps_histmin, eps_histmax, n_epshist, 
    tmprtr_sclr, rho_chm, kappa_chm, chem_species, void_kappa, n_rfiters, 
    e_rfiters, n_seeds, n_mb, n_chem, n_part, tch_device, tch_dtype, 
    input_type='hst', strict=False):
    """
    This is a composite function, taking mass and count distribution histograms, and 
    doing a few things:

        1. Cleans the data and generates an approximated particle data 
            using the `cnvrt_hist2prtcls` function.

        2. Computes the critical relative humidity of the approximated particles 
            using the `calc_crh` function.

        3. Computes the CCN fractions for a range of log-epsilon values (i.e., 
            getting a log-eps CDF) using the `calc_ccncdf` function.

    The input parameters are the same ones used for these functions.
    """

    if input_type == 'hst':
        # Converting the histogram data into approximated particles
        m_chmprt, n_prt, n_partset, n_part = cnvrt_hist2prtcls(m_input, n_input, n_seeds, 
            n_mb, n_chem, n_part)
        assert m_chmprt.shape == (n_partset, n_part, n_chem)
        assert n_prt.shape == (n_partset, n_part)
    elif input_type == 'prt':
        m_chmprt_, n_prt_ = clean_prtcls(m_input, n_input, n_seeds, n_mb, n_part, n_chem)
        assert m_chmprt_.shape == (n_seeds, n_mb, n_part, n_chem)
        assert n_prt_.shape == (n_seeds, n_mb, n_part, 1)

        n_partset = n_seeds * n_mb
        m_chmprt = m_chmprt_.flatten(0, 1)
        assert m_chmprt.shape == (n_partset, n_part, n_chem)
        n_prt = n_prt_.flatten(0, 1).squeeze(-1)
        assert n_prt.shape == (n_partset, n_part)
    else:
        raise ValueError(f'undefined input/type={input_type}')

    # The particle set tempratures
    tmprtr = torch.full((n_partset,), tmprtr_sclr, device=tch_device, dtype=tch_dtype)
    assert tmprtr.shape == (n_partset,)

    # Computing the Critical Relative Humidity of the Particles
    crh_info = calc_crh(m_chmprt, n_prt, tmprtr, rho_chm, kappa_chm, chem_species, 
        n_partset, n_part, n_chem, void_kappa, n_rfiters, e_rfiters, strict=strict)

    # Retrieving the CRH for all particles
    crh_prt = crh_info['crh_prt']
    assert crh_prt.shape == (n_partset, n_part)

    # Computing the CCN fractions for a range of log-epsilon values (i.e., getting a log-eps CDF)
    ccncdf_info = calc_ccncdf(crh_prt, n_prt, n_partset, n_part, 
        eps_histmin=eps_histmin, eps_histmax=eps_histmax, n_epshist=n_epshist, 
        tch_device=tch_device, tch_dtype=tch_dtype)

    ccncdf_info['n_partset'] = n_partset
    ccncdf_info['n_part'] = n_part
    return ccncdf_info

@torch.no_grad()
def get_ccnerr(x_mbs, xhat_mbs, eps_histmin, eps_histmax, n_epshist, tmprtr_sclr, 
    void_kappa, n_rfiters, e_rfiters, rho_chm, kappa_chm, chem_species, n_seeds, n_mb, 
    n_chem, n_part, tch_device, tch_dtype, input_type='hst', strict=False):
    """
    This function computes the CCN fraction error integral between an original 
    and reconstructed pair of data points. This is a composite function:

        1. First, it computes the CCN spectrum of the original data on a user-specified
            log-epsilon interval using the  `get_apprxccncdf` function.
        
        2. Then, it does the same thing for the reconstructed data.

        3. Finally, it computes the integral of the absulte error between the two 
            CCN spectrum plots.
    
    Parameters
    ----------
    x_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the original data. 

    xhat_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the reconstructed data. 
    
    eps_histmin: (float) The lower-end of the epsilon spectrum. Do not take the 
        logarithm of epsilon before passing it here.

    eps_histmax: (float) The higher-end of the epsilon spectrum. Do not take the 
        logarithm of epsilon before passing it here.
    
    n_epshist: (int) The number of integration bins for the CCN spectrum. The 
        interval defined by `(eps_histmin, eps_histmax)` will be transformed 
        into the log-space, and the log-interval will be divided into `n_epshist` 
        equally-spaced points for defining the empirical CCN spectrum.

    tmprtr_sclr: (float) The temperature used to compute the critical relative 
        humidity values; The `A` factor in the computation relies on this temperature.

    void_kappa: (float) The kappa value for empty particles.

    n_rfiters: (int) The number of Newtonian root finding iterations to compute 
        the critical diameters.

    rho_chm: (torch.tensor) The physical density of the chemical species.
        `assert rho_chm.shape == (n_chem,)`

    kappa_chm: (torch.tensor) The Kappa of the chemical species.
        `assert kappa_chm.shape == (n_chem,)`

    chem_species: (list) The chemical species names. This will be used to detect 
        and remove the water (H2O) chemical.

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_part: (int) The number of particles/diameter bins.

    tch_device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    tch_dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    Returns
    -------
    ccn_errinfo: (dict) a dictionary mapping a few keys to tensors:

        * ccn_err: (torch.tensor) The computed integral of absolute error 
            between the original and reconstructed CCN spectrums.
            `assert ccn_err.shape == (n_seeds, n_mb)`

        * ccn_cdforig: (torch.tensor) The original CCN spectrums.
            `assert ccn_cdforig.shape == (n_seeds * n_mb, n_epshist)`

        * ccn_cdfrcnst: (torch.tensor) The reconstructed CCN spectrums.
            `assert ccn_cdfrcnst.shape == (n_seeds * n_mb, n_epshist)`

        * eps_histbins: The epsilon values.
            `assert eps_histbins.shape == (n_epshist,)`

        * n_epshist: (int) The number of epsilon bins.
    """
    m_key = {'hst': 'm_chmprthst', 'prt': 'm_prtchm'}[input_type]
    n_key = {'hst': 'n_prthst', 'prt': 'n_prt'}[input_type]

    ccncdf_infoorig = get_apprxccncdf(x_mbs[m_key], x_mbs[n_key], 
        eps_histmin, eps_histmax, n_epshist, tmprtr_sclr, rho_chm, kappa_chm, 
        chem_species, void_kappa, n_rfiters, e_rfiters, n_seeds, n_mb, 
        n_chem, n_part, tch_device, tch_dtype, input_type, strict=strict)

    eps_histbins = ccncdf_infoorig['eps_histbins']
    assert eps_histbins.shape == (n_epshist,)

    n_partset, n_part = ccncdf_infoorig['n_partset'], ccncdf_infoorig['n_part']
    assert n_partset == (n_seeds * n_mb)

    ccn_cdforig = ccncdf_infoorig['ccn_cdf']
    assert ccn_cdforig.shape == (n_partset, n_epshist)

    ccncdf_inforcnst = get_apprxccncdf(xhat_mbs[m_key], xhat_mbs[n_key], 
        eps_histmin, eps_histmax, n_epshist, tmprtr_sclr, rho_chm, kappa_chm, 
        chem_species, void_kappa, n_rfiters, e_rfiters, n_seeds, n_mb, 
        n_chem, n_part, tch_device, tch_dtype, input_type, strict=strict)

    ccn_cdfrcnst = ccncdf_inforcnst['ccn_cdf']
    assert ccn_cdfrcnst.shape == (n_partset, n_epshist)

    logeps_diff = math.log(eps_histmax) - math.log(eps_histmin)
    ccn_err1 = (ccn_cdforig - ccn_cdfrcnst).abs().mean(dim=-1) * logeps_diff
    assert ccn_err1.shape == (n_partset,)

    ccn_err = ccn_err1.reshape(n_seeds, n_mb)
    assert ccn_err.shape == (n_seeds, n_mb)

    ccn_errinfo = dict(ccn_err=ccn_err, ccn_cdforig=ccn_cdforig, 
        ccn_cdfrcnst=ccn_cdfrcnst, eps_histbins=eps_histbins, 
        n_epshist=n_epshist)

    return ccn_errinfo

#########################################################
############ Optical Error Utility Functions ############
#########################################################

@torch.no_grad()
def interp_refr(refr_wvchmref, len_wvref, wave_length, chem_species=None):
    """
    This function takes 
        1. a 2D array of refraction indices (n_wave, n_chem),  
        2. a vector of source wave-lengths, and 
        3. a list of target wave-lengths, for which you want the 
            interpolated refraction indices. 

    The output is the 2D array of interpolated refraction indices for 
    the target wave-lengths.
    
    Parameters
    ----------
        refr_wvchmref: (np.ndarray | torch.tensor) The source/reference 
            complex refraction indices.

            `assert refr_wvchmref.shape == (n_waveref, n_chem)`
            `assert refr_wvchmref.dtype == 'complex128'`

        len_wvref: (np.ndarray | torch.tensor) The source/reference
            wave-lengths.

            `assert len_wvref.dtype == 'float64'`
            `assert len_wvref.shape == (n_waveref,)`
        
        wave_length: (float | list[float]) Target wave-lengths.

            `assert len(wave_length) == n_wave`
            `assert all(isinstance(wvlen, float) for wvlen in wave_length)`
    
    Returns
    -------
        refr_wvchm: (np.ndarray | torch.tensor) The target 
            interpolated complex refraction indices.

            `assert refr_wvchm.shape == (n_wave, n_chem)`
            `assert refr_wvchm.dtype == 'complex128'`
        
        len_wv: (np.ndarray | torch.tensor) The target 
            wave lengths array/tensor.

            `assert len_wv.shape == (n_wave,)`
            `assert len_wv.dtype == 'float64'`
    """
    if torch.is_tensor(refr_wvchmref):
        refr_wvchmrefnp = refr_wvchmref.detach().cpu().numpy()
    else:
        refr_wvchmrefnp = refr_wvchmref

    assert isinstance(refr_wvchmrefnp, np.ndarray)
    assert refr_wvchmrefnp.dtype == 'complex128'
    n_waveref, n_chem = refr_wvchmrefnp.shape
    assert n_waveref >= 2, 'We need at least two wave-lengths for interpolation!'

    n_refrwell = np.logical_not(np.isnan(refr_wvchmrefnp)).sum(axis=0)
    assert (n_refrwell > 1).all(), dedent(f'''
        Some source chemicals did not define at least two well-defined 
        refraction indices:
            The number of non-zero elements for each chemical:
                {n_refrwell}''')

    if torch.is_tensor(len_wvref):
        len_wvrefnp = len_wvref.detach().cpu().numpy()
    else:
        len_wvrefnp = len_wvref
    
    assert isinstance(len_wvrefnp, np.ndarray)
    assert len_wvrefnp.dtype == 'float64'
    assert len_wvrefnp.shape == (n_waveref,)

    u, c = np.unique(len_wvrefnp, return_counts=True)
    assert (c <= 1).all(), f'Duplicated source wave-lengths: {u[c > 1]}'

    wave_length = [wave_length] if isinstance(wave_length, float) else wave_length
    assert isinstance(wave_length, list)
    assert all(isinstance(wvlen, float) for wvlen in wave_length)
    n_wave = len(wave_length)

    trgmin, srcmin = min(wave_length), len_wvrefnp.min()
    assert (trgmin - srcmin) >= -(1e-5 * abs(srcmin)), dedent(f'''
        Target needs extrapolation. We only do interpolation:
            target min: {min(wave_length)}
            source min: {len_wvrefnp.min()}''')
    
    trgmax, srcmax = max(wave_length), len_wvrefnp.max()
    assert (trgmax - srcmax) <= (1e-5 * abs(srcmax)), dedent(f'''
        Target needs extrapolation. We only do interpolation:
            target max: {max(wave_length)}
            source max: {len_wvrefnp.max()}''')
    
    # Making a source data-frame of refraction indices
    df_refr1 = pd.DataFrame(refr_wvchmrefnp, columns=chem_species, index=len_wvrefnp)
    assert df_refr1.shape == (n_waveref, n_chem)
    
    df_refr2 = df_refr1.copy(deep=True)
    # Populating the target values with NaNs.
    for wvlen in wave_length:
        # if wvlen not in df_refr2.index:
        df_refr2.loc[wvlen] = None
    # Making sure the original values are not over-written with NaNs.
    for wvlen in len_wvrefnp:
        df_refr2.loc[wvlen] = df_refr1.loc[wvlen]

    # Performing the linear interpolation
    # Note: Make sure to use the `index` method. Otherwise, the source 
    #   wave-length values will be ignored during interpolation, and 
    #   the interpolated results will be meaningless.
    
    df_refr3 = df_refr2.interpolate(method='index', axis=0).loc[wave_length]
    assert df_refr3.shape == (n_wave, n_chem)
    
    df_refr3nan = df_refr3[df_refr3.isna().any(axis=1)]
    assert df_refr3nan.shape[0] == 0, dedent(f'''
        The following had problems with interpolation: \n{df_refr3nan}''')

    refr_wvchmnp = df_refr3.values
    assert isinstance(refr_wvchmnp, np.ndarray)
    assert refr_wvchmnp.shape == (n_wave, n_chem)
    assert refr_wvchmnp.dtype == 'complex128'

    if torch.is_tensor(refr_wvchmref):
        refr_wvchm = torch.from_numpy(refr_wvchmnp).to( 
            device=refr_wvchmref.device, dtype=refr_wvchmref.dtype)
    else:
        refr_wvchm = refr_wvchmnp

    if torch.is_tensor(len_wvref):
        len_wv = torch.tensor(wave_length, 
            device=len_wvref.device, dtype=len_wvref.dtype)
    else:
        len_wv = np.array(wave_length, dtype=len_wvref.dtype)
    
    return refr_wvchm, len_wv, n_wave

@torch.no_grad()
def get_bulkoptical(m_input, n_input, rho_chm, refr_wvchm, len_wv, 
    n_max, void_refr, chem_species, n_seeds, n_mb, n_chem, n_part, n_wave,
    tch_device, tch_dtype, tch_cdtype, input_type='hst'):
    r"""
    Computes the aerosol "population" optical properties. 
    
        1. The optical properties are the extinction, scattering, and absorption 
            efficiencies.

        2. The TAMie algorithm computes optical properties per-particle.

        3. To compute a "population" optical property, we compute the particle 
            scattering/absorption cross-section and sum them up according to the 
            input particle number concentrations.

    Here is a more detailed description:

        1. From the per-particle scattering/absorption efficiencies, we need to 
           first calculate the scattering/absorption cross sections.

        2. "Scattering cross section" is equal to the "particle scattering efficiency" 
           times the "particle cross section area".

            1. Similarly, you can define the "absorption cross section".

            2. The "particle cross section area" is `$\frac{\pi}{4} d_i^2$`, where 
                
                1. `d_i` is the diameter of particle `i`,

                2. `qs_i` is the "particle scattering efficiency".

                3. `qa_i` is the "particle scattering efficiency".

                4. Therefore, `$\frac{\pi}{4} d_i^2 qs_i$` is the "scattering cross 
                   section".

        3. Once we have the "scattering cross sections", we need to add them up over 
           all particles.

            1. This then gives us the "scattering coefficient" for the population. 

            2. In other words, 
               `$ \beta_{scat} = \sum_{i=1}^{N_p} \frac{\pi}{4} d_i^2 * qs_i / V $`.

            3. `V` is the volume (of air, in `m^3`) where the particles live in.
            
            4. Similarly, you can define the "absorption coefficient" for the population.

        4. The cross section area is in `m^2`, so `\beta_{scat}` is a quantity in `m^{-1}`.

        5. In PartMC, each particle has a number concentration weight associated with it.

            1. `n_prt` corresponds to the aerosel number concentration 
               (i.e., `aero_num_conc` in PartMC). 

            2. The number concentration is basically the count of the particle divided 
               by the volume of the aerosol population's grid/box. 

            3. Therefore, `n_prt` has a physical unit of `1/m^3`. 

            4. Therefore, `\beta` is summing up the number concentration weight times 
               the cross section.

            5. `aero_num_conc` is the associated number concentration weight of each 
               particle (with a `m^{-3}` physical unit).
            
                1. So, this `m^{-3}` times the cross section `m^2` gets you `m^{-1}`.

            6. This means that 
               `$ \beta_{scat} = \sum_{i=1}^{N_p} \frac{\pi}{4} d_i^2 * qs_i * n_i $`.

    Parameters
    ----------
    m_chmprthst: (torch.tensor) The speciated mass of the chemicals within each diameter bin.
        `assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)`

    n_prthst: (torch.tensor) The total particle count within each diameter bin.
        `assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)`
        
    rho_chm: (torch.tensor) The physical density of the chemical species.
        `assert rho_chm.shape == (n_chem,)`

    refr_wvchm: (torch.tensor) The complex refraction indices of the chemical species 
        at the input wave length.
        `assert refr_wvchm.shape == (n_chem, n_wave)`

    len_wv: (torch.tensor) The optical wave length for measuring the optical 
        efficiency properties.
        `assert len_wv.shape == (n_wave,)`

    n_max: (int) The number of iterations needed for the mie sequence 
        summations to converge.

    void_refr: (complex) The optical refraction coefficient for particles
        with zero volume.

    chem_species: (list) The chemical species names. This will be used to detect 
        and remove the water (H2O) chemical.

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_wave: (int) The number of wave-lengths.

    n_part: (int) The number of particles/diameter bins.

    tch_device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    tch_dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    tch_cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    Returns
    -------
    optical_info: (dict) the optical properties with a few keys:

        * qs_pop: (torch.tensor) the scattering Q efficiency of the particles.
            `assert qs_pop.shape == (n_seeds, n_mb)`

        * qa_pop: (torch.tensor) the absorption Q efficiency of the particles.
            `assert qa_pop.shape == (n_seeds, n_mb)`
    """

    if input_type == 'hst':
        # Converting the histogram data into approximated particles
        m_chmprt, n_prt, n_partset, n_part = cnvrt_hist2prtcls(m_input, n_input, n_seeds, 
            n_mb, n_chem, n_part)
        assert m_chmprt.shape == (n_partset, n_part, n_chem)
        assert n_prt.shape == (n_partset, n_part)
    elif input_type == 'prt':
        m_chmprt_, n_prt_ = clean_prtcls(m_input, n_input, n_seeds, n_mb, n_part, n_chem)
        assert m_chmprt_.shape == (n_seeds, n_mb, n_part, n_chem)
        assert n_prt_.shape == (n_seeds, n_mb, n_part, 1)

        n_partset = n_seeds * n_mb
        m_chmprt = m_chmprt_.flatten(0, 1)
        assert m_chmprt.shape == (n_partset, n_part, n_chem)
        n_prt = n_prt_.flatten(0, 1).squeeze(dim=-1)
        assert n_prt.shape == (n_partset, n_part)
    else:
        raise ValueError(f'undefined input/type={input_type}')

    ###########################################################
    ##### Full Particle Diameter, Volume, and Refractions #####
    ###########################################################
    # The volume of each chemical species within the particles
    v_chmprt = m_chmprt / rho_chm.reshape(1, 1, n_chem)
    assert v_chmprt.shape == (n_partset, n_part, n_chem)

    # The subset volume of each particle
    v_prt = v_chmprt.sum(axis=-1)
    assert v_prt.shape == (n_partset, n_part)

    # The subset diameter of each particle
    d_prt = (6 * v_prt / math.pi).pow(1.0 / 3.0)
    assert d_prt.shape == (n_partset, n_part)

    ###########################################################
    ##### Core Particle Diameter, Volume, and Refractions #####
    ###########################################################
    chem_inccore, chem_exccore = ['BC', 'OIN'], None

    # The chemical indices within the core
    i_chmcore, n_chemcore = get_chemsubidx(chem_inccore, chem_exccore, 
        chem_species, n_chem)
    assert n_chemcore == len(chem_inccore)

    # Computing the core refraction
    # Note: This variable corresponds to `xc` in the `TAMie.py` code.
    v_wvchmprt = v_chmprt.reshape(n_partset, 1, n_part, n_chem
        ).expand(n_partset, n_wave, n_part, n_chem)
    assert v_wvchmprt.shape == (n_partset, n_wave, n_part, n_chem)
    
    refr_prtcore = get_vwakappa(v_wvchmprt, refr_wvchm[None, :, None, :], 
        n_chem, void_refr, i_chmcore)
    assert refr_prtcore.shape == (n_partset, n_wave, n_part)

    # The core volume of each particle
    v_prtcore = v_chmprt[..., i_chmcore].sum(axis=-1)
    assert v_prtcore.shape == (n_partset, n_part)

    # The core diameter of each particle
    d_prtcore = (6 * v_prtcore / math.pi).pow(1.0 / 3.0)
    assert d_prtcore.shape == (n_partset, n_part)

    ###########################################################
    #### Shell Particle Diameter, Volume, and Refractions #####
    ###########################################################
    # Getting the particle diameter and volume information
    chem_incshl, chem_excshl = None, chem_inccore

    # The chemical indices within the shell
    i_chmshl, n_chemshl = get_chemsubidx(chem_incshl, chem_excshl, chem_species, n_chem)
    assert n_chemshl == n_chem - len(chem_excshl)

    # Computing the shell refraction. 
    # Note: This variable corresponds to `ms` in the `TAMie.py` code.
    refr_prtshll = get_vwakappa(v_wvchmprt, refr_wvchm[None, :, None, :], 
        n_chem, void_refr, i_chmshl)
    assert refr_prtshll.shape == (n_partset, n_wave, n_part)

    # The shell volume of each particle
    v_prtshll = v_chmprt[..., i_chmshl].sum(axis=-1)
    assert v_prtshll.shape == (n_partset, n_part)

    # The shell diameter of each particle
    d_prtshll = (6 * v_prtshll / math.pi).pow(1.0 / 3.0)
    assert d_prtshll.shape == (n_partset, n_part)

    # The optical size parameter for the entire particle, a.k.a. the shell: 
    #   $ x = 2 \pi radius_particle / \lambda$.
    # Note: This variable corresponds to `xs` in the `TAMie.py` code.
    x_prtshll = d_prt[:, None, :] * (math.pi / len_wv[None, :, None])
    assert x_prtshll.shape == (n_partset, n_wave, n_part)

    # The optical size parameter for the particle core: 
    #   $ x = 2 \pi radius_particle / \lambda$.
    # Note: This variable corresponds to `xc` in the `TAMie.py` code.
    x_prtcore = d_prtcore[:, None, :] * (math.pi / len_wv[None, :, None])
    assert x_prtcore.shape == (n_partset, n_wave, n_part)

    ###########################################################
    #### Finding the Per-Particle Q Scattering Efficiency #####
    ###########################################################
    assert n_partset == n_seeds * n_mb
    assert refr_prtcore.shape == (n_partset, n_wave, n_part) # mc
    assert refr_prtshll.shape == (n_partset, n_wave, n_part) # ms
    assert x_prtcore.shape    == (n_partset, n_wave, n_part) # xc
    assert x_prtshll.shape    == (n_partset, n_wave, n_part) # xs
    
    qe_prt_, qs_prt_, g_prt_ = get_coreshell(refr_prtcore, refr_prtshll, x_prtcore, 
        x_prtshll, n_max, tch_device, tch_dtype, tch_cdtype, void_qe=0.0, void_qs=0.0, 
        void_g=0.0)

    # The per-particle extinction Q efficiency
    assert qe_prt_.shape == (n_partset, n_wave, n_part)
    # The per-particle scattering Q efficiency
    assert qs_prt_.shape == (n_partset, n_wave, n_part)
    # The per-particle absorption Q efficiency
    qa_prt_ = qe_prt_ - qs_prt_
    assert qa_prt_.shape == (n_partset, n_wave, n_part)
    # The per-particle g asymmetry parameter
    assert g_prt_.shape == (n_partset, n_wave, n_part)
    
    ###########################################################
    ##### Finding the Population Q Scattering Efficiency ######
    ###########################################################
    # The particle cross sections
    crossec_prt = d_prtshll.square() * (math.pi / 4)
    assert crossec_prt.shape == (n_partset, n_part)

    # The population level Q scattering efficiency
    # Note that 
    #   1. `n_prt` corresponds to the aerosel number concentration 
    #       (i.e., `aero_num_conc` in PartMC). The number concentration 
    #       is basically the count of the particle divided by the volume 
    #       of the aerosol population's grid/box. Therefore, `n_prt` has 
    #       a physical unit of `1/m^3`.
    #   2. `crossec_prt * qs_prt_` gives the scattering cross-section 
    #       of each particle. This has a physical unit of `m^2`.
    #   3. To get the population scattering/absorption q efficiency, you 
    #      have to sum up the particle scattering/absorption cross-sections 
    #      and divide by the box/grid volume where the particles reside. 
    #      Therefore, it will have a physical unit of `1/m`.

    # The population scattering coefficient.
    qscs_prt_ = (crossec_prt[:, None, :] * qs_prt_ * n_prt[:, None, :])
    assert qscs_prt_.shape == (n_partset, n_wave, n_part)
    qs_pop_ = qscs_prt_.sum(dim=-1)
    # qs_pop_ = torch.einsum('ij,ikj,ij->ik', crossec_prt, qs_prt_, n_prt)
    assert qs_pop_.shape == (n_partset, n_wave)
    
    # The population absorption coefficient.
    qacs_prt_ = (crossec_prt[:, None, :] * qa_prt_ * n_prt[:, None, :])
    assert qacs_prt_.shape == (n_partset, n_wave, n_part)
    qa_pop_ = qacs_prt_.sum(dim=-1)
    # qa_pop_ = torch.einsum('ij,ikj,ij->ik', crossec_prt, qa_prt_, n_prt)
    assert qa_pop_.shape == (n_partset, n_wave)

    # Unflattenning the first two dimensions
    qs_prt = qs_prt_.reshape(n_seeds, n_mb, n_wave, n_part)
    assert qs_prt.shape == (n_seeds, n_mb, n_wave, n_part)
    qa_prt = qa_prt_.reshape(n_seeds, n_mb, n_wave, n_part)
    assert qa_prt.shape == (n_seeds, n_mb, n_wave, n_part)
    qscs_prt = qscs_prt_.reshape(n_seeds, n_mb, n_wave, n_part)
    assert qscs_prt.shape == (n_seeds, n_mb, n_wave, n_part)
    qacs_prt = qacs_prt_.reshape(n_seeds, n_mb, n_wave, n_part)
    assert qacs_prt.shape == (n_seeds, n_mb, n_wave, n_part)
    qs_pop = qs_pop_.reshape(n_seeds, n_mb, n_wave)
    assert qs_pop.shape == (n_seeds, n_mb, n_wave)
    qa_pop = qa_pop_.reshape(n_seeds, n_mb, n_wave)
    assert qa_pop.shape == (n_seeds, n_mb, n_wave)

    optical_info = dict(qs_pop=qs_pop, qa_pop=qa_pop, qs_prt=qs_prt, qa_prt=qa_prt, 
        qscs_prt=qscs_prt, qacs_prt=qacs_prt, n_partset=n_partset, n_part=n_part)

    return optical_info

@torch.no_grad()
def get_opticalerr(x_mbs, xhat_mbs, rho_chm, refr_wvchm, len_wv, 
    n_max, void_refr, chem_species, n_seeds, n_mb, n_chem, n_part, 
    n_wave, tch_device, tch_dtype, tch_cdtype, input_type='hst'):
    """
    This function computes the "optical error" between an original 
    and reconstructed pair of data points. This is a composite function:

        1. First, it computes the bulk optical properties of the original data using 
            the `get_bulkoptical` function. The optical properties of interest are 
            the population Q scattering and absorption efficiency values.
        
        2. Then, it does the same thing for the reconstructed data.

        3. Next, we compute the relative error between the population Q scattering 
            efficiency values.

        4. Similarly, we compute the relative error between the population Q absorption 
            efficiency values.

        5. We report the average of these two relative errors as the "optical 
            reconstruction error".

    Parameters
    ----------
    x_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the original data. 

    xhat_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the reconstructed data. 

    rho_chm: (torch.tensor) The physical density of the chemical species.
        `assert rho_chm.shape == (n_chem,)`

    refr_wvchm: (torch.tensor) The complex refraction indices of the chemical species 
        at the input wave length.
        `assert refr_wvchm.shape == (n_chem, n_wave)`

    len_wv: (torch.tensor) The optical wave length for measuring the optical 
        efficiency properties.
        `assert len_wv.shape == (n_wave,)`

    n_max: (int) The number of iterations needed for the mie sequence 
        summations to converge.

    void_refr: (complex) The optical refraction coefficient for particles
        with zero volume.

    chem_species: (list) The chemical species names. This will be used to detect 
        and remove the water (H2O) chemical.

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_part: (int) The number of particles/diameter bins.

    n_wave: (int) The number of wave-lengths.

    tch_device: (torch.device) The torch device
        (e.g., `torch.device('cuda:0')`).

    tch_dtype: (torch.dtype) The real floating point torch data type
        (e.g., `torch.float64`).

    tch_cdtype: (torch.dtype) The complex floating point torch data type
        (e.g., `torch.complex128`).

    Returns
    -------
    opt_errinfo: (dict) a dictionary mapping a few keys to tensors:

        * opt_err: (torch.tensor) The computed "optical" error 
            between the original and reconstructed Q efficiencies.
            `assert opt_err.shape == (n_seeds, n_mb)`

        * qs_relerr: (torch.tensor) The relative scattering efficiency 
            error between the original and reconstructed particles.
            `assert qs_relerr.shape == (n_seeds, n_mb)`

        * qa_relerr: (torch.tensor) The relative absorption efficiency 
            error between the original and reconstructed particles.
            `assert qa_relerr.shape == (n_seeds, n_mb)`
    """

    m_key = {'hst': 'm_chmprthst', 'prt': 'm_prtchm'}[input_type]
    n_key = {'hst': 'n_prthst', 'prt': 'n_prt'}[input_type]
    
    ###########################################################
    ###### Optical Properties of the Original Particles #######
    ###########################################################
    qsqa_infoorig = get_bulkoptical(x_mbs[m_key], x_mbs[n_key], 
        rho_chm, refr_wvchm, len_wv, n_max, void_refr, chem_species, n_seeds, 
        n_mb, n_chem, n_part, n_wave, tch_device, tch_dtype, 
        tch_cdtype, input_type)

    # The scattering efficiency of the original particles
    qs_poporig = qsqa_infoorig['qs_pop']
    assert qs_poporig.shape == (n_seeds, n_mb, n_wave)

    # The absorption efficiency of the original particles
    qa_poporig = qsqa_infoorig['qa_pop']
    assert qa_poporig.shape == (n_seeds, n_mb, n_wave)

    ###########################################################
    #### Optical Properties of the Reconstructed Particles ####
    ###########################################################
    qsqa_inforcnst = get_bulkoptical(xhat_mbs[m_key], xhat_mbs[n_key], 
        rho_chm, refr_wvchm, len_wv, n_max, void_refr, chem_species, n_seeds, 
        n_mb, n_chem, n_part, n_wave, tch_device, tch_dtype, 
        tch_cdtype, input_type)

    # The scattering efficiency of the reconstructed particles
    qs_poprcnst = qsqa_inforcnst['qs_pop']
    assert qs_poprcnst.shape == (n_seeds, n_mb, n_wave)

    # The absorption efficiency of the reconstructed particles
    qa_poprcnst = qsqa_inforcnst['qa_pop']
    assert qa_poprcnst.shape == (n_seeds, n_mb, n_wave)

    ###########################################################
    ######## The Scattering Efficiency Relative Error #########
    ###########################################################
    qs_relerr = get_relerr(qs_poporig, qs_poprcnst, pnorm=2, rdcdims=-1)
    assert qs_relerr.shape == (n_seeds, n_mb)
    assert not qs_relerr.isnan().any()

    logqs_poporig = qs_poporig.clip(min=1e-7).log()
    assert logqs_poporig.shape == (n_seeds, n_mb, n_wave)

    logqs_poprcnst = qs_poprcnst.clip(min=1e-7).log()
    assert logqs_poprcnst.shape == (n_seeds, n_mb, n_wave)

    logqs_relerr = get_relerr(logqs_poporig, logqs_poprcnst, pnorm=2, rdcdims=-1)
    assert logqs_relerr.shape == (n_seeds, n_mb)
    assert not logqs_relerr.isnan().any()

    ###########################################################
    ######## The Scattering Efficiency Relative Error #########
    ###########################################################
    qa_relerr = get_relerr(qa_poporig, qa_poprcnst, pnorm=2, rdcdims=-1)
    assert qa_relerr.shape == (n_seeds, n_mb)
    assert not qa_relerr.isnan().any()

    logqa_poporig = qa_poporig.clip(min=1e-7).log()
    assert logqa_poporig.shape == (n_seeds, n_mb, n_wave)

    logqa_poprcnst = qa_poprcnst.clip(min=1e-7).log()
    assert logqa_poprcnst.shape == (n_seeds, n_mb, n_wave)

    logqa_relerr = get_relerr(logqa_poporig, logqa_poprcnst, pnorm=2, rdcdims=-1)
    assert logqa_relerr.shape == (n_seeds, n_mb)
    assert not logqa_relerr.isnan().any()

    ###########################################################
    #################### The Optical Error ####################
    ###########################################################
    # The optical error is the average of the scattering and absorption
    opt_err = (qs_relerr + qa_relerr) / 2.0
    assert opt_err.shape == (n_seeds, n_mb)
    assert not opt_err.isnan().any()

    logopt_err = (logqs_relerr + logqa_relerr) / 2.0
    assert logopt_err.shape == (n_seeds, n_mb)
    assert not logopt_err.isnan().any()

    opt_errinfo = dict(opt_err=opt_err, qs_relerr=qs_relerr, 
        qa_relerr=qa_relerr, logopt_err=opt_err, logqs_relerr=qs_relerr, 
        logqa_relerr=qa_relerr)

    return opt_errinfo

#########################################################
###### INP Frozen Fraction Error Utility Functions ######
#########################################################

@torch.no_grad()
def get_bulkfrznfrac(m_input, n_input, tmprtr_inpmin, tmprtr_inpmax, n_tmprtr, rho_chm, 
    chem_species, n_seeds, n_mb, n_chem, n_part, input_type='hst'):
    r"""
    Computes the aerosol "population" frozen fraction at a range of temperatures. 

    Here's a simple recipe:
    
        1. $$ n_{s,OIN}(T) = exp(-0.517 * (T - 273.15) + 8.934) $$

        2. $$ n_{s,BC}(T) = exp(1.844 - 0.687 * (T - 273.15) - 0.00597 * (T - 273.15)^2) $$

        3. $$ s_{ae,OIN} = \pi * d_{total}^2 * \frac{v_{OIN}}{v_{total}}^(2/3) $$

        4. $$ s_{ae,BC} = \pi * d_{total}^2 * \frac{v_{BC}}{v_{total}}^(2/3) $$

        5. $$ f_{i,j}(T) = 1 - exp(- s_{ae,BC} * n_{s,BC}(T) - s_{ae,OIN} * n_{s,OIN}(T)) $$

    This recipe is based on the following papers:

        1. "A particle-surface-area-based parameterization of immersion freezing on 
            desert dust particles." 

            Niemand, Monika, Ottmar Möhler, Bernhard Vogel, Heike Vogel, Corinna Hoose, 
            Paul Connolly, Holger Klein et al. 

            Journal of the Atmospheric Sciences 69, no. 10 (2012): 3077-3092.

            https://journals.ametsoc.org/view/journals/atsc/69/10/jas-d-11-0249.1.xml
        
        2. "The contribution of black carbon to global ice nucleating particle 
            concentrations relevant to mixed-phase clouds."

            Schill, Gregory P., Paul J. DeMott, Ethan W. Emerson, Anne Marie C. Rauker, 
            John K. Kodros, Kaitlyn J. Suski, Thomas CJ Hill et al.  

            Proceedings of the National Academy of Sciences 117, no. 37 (2020): 22705-22711.

            https://www.pnas.org/doi/10.1073/pnas.2001674117

    Once we have a frozen fraction spectrum for each particle, we can average it out with the 
    number of particles as the average weight, and report it as a bulk property.

    Parameters
    ----------
    m_chmprthst: (torch.tensor) The speciated mass of the chemicals within each diameter bin.
        `assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)`

    n_prthst: (torch.tensor) The total particle count within each diameter bin.
        `assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)`
    
    tmprtr_inpmin: (float) The lower end of the the temperature range in Celsius.
        Usually, this value should be around -40 degrees.

    tmprtr_inpmax: (float) The higher end of the the temperature range in Celsius.
        Usually, this value should be around 0 degrees.

    n_tmprtr: (int) The number of temperature bins.
        
    rho_chm: (torch.tensor) The physical density of the chemical species.
        `assert rho_chm.shape == (n_chem,)`

    chem_species: (list) The chemical species names. This will be used to find 
        the mineral dust (OIN) and black carbon (BC) chemicals.

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_part: (int) The number of particles/diameter bins.

    Returns
    -------
    frznfrac_info: (dict) the frozen fraction properties with a few keys:

        * frznfrac_tmp: (torch.tensor) the bulk frozen fractions at different 
            temperatures.
            `assert frznfrac_tmp.shape == (n_seeds, n_mb, n_tmprtr)`

        * temprtr_bins: (torch.tensor) the temperature range bins in Celsius.
            `assert temprtr_bins.shape == (n_tmprtr,)`
    """

    tch_device = m_input.device
    tch_dtype = m_input.dtype

    if input_type == 'hst':
        # Converting the histogram data into approximated particles
        m_chmprt, n_prt, n_partset, n_part = cnvrt_hist2prtcls(m_input, n_input, n_seeds, 
            n_mb, n_chem, n_part)
        assert m_chmprt.shape == (n_partset, n_part, n_chem)
        assert n_prt.shape == (n_partset, n_part)
    elif input_type == 'prt':
        m_chmprt_, n_prt_ = clean_prtcls(m_input, n_input, n_seeds, n_mb, n_part, n_chem)
        assert m_chmprt_.shape == (n_seeds, n_mb, n_part, n_chem)
        assert n_prt_.shape == (n_seeds, n_mb, n_part, 1)

        n_partset = n_seeds * n_mb
        m_chmprt = m_chmprt_.flatten(0, 1)
        assert m_chmprt.shape == (n_partset, n_part, n_chem)
        n_prt = n_prt_.flatten(0, 1).squeeze(dim=-1)
        assert n_prt.shape == (n_partset, n_part)
    else:
        raise ValueError(f'undefined input/type={input_type}')

    temprtr_bins = torch.linspace(tmprtr_inpmin, tmprtr_inpmax, 
        n_tmprtr, device=tch_device, dtype=tch_dtype)
    assert temprtr_bins.shape == (n_tmprtr,)

    ###########################################################
    ##### Full Particle Diameter, Volume, and Refractions #####
    ###########################################################
    # The volume of each chemical species within the particles
    v_chmprt = m_chmprt / rho_chm.reshape(1, 1, n_chem)
    assert v_chmprt.shape == (n_partset, n_part, n_chem)

    # The subset volume of each particle
    v_prt = v_chmprt.sum(axis=-1) + 1e-50
    assert v_prt.shape == (n_partset, n_part)

    # The subset diameter of each particle
    d_prt = (6 * v_prt / math.pi).pow(1.0 / 3.0)
    assert d_prt.shape == (n_partset, n_part)

    ###########################################################
    ###### Particle Diameter, Volume, and Surface Areas #######
    ###########################################################
    i_bc = chem_species.index('BC')

    # The surface area of the entire particle
    surf_prt = math.pi * d_prt.square()
    assert surf_prt.shape == (n_partset, n_part)

    # The OIN chemical index
    i_oin = chem_species.index('OIN')

    # The OIN volume of each particle
    v_oinprt = v_chmprt[..., i_oin]
    assert v_oinprt.shape == (n_partset, n_part)

    # The surface area of the OIN chemical
    surf_oin1 = surf_prt * (v_oinprt / v_prt).pow(2.0 / 3.0)
    assert surf_oin1.shape == (n_partset, n_part)

    # The BC chemical index
    i_bc = chem_species.index('BC')

    # The BC volume of each particle
    v_bcprt = v_chmprt[..., i_bc]
    assert v_bcprt.shape == (n_partset, n_part)

    # The surface area of the BC chemical
    surf_bc1 = surf_prt * (v_bcprt / v_prt).pow(2.0 / 3.0)
    assert surf_bc1.shape == (n_partset, n_part)

    ###########################################################
    ### Computing the Frozen Fraction at Input Temperatures ###
    ###########################################################

    # According to Equation 5 and Page 9 in 
    #   https://journals.ametsoc.org/view/journals/atsc/69/10/jas-d-11-0249.1.xml
    # we have
    #   $$ n_{s,OIN}(T) = exp(-0.517 * (T - 273.15) + 8.934) $$
    ns_oin = (-0.517 * temprtr_bins + 8.934).exp()
    assert ns_oin.shape == (n_tmprtr,)

    # According to Equation 2 and Page 4 in 
    #   https://www.pnas.org/doi/10.1073/pnas.2001674117
    # we have
    #   $$ n_{s,BC}(T) = exp(1.844 − 0.687 * T − 0.00597 * T^2) $$
    ns_bc = (1.844 - 0.687 * temprtr_bins - 0.00597 * temprtr_bins.square()).exp()
    assert ns_bc.shape == (n_tmprtr,)

    # The multiplication of $n_{s,OIN}$ by the OIN surface area
    surfns_oin = ns_oin.reshape(1, 1, n_tmprtr) * surf_oin1.reshape(n_partset, n_part, 1)
    assert surfns_oin.shape == (n_partset, n_part, n_tmprtr)

    # The multiplication of $n_{s,BC}$ by the BC surface area
    surfns_bc = ns_bc.reshape(1, 1, n_tmprtr) * surf_bc1.reshape(n_partset, n_part, 1)
    assert surfns_bc.shape == (n_partset, n_part, n_tmprtr)

    # The frozen fraction at each temperature
    #   $$ f_{i,j}(T) = 1 - exp(- s_{ae,BC} * n_{s,BC}(T) - s_{ae,OIN} * n_{s,OIN}(T)) $$
    frznfrac_prttmp = 1.0 - (- surfns_oin - surfns_bc).exp()
    assert frznfrac_prttmp.shape == (n_partset, n_part, n_tmprtr)

    # The survival function of the frozen particles at each temperature
    frznfrac_tmp1 = torch.einsum('ijk,ij->ik', frznfrac_prttmp, 
        n_prt / (n_prt.sum(dim=-1, keepdim=True) + 1e-30))
    assert frznfrac_tmp1.shape == (n_partset, n_tmprtr)

    # Making sure the frozen fraction is always non-negative
    frznfrac_tmp1[frznfrac_tmp1 <= 0.0] = 0.0 

    assert n_partset == (n_seeds * n_mb)
    frznfrac_tmp = frznfrac_tmp1.reshape(n_seeds, n_mb, n_tmprtr)
    assert frznfrac_tmp.shape == (n_seeds, n_mb, n_tmprtr)

    # The log-survival function of the frozen particles at each temperature
    logfrznfrac_tmp = (frznfrac_tmp + 1e-30).log()
    assert logfrznfrac_tmp.shape == (n_seeds, n_mb, n_tmprtr)

    # The OIN surface area
    surf_oin = surf_oin1.reshape(n_seeds, n_mb, n_part)
    assert surf_oin.shape == (n_seeds, n_mb, n_part)

    # The BC surface area
    surf_bc = surf_bc1.reshape(n_seeds, n_mb, n_part)
    assert surf_bc.shape == (n_seeds, n_mb, n_part)

    frznfrac_info = dict(frznfrac_tmp=frznfrac_tmp, logfrznfrac_tmp=logfrznfrac_tmp, 
        temprtr_bins=temprtr_bins, surf_oin=surf_oin, surf_bc=surf_bc, 
        frznfrac_prttmp=frznfrac_prttmp)

    return frznfrac_info

@torch.no_grad()
def get_inperr(x_mbs, xhat_mbs, tmprtr_inpmin, tmprtr_inpmax, n_tmprtr, rho_chm, 
    chem_species, n_seeds, n_mb, n_chem, n_part, input_type='hst'):
    """
    This function computes the INP frozen fraction error integral between an original 
    and reconstructed pair of data points. This is a composite function:

        1. First, it computes the INP frozen fraction spectrum of the original data 
            on a user-specified temperature interval using the  `get_bulkfrznfrac` function.
        
        2. Then, it does the same thing for the reconstructed data.

        3. Finally, it computes the integral of the absulte error between the two 
            CCN spectrum plots.
    
    Parameters
    ----------
    x_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the original data. 

    xhat_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the reconstructed data. 

    tmprtr_inpmin: (float) The lower end of the the temperature range in Celsius.
        Usually, this value should be around -40 degrees.

    tmprtr_inpmax: (float) The higher end of the the temperature range in Celsius.
        Usually, this value should be around 0 degrees.

    n_tmprtr: (int) The number of temperature bins.
        
    rho_chm: (torch.tensor) The physical density of the chemical species.
        `assert rho_chm.shape == (n_chem,)`

    chem_species: (list) The chemical species names. This will be used to find 
        the mineral dust (OIN) and black carbon (BC) chemicals.

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_part: (int) The number of particles/diameter bins.

    Returns
    -------
    inp_errinfo: (dict) a dictionary mapping a few keys to tensors:

        * inp_err: (torch.tensor) The computed integral of the absolute error 
            between the original and reconstructed frozen fraction curves.
            `assert inp_err.shape == (n_seeds, n_mb)`

        * loginp_err: (torch.tensor) The computed integral of the absolute error 
            between the original and reconstructed log frozen fraction curves.
            `assert loginp_err.shape == (n_seeds, n_mb)`

        * frznfrac_orig: (torch.tensor) The original frozen fraction spectrums.
            `assert frznfrac_orig.shape == (n_seeds, n_mb, n_tmprtr)`

        * frznfrac_rcnst: (torch.tensor) The reconstructed frozen fraction spectrums.
            `assert frznfrac_rcnst.shape == (n_seeds, n_mb, n_tmprtr)`

        * temprtr_bins: (torch.tensor) The temperature range bin values.
            `assert temprtr_bins.shape == (n_tmprtr,)`
    """
    m_key = {'hst': 'm_chmprthst', 'prt': 'm_prtchm'}[input_type]
    n_key = {'hst': 'n_prthst', 'prt': 'n_prt'}[input_type]

    # The original data's frozen fraction plot
    frznfrac_infoorig = get_bulkfrznfrac(x_mbs[m_key], x_mbs[n_key], 
        tmprtr_inpmin, tmprtr_inpmax, n_tmprtr, rho_chm, chem_species, 
        n_seeds, n_mb, n_chem, n_part, input_type)

    temprtr_bins = frznfrac_infoorig['temprtr_bins']
    assert temprtr_bins.shape == (n_tmprtr,)

    frznfrac_orig = frznfrac_infoorig['frznfrac_tmp']
    assert frznfrac_orig.shape == (n_seeds, n_mb, n_tmprtr)

    logfrznfrac_orig = frznfrac_infoorig['logfrznfrac_tmp']
    assert logfrznfrac_orig.shape == (n_seeds, n_mb, n_tmprtr)

    # The reconstructed data's frozen fraction plot
    frznfrac_inforcnst = get_bulkfrznfrac(xhat_mbs[m_key], xhat_mbs[n_key], 
        tmprtr_inpmin, tmprtr_inpmax, n_tmprtr, rho_chm, chem_species, 
        n_seeds, n_mb, n_chem, n_part, input_type)

    frznfrac_rcnst = frznfrac_inforcnst['frznfrac_tmp']
    assert frznfrac_rcnst.shape == (n_seeds, n_mb, n_tmprtr)

    logfrznfrac_rcnst = frznfrac_inforcnst['logfrznfrac_tmp']
    assert logfrznfrac_rcnst.shape == (n_seeds, n_mb, n_tmprtr)

    tmprtr_diff = tmprtr_inpmax - tmprtr_inpmin
    # The frozen fraction curve error
    inp_err = (frznfrac_orig - frznfrac_rcnst).abs().mean(dim=-1) * tmprtr_diff
    assert inp_err.shape == (n_seeds, n_mb)

    # The log-frozen fraction curve error
    loginp_err = (logfrznfrac_orig - logfrznfrac_rcnst).abs().mean(dim=-1) * tmprtr_diff
    assert loginp_err.shape == (n_seeds, n_mb)

    inp_errinfo = dict(inp_err=inp_err, loginp_err=loginp_err, 
        frznfrac_orig=frznfrac_orig, frznfrac_rcnst=frznfrac_rcnst, 
        temprtr_bins=temprtr_bins)

    return inp_errinfo

#########################################################
##### Diameter Distribution Error Utility Functions #####
#########################################################

def get_diamhst(m_prtchm, n_prt, rho_chm, n_seeds, n_mb, n_part, n_chem, n_bins, diam_low, 
    diam_high, diam_logbase, lib='numpy'):

    assert m_prtchm.shape == (n_seeds, n_mb, n_part, n_chem)
    assert n_prt.shape == (n_seeds, n_mb, n_part, 1)
    assert rho_chm.shape == (n_chem,)

    n_partset = n_seeds * n_mb

    m_prtchm2 = m_prtchm.reshape(n_partset, n_part, n_chem)
    assert m_prtchm2.shape == (n_partset, n_part, n_chem)
    n_prt2 = n_prt.reshape(n_partset, n_part)
    assert n_prt2.shape == (n_partset, n_part)
    
    # The volume of each chemical species within particles
    v_prtchm = m_prtchm2 / rho_chm.reshape(1, 1, n_chem)
    assert v_prtchm.shape == (n_partset, n_part, n_chem)

    # The volume of each particle
    v_prt = v_prtchm.sum(axis=2)
    assert v_prt.shape == (n_partset, n_part)

    # The diameter of each particle
    d_prt = (6 * v_prt / np.pi) ** (1 / 3)
    assert d_prt.shape == (n_partset, n_part)

    # The mass of each particle
    m_prt = m_prtchm2.sum(axis=2)
    assert m_prt.shape == (n_partset, n_part)

    # The normalized weight of chemicals within each particle
    w_prtchm = m_prtchm2 / m_prt.reshape(n_partset, n_part, 1)
    assert w_prtchm.shape == (n_partset, n_part, n_chem)

    # The log-dimeter logistics
    logdiam_low, logdiam_high = np.log(diam_low) / diam_logbase, np.log(diam_high) / diam_logbase

    # The log-diameter of all particles
    lib_log = np.log if lib == 'numpy' else torch.log
    logd_prt = lib_log(d_prt) / diam_logbase
    assert logd_prt.shape == (n_partset, n_part)

    # The normalized log-dimater within the `[logdiam_high, logdiam_low]` range.
    logdn_part = (logd_prt - logdiam_low) / (logdiam_high - logdiam_low)
    assert logdn_part.shape == (n_partset, n_part)

    # The log-dimater bin of each particle
    dbin_part = (logdn_part * n_bins)
    dbin_part = (dbin_part.astype(np.int64) if lib == 'numpy'
        else dbin_part.to(torch.long))
    dbin_part[dbin_part >= n_bins] = n_bins - 1
    dbin_part[dbin_part < 0] = 0
    assert dbin_part.shape == (n_partset, n_part)

    if lib == 'numpy':
        arange_nmb = np.arange(n_mb)[:, None]
        assert arange_nmb.shape == (n_partset, 1)

        # The number of particles histogram within each diamater bin
        n_prthst = np.zeros((n_partset, n_bins), n_prt2.dtype)
        np.add.at(n_prthst, (arange_nmb, dbin_part), n_prt2)
        assert n_prthst.shape == (n_partset, n_bins)

        # The total mass of particles within each diameter bin
        m_prthst = np.zeros((n_partset, n_bins), np.float64)
        np.add.at(m_prthst, (arange_nmb, dbin_part), m_prt * n_prt2)
        assert m_prthst.shape == (n_partset, n_bins)

        # The total mass of each chemical species in each diameter bin
        m_chmprthst_ = np.zeros((n_partset, n_bins, n_chem), np.float64)
        np.add.at(m_chmprthst_, (arange_nmb, dbin_part), 
            m_prtchm2 * n_prt2.reshape(n_partset, n_part, 1))
        assert m_chmprthst_.shape == (n_partset, n_bins, n_chem)

        # Making the mass data "channels"-compatible
        m_chmprthst = np.moveaxis(m_chmprthst_, -1, -2)
        assert m_chmprthst.shape == (n_partset, n_chem, n_bins)
    elif lib == 'torch':
        # The number of particles histogram within each diamater bin
        n_prthst = torch.zeros(n_partset, n_bins, dtype=n_prt2.dtype, device=n_prt2.device)
        n_prthst.scatter_add_(dim=1, index=dbin_part, src=n_prt2)
        assert n_prthst.shape == (n_partset, n_bins)

        # The total mass of particles within each diameter bin
        m_prthst = torch.zeros(n_partset, n_bins, dtype=m_prt.dtype, device=m_prt.device)
        m_prthst.scatter_add_(dim=1, index=dbin_part, src=m_prt * n_prt2)
        assert m_prthst.shape == (n_partset, n_bins)

        # The total mass of each chemical species in each diameter bin
        m_chmprthst_ = torch.zeros(n_partset, n_bins, n_chem, dtype=m_prt.dtype, device=m_prt.device)
        m_chmprthst_.scatter_add_(dim=1, 
            index=dbin_part.reshape(n_partset, n_part, 1).expand(n_partset, n_part, n_chem), 
            src=m_prtchm2 * n_prt2.reshape(n_partset, n_part, 1))
        assert m_chmprthst_.shape == (n_partset, n_bins, n_chem)

        m_chmprthst = m_chmprthst_.transpose(-1, -2)
        assert m_chmprthst.shape == (n_partset, n_chem, n_bins)
    else:
        raise ValueError(f'lib={lib} is not implemented')

    n_prthstu = n_prthst.reshape(n_seeds, n_mb, 1, n_bins)
    assert n_prthstu.shape == (n_seeds, n_mb, 1, n_bins)

    m_prthstu = m_prthst.reshape(n_seeds, n_mb, 1, n_bins)
    assert m_prthstu.shape == (n_seeds, n_mb, 1, n_bins)

    m_chmprthstu = m_chmprthst.reshape(n_seeds, n_mb, n_chem, n_bins)
    assert m_chmprthstu.shape == (n_seeds, n_mb, n_chem, n_bins)

    out_dict = dict(n_prthst=n_prthstu, m_prthst=m_prthstu, 
        m_chmprthst=m_chmprthstu)

    return out_dict

def get_diamhstsngl(m_prtchm, n_prt, rho_chm, n_part, n_chem, n_bins, diam_low, 
    diam_high, diam_logbase, lib='numpy', verbose=False):
    
    # The volume of each chemical species within particles
    v_prtchm = m_prtchm / rho_chm.reshape(1, n_chem)
    assert v_prtchm.shape == (n_part, n_chem)

    # The volume of each particle
    v_prt = v_prtchm.sum(axis=1)
    assert v_prt.shape == (n_part,)

    # The diameter of each particle
    d_prt = (6 * v_prt / np.pi) ** (1 / 3)
    assert d_prt.shape == (n_part,)

    # The mass of each particle
    m_prt = m_prtchm.sum(axis=1)
    assert m_prt.shape == (n_part,)

    # The normalized weight of chemicals within each particle
    w_prtchm = m_prtchm / m_prt.reshape(n_part, 1)
    assert w_prtchm.shape == (n_part, n_chem)

    # The volume of each chemical species within particles
    v_prtchm = m_prtchm / rho_chm.reshape(1, n_chem)
    assert v_prtchm.shape == (n_part, n_chem)

    # The volume of each particle
    v_prt = v_prtchm.sum(axis=1)
    assert v_prt.shape == (n_part,)

    # The diameter of each particle
    d_prt = (6 * v_prt / np.pi) ** (1 / 3)
    assert d_prt.shape == (n_part,)

    # The mass of each particle
    m_prt = m_prtchm.sum(axis=1)
    assert m_prt.shape == (n_part,)

    # The normalized weight of chemicals within each particle
    w_prtchm = m_prtchm / m_prt.reshape(n_part, 1)
    assert w_prtchm.shape == (n_part, n_chem)

    # The log-dimeter logistics
    logdiam_low, logdiam_high = np.log(diam_low) / diam_logbase, np.log(diam_high) / diam_logbase
    if verbose and not (d_prt >= diam_low).all():
        print(f'\nFound diameters as small as {d_prt.min()}.')
    if verbose and not (d_prt < diam_high).all():
        print(f'\nFound diameters as large as {d_prt.max()}.')

    # The log-diameter of all particles
    logd_prt = (np.log(d_prt) / diam_logbase) if (lib == 'numpy') else (d_prt.log() / diam_logbase)
    assert logd_prt.shape == (n_part,)

    # The normalized log-dimater within the `[logdiam_high, logdiam_low]` range.
    logdn_part = (logd_prt - logdiam_low) / (logdiam_high - logdiam_low)
    assert logdn_part.shape == (n_part,)

    # The log-dimater bin of each particle
    dbin_part = (logdn_part * n_bins)
    dbin_part = (dbin_part.astype(np.int64) if (lib == 'numpy')
        else dbin_part.to(torch.long))
    dbin_part[dbin_part >= n_bins] = n_bins - 1
    dbin_part[dbin_part < 0] = 0
    assert dbin_part.shape == (n_part,)

    if lib == 'numpy':
        # The number of particles histogram within each diamater bin
        n_prthst = np.zeros(n_bins, n_prt.dtype)
        np.add.at(n_prthst, dbin_part, n_prt)
        assert n_prthst.shape == (n_bins,)

        # The total mass of particles within each diameter bin
        m_prthst = np.zeros(n_bins, np.float64)
        np.add.at(m_prthst, dbin_part, m_prt * n_prt)
        assert m_prthst.shape == (n_bins,)

        # The total mass of each chemical species in each diameter bin
        m_chmprthst_ = np.zeros((n_bins, n_chem), np.float64)
        np.add.at(m_chmprthst_, dbin_part, m_prtchm * n_prt.reshape(n_part, 1))
        assert m_chmprthst_.shape == (n_bins, n_chem)
    elif lib == 'torch':
        # The number of particles histogram within each diamater bin
        n_prthst = torch.zeros(n_bins, dtype=n_prt.dtype, device=n_prt.device)
        n_prthst.scatter_add_(dim=0, index=dbin_part, src=n_prt)
        assert n_prthst.shape == (n_bins,)

        # The total mass of particles within each diameter bin
        m_prthst = torch.zeros(n_bins, dtype=m_prt.dtype, device=m_prt.device)
        m_prthst.scatter_add_(dim=0, index=dbin_part, src=m_prt * n_prt)
        assert m_prthst.shape == (n_bins,)

        # The total mass of each chemical species in each diameter bin
        m_chmprthst_ = torch.zeros(n_bins, n_chem, dtype=m_prt.dtype, device=m_prt.device)
        m_chmprthst_.scatter_add_(dim=0, index=dbin_part.reshape(n_part, 1).expand(n_part, n_chem), 
            src=m_prtchm * n_prt.reshape(n_part, 1))
        assert m_chmprthst_.shape == (n_bins, n_chem)
    else:
        raise ValueError(f'lib={lib} is not implemented')

    # Making the mass data "channels"-compatible
    m_chmprthst = m_chmprthst_.T
    assert m_chmprthst.shape == (n_chem, n_bins)

    out_dict = dict(n_prthst=n_prthst, m_prthst=m_prthst, 
        m_chmprthst=m_chmprthst)

    return out_dict

def test_diamhst(h5_path):
    with h5py.File(h5_path, "a") as h5fp:
        m_prtchm1 = h5fp_out['m_prtchm'][:]
        n_pop, n_part, n_chem = m_prtchm1.shape
        assert m_prtchm1.shape == (n_pop, n_part, n_chem)
        n_prt1 = h5fp_out['n_prt'][:]
        assert n_prt1.shape == (n_pop, n_part)
        rho_chm1 = h5fp_out['chem_rho'][:]
        assert rho_chm1.shape == (n_chem,)
        print(f'Finished Loading the Data')

    # Numpy Data
    m_prtchm2 = m_prtchm1.reshape(n_pop, n_part, n_chem)
    assert m_prtchm2.shape == (n_pop, n_part, n_chem)
    n_prt2 = n_prt1.reshape(n_pop, n_part)
    assert n_prt2.shape == (n_pop, n_part)
    rho_chm2 = rho_chm1
    assert rho_chm2.shape == (n_chem,)

    # PyTorch Data
    tch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    m_prtchm3 = torch.from_numpy(m_prtchm2).to(dtype=torch.float64, device=tch_device)
    assert m_prtchm3.shape == (n_pop, n_part, n_chem)
    n_prt3 = torch.from_numpy(n_prt2).to(dtype=torch.float64, device=tch_device)
    assert n_prt3.shape == (n_pop, n_part)
    rho_chm3 = torch.from_numpy(rho_chm2).to(dtype=torch.float64, device=tch_device)
    assert rho_chm3.shape == (n_chem,)

    # Single population at a time - Numpy
    hist_datas1np = defaultdict(list)
    for i_samp in range(n_pop):
        hist_data1 = get_diamhstsngl(m_prtchm2[i_samp], n_prt2[i_samp], 
            rho_chm2, n_part, n_chem, n_bins, diam_low, 
            diam_high, diam_logbase, lib='numpy', verbose=False)
        for key, val in hist_data1.items():
            hist_datas1np[key].append(val)
    hist_datas1np = {key: np.stack(val, axis=0) for key, val in hist_datas1np.items()}
    print(f'Finished single population at a time with Numpy')

    # Single population at a time - PyTorch
    hist_datas1th = defaultdict(list)
    for i_samp in range(n_pop):
        hist_data1 = get_diamhstsngl(m_prtchm3[i_samp], n_prt3[i_samp], 
            rho_chm3, n_part, n_chem, n_bins, diam_low, 
            diam_high, diam_logbase, lib='torch', verbose=False)
        for key, val in hist_data1.items():
            hist_datas1th[key].append(val)
    hist_datas1th = {key: torch.stack(val, axis=0) for key, val in hist_datas1th.items()}
    print(f'Finished single population at a time with PyTorch')

    # Multiple population at a time - Numpy
    hist_datas2np = get_diamhst(m_prtchm2[None], n_prt2[None], 
        rho_chm2, 1, n_pop, n_part, n_chem, n_bins, diam_low, 
        diam_high, diam_logbase, lib='numpy')
    print(f'Finished multiple population at a time with numpy')

    # Multiple population at a time - PyTorch
    hist_datas2th = get_diamhst(m_prtchm3[None], n_prt3[None], 
        rho_chm3, 1, n_pop, n_part, n_chem, n_bins, diam_low, 
        diam_high, diam_logbase, lib='torch')
    print(f'Finished multiple population at a time with PyTorch')

    for key in ['m_chmprthst', 'm_prthst', 'n_prthst']:
        aaa = hist_datas1np[key]
        bbb = hist_datas2th[key][0]

        aaaa = aaa.detach().cpu().numpy() if torch.is_tensor(aaa) else aaa
        bbbb = bbb.detach().cpu().numpy() if torch.is_tensor(bbb) else bbb
        
        assert np.allclose(aaaa, bbbb)
    print(f'All tests were successful!')

#########################################################
########### Relative Error Utility Functions ############
#########################################################

@torch.no_grad()
def get_relerr(x_mb, xhat_mb, pnorm=2, rdcdims=(-1, -2)):
    """
    This function computes the relative error between two tensors. The 
    last two dimensions are data dimensions, and the second dimension 
    is the mini-batch dimension.

    Parameters
    ----------
    x_mb: (torch.tensor) The reference data.
        `assert x_mb.shape == (n_seeds, n_mb, n_chnls, n_len)`

    xhat_mb: (torch.tensor) The predicted data.
        `assert xhat_mb.shape == (n_seeds, n_mb, n_chnls, n_len)`
    
    pnorm: (float | int) The relative error p-norm.

    rdcdims: (list | tuple | int) The reduction (data) dimensions. These are 
        the opposite/complementaries of the batch dimensions which 
        would be preserved in the output. 

    Returns
    -------
    x_relerr: (torch.tensor) The relative error.
        `assert x_relerr.shape == (n_seeds, n_mb)`
    """
    assert pnorm > 0
    assert x_mb.shape == xhat_mb.shape
    rdcdims = (rdcdims,) if isinstance(rdcdims, int) else rdcdims
    assert isinstance(rdcdims, (list, tuple))
    assert all(isinstance(dim, int) for dim in rdcdims)
    
    n_dims = x_mb.ndim
    rdc_dims = [dim % n_dims for dim in rdcdims]
    mb_dims = tuple(x_mb.shape[dim] for dim in range(n_dims) if dim not in rdc_dims)

    x_relerrnum = (x_mb - xhat_mb).abs().pow(pnorm).sum(dim=rdc_dims).pow(1.0 / pnorm)
    assert x_relerrnum.shape == mb_dims
    assert not x_relerrnum.isnan().any()

    x_relerrdenom1 = x_mb.abs().pow(pnorm).sum(dim=rdc_dims).pow(1.0 / pnorm)
    assert x_relerrdenom1.shape == mb_dims
    assert not x_relerrdenom1.isnan().any()

    x_relerrdenom2 = xhat_mb.abs().pow(pnorm).sum(dim=rdc_dims).pow(1.0 / pnorm)
    assert x_relerrdenom2.shape == mb_dims
    assert not x_relerrdenom2.isnan().any()

    x_relerrdenom = x_relerrdenom1 + x_relerrdenom2 + 1e-30
    assert x_relerrdenom.shape == mb_dims
    assert not x_relerrdenom.isnan().any()

    x_relerr = 2 * x_relerrnum / x_relerrdenom
    assert x_relerr.shape == mb_dims
    assert not x_relerr.isnan().any()
    assert (x_relerr >= (0.0 - 1e-10)).all()
    assert (x_relerr <= (2.0 + 1e-10)).all()

    return x_relerr

@torch.no_grad()
def get_histrelerrs(x_mbs, xhat_mbs, n_seeds, n_mb, n_chem, n_bins, pnorm=2):
    """
    Computes the Relative error between two aerosol mass and counts distributions.

    Parameters
    ----------
    x_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the original data. 

    xhat_mbs: (dict) a dictionary containing the `('m_chmprthst', 'n_prthst')` keys 
        mapping to pytorch tensors for the reconstructed data. 

    n_seeds: (int) The number of rng seeds.

    n_mb: (int) The mini-batch size within each random seed.

    n_chem: (int) The total number of chemical species including water.

    n_bins: (int) The number of diameter bins.

    pnorm: (float | int) The relative error p-norm.

    Returns
    -------
    relerr_info: (dict) a dictionary containing multiple relative erros.

        n_relerr: (torch.tensor) The particle counts relative error.
            `assert n_relerr.shape == (n_seeds, n_mb)`
        
        m_chmrelerr: (torch.tensor) The speciated mass relative error.
            `assert m_chmrelerr.shape == (n_seeds, n_mb)`
        
        m_relerr: (torch.tensor) The total mass relative error.
            `assert m_relerr.shape == (n_seeds, n_mb)`
        
        m_perchmrelerr: (torch.tensor) The average per-species mass relative error.
            `assert m_perchmrelerr.shape == (n_seeds, n_mb)`

    """
    m_chmprthst, n_prthst = clean_hist(x_mbs['m_chmprthst'], x_mbs['n_prthst'], 
        n_seeds, n_mb, n_chem, n_bins)
    assert n_prthst.shape == (n_seeds, n_mb, 1, n_bins)
    assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)

    mhat_chmprthst, nhat_prthst = clean_hist(xhat_mbs['m_chmprthst'], xhat_mbs['n_prthst'], 
        n_seeds, n_mb, n_chem, n_bins)
    assert nhat_prthst.shape == (n_seeds, n_mb, 1, n_bins)
    assert mhat_chmprthst.shape == (n_seeds, n_mb, n_chem, n_bins)

    #####################################################################
    ################ Computing the counts relative error ################
    #####################################################################
    n_relerr = get_relerr(n_prthst, nhat_prthst, pnorm=pnorm)
    assert n_relerr.shape == (n_seeds, n_mb)

    #####################################################################
    ############ Computing the speciated mass relative error ############
    #####################################################################
    m_chmrelerr = get_relerr(m_chmprthst, mhat_chmprthst, 
        pnorm=pnorm)
    assert m_chmrelerr.shape == (n_seeds, n_mb)

    #####################################################################
    ############## Computing the total mass relative error ##############
    #####################################################################
    m_prthst = m_chmprthst.sum(dim=-2, keepdims=True)
    assert m_prthst.shape == (n_seeds, n_mb, 1, n_bins)
    mhat_prthst = mhat_chmprthst.sum(dim=-2, keepdims=True)
    assert mhat_prthst.shape == (n_seeds, n_mb, 1, n_bins)

    m_relerr = get_relerr(m_prthst, mhat_prthst, pnorm=pnorm)
    assert m_relerr.shape == (n_seeds, n_mb)

    #####################################################################
    ######### Computing the per-species mass relative error #############
    #####################################################################
    # This is different from "speciated mass relative error".
    #
    # In this version, we want to compute the per-species relative error of the 
    # mass histograms, and then average them out across all chemical species.
    # One trick to avhieve this, is to merge the `n_mb` and `n_chem` dimensions 
    # and think of the `n_chem` dimension as an extra sample dimension.
    m_perchmprthst = m_chmprthst.flatten(1, 2).unsqueeze(2)
    assert m_perchmprthst.shape == (n_seeds, n_mb * n_chem, 1, n_bins)
    mhat_perchmprthst = mhat_chmprthst.flatten(1, 2).unsqueeze(2)
    assert mhat_perchmprthst.shape == (n_seeds, n_mb * n_chem, 1, n_bins)

    m_perchmrelerr1 = get_relerr(m_perchmprthst, mhat_perchmprthst, 
        pnorm=pnorm)
    assert m_perchmrelerr1.shape == (n_seeds, n_mb * n_chem)

    m_perchmrelerr = m_perchmrelerr1.reshape(n_seeds, n_mb, n_chem).mean(dim=-1)
    assert m_perchmrelerr.shape == (n_seeds, n_mb)

    relerr_info = dict(n_relerr=n_relerr, m_chmrelerr=m_chmrelerr, 
        m_relerr=m_relerr, m_perchmrelerr=m_perchmrelerr)

    return relerr_info

#########################################################
############# Aerosol Metric Pre-Processor ##############
#########################################################

class AeroMeasures(nn.Module):
    def __init__(self, pp_config: dict | None, device: torch.device, 
        dtype: torch.dtype, cdtype: torch.dtype):
        """
        Creates an aersol measurement pre-processor with a forward call.

        Args:
            pp_config (dict): Transformation specification dictionary.

                Example:
                    ```
                    pp_config = {
                        # The physical processes to be computationally measured
                        hparams/measures: ['ccn_cdf', 'qs_pop', 'qs_prt', 'qa_pop', 
                            'qa_prt', 'frznfrac_tmp', 'logfrznfrac_tmp', 'surf_oin', 
                            'surf_bc']

                        #######################################
                        ### The Aerosol Histogram Measures ####
                        #######################################
                        # The number of diameter histogram bins
                        hparams/aerohst/n_bins: 20
                        # The lower range of the diamter histogram
                        hparams/aerohst/diam_low: 1e-9
                        # The higher range of the diamter histogram
                        hparams/aerohst/diam_high: 1e-4
                        # The logarithmic base for diameter
                        hparams/aerohst/diam_logbase: 2.30258509299
                        # The evaluation mini-batch size for computing the metric
                        hparams/aerohst/eval/bs: 512

                        #######################################
                        ###### The Aerosol CCN Spectrum #######
                        #######################################
                        # The minimum epsilon value for the CDF interval.
                        hparams/aeroccn/eps_histmin: 1e-3
                        # The maximum epsilon value for the CDF interval.
                        hparams/aeroccn/eps_histmax: 1e-1
                        # The number of bins to divide the log-epsilon interval into.
                        hparams/aeroccn/n_epshist: 100
                        # The temperature used to compute the critical relative humidity values; 
                        # the `A` factor in the computation relies on this temperature.
                        hparams/aeroccn/tmprtr_sclr: 287.0
                        # The number of newtonian root finding iterations to compute the critical diameters.
                        hparams/aeroccn/void_kappa: 0.0
                        # The number of newtonian root finding iterations to compute the critical diameters.
                        hparams/aeroccn/n_rfiters: 150
                        # The relative update size stopping threshold for the newtonian root finder.
                        hparams/aeroccn/e_rfiters: 1e-10
                        # The evaluation mini-batch size for computing the metric
                        hparams/aeroccn/eval/bs: 512
                        # Whether to be strict about the newton root finding convergence
                        hparams/aeroccn/strict: true
                        # The input data type for the ccn calculations. One of prt, hst, or null.
                        hparams/aeroccn/input/type: null

                        #######################################
                        ## The Aerosol Optical Efficiencies ###
                        #######################################
                        # The optical wave length for measuring the optical efficiency properties.
                        hparams/aeroopt/wave_length: [300e-9, 400e-9, 550e-9, 600e-9, 1000e-9]
                        # The number of iterations needed for the mie sequence summations to converge.
                        hparams/aeroopt/n_max: 120
                        # The optical refraction coefficient for particles with zero volume.
                        hparams/aeroopt/void_refr: '1.5+0.0j'
                        # Whether the output shapes should be [n_part, n_wave] (true) or [n_wave, n_part] (false)
                        hparams/aeroopt/trnsps: false
                        # The evaluation mini-batch size for computing the metric
                        hparams/aeroopt/eval/bs: 512
                        # The input data type for the optical calculations. One of prt, hst, or null.
                        hparams/aeroopt/input/type: null
                        
                        #######################################
                        ###### The Aerosol INP Measures #######
                        #######################################
                        # The lower end of the the temperature range in Celsius.
                        # Usually, this value should be around -40 degrees.
                        hparams/aeroinp/tmprtr_inpmin: -40.0
                        # The higher end of the the temperature range in Celsius.
                        # Usually, this value should be around 0 degrees.
                        hparams/aeroinp/tmprtr_inpmax: 0.0
                        # The number of temperature bins.
                        hparams/aeroinp/n_tmprtr: 100
                        # Whether the output shapes should be [n_part, 1] (true) or [1, n_part] (false)
                        hparams/aeroinp/trnsps: false
                        # The evaluation mini-batch size for computing the metric
                        hparams/aeroinp/eval/bs: 512
                        # The input data type for the inp calculations. One of prt, hst, or null.
                        hparams/aeroinp/input/type: null

                        #######################################
                        ### The Aerosol Physical Constants ####
                        ####### (Direct Specification) ########
                        #######################################
                        hparams/aerocst/chem_species: 
                        hparams/aerocst/n_chem: 
                        hparams/aerocst/n_bins:
                        hparams/aerocst/n_part: 
                        hparams/aerocst/rho_chm: 
                        hparams/aerocst/kappa_chm: 
                        hparams/aerocst/refr_wvchmref: 
                        hparams/aerocst/len_wvref:

                        #######################################
                        ### The Aerosol Physical Constants ####
                        ###### (Indirect Specification) #######
                        #######################################
                        # These will be popped and converted to the above (direct) constants.
                        # Specifying the following correctly can generate the above (direct) constants entirely.

                        # The chemical species names
                        hparams/aerocst/chem_species: 
                        # The number of particles/diameter bins
                        hparams/aerocst/n_part: 
                        # The mapping of chemical species (str) to physical density values (float, kg/m^3)
                        hparams/aerocst/chem2rho: 
                        # The mapping of chemical species (str) to physical density values (float, kg/m^3)
                        hparams/aerocst/chem2kappa: 
                        # The mapping of wave length id (str) to wave length
                        hparams/aerocst/wvid2len: 
                        # The mapping of chemical species (str) to refractive index values (complex) for optical calculations
                        hparams/aerocst/chem2refr: 
                    }
                    ```

            device (torch.device | str): The pytorch device of the tensors.

            dtype (torch.dtype | str): The pytorch float dtype of the tensors.

            cdtype (torch.dtype | str): The pytorch complex dtype of the tensors.
        """
        super().__init__()
        if pp_config is not None:
            # Converting the hierarchical options dictionary into a deep one.
            pp_config2 = deep2hie(pp_config, dictcls=odict, sep="/")
        else:
            pp_config2 = dict()

        pp_dflthps = {
            'hparams/measures': ['ccn_cdf', 'qs_pop', 'qs_prt', 'qscs_prt', 
                'qa_pop', 'qa_prt', 'qa_csprt', 'frznfrac_tmp', 'logfrznfrac_tmp', 
                'surf_oin', 'surf_bc'],
            'hparams/input/type': 'hst', 
            'hparams/output/type': None,
            'hparams/aerohst/n_bins': 20,
            'hparams/aerohst/diam_low': 1e-9,
            'hparams/aerohst/diam_high': 1e-4,
            'hparams/aerohst/diam_logbase': math.log(10),
            'hparams/aerohst/eval/bs': 512,
            'hparams/aeroccn/eps_histmin': 1e-3,
            'hparams/aeroccn/eps_histmax': 1e-1,
            'hparams/aeroccn/n_epshist': 100,
            'hparams/aeroccn/tmprtr_sclr': 287.0,
            'hparams/aeroccn/void_kappa': 0.0,
            'hparams/aeroccn/n_rfiters': 150,
            'hparams/aeroccn/e_rfiters': 1e-10,
            'hparams/aeroccn/eval/bs': 512,
            'hparams/aeroccn/strict': True,
            'hparams/aeroccn/input/type': None,
            'hparams/aeroopt/wave_length': [300e-9, 400e-9, 550e-9, 600e-9, 1000e-9],
            'hparams/aeroopt/n_max': 120,
            'hparams/aeroopt/void_refr': '1.5+0.0j',
            'hparams/aeroopt/trnsps': False,
            'hparams/aeroopt/eval/bs': 128,
            'hparams/aeroopt/input/type': None,
            'hparams/aeroinp/tmprtr_inpmin': -40.0,
            'hparams/aeroinp/tmprtr_inpmax': 0.0,
            'hparams/aeroinp/n_tmprtr': 100,
            'hparams/aeroinp/trnsps': False,
            'hparams/aeroinp/eval/bs': 512,
            'hparams/aeroinp/input/type': None,
            'hparams/aerocst/n_bins': None,
            'hparams/aerocst/n_part': None}
        
        # Setting up the default options
        pp_config3 = {**pp_dflthps, **pp_config2}
        measures = set(pp_config3['hparams/measures'])

        # Sanity checking the aero constants, and interpolating the necessary 
        # wave-lengths for the refraction constants.
        self.vrfy_aerocsts(pp_config3, measures, device, dtype, cdtype)

        self.pp_config = dict(pp_config3)
        self.n_seeds = None
        
        # Pytorch device and data type
        self.tch_device = device
        self.tch_dtype = dtype
        self.tch_cdtype = cdtype

    @staticmethod
    @torch.no_grad()
    def vrfy_aerocsts(pp_config, measures, device, dtype, cdtype):
        diag_types = AeroMeasures.get_diagtypes(measures)
        need_hstcalc = 'hst' in diag_types
        need_ccncalc = 'ccn' in diag_types
        need_optcalc = 'opt' in diag_types
        need_inpcalc = 'inp' in diag_types

        # Sanity checking `chem_species`
        chem_species = pp_config.get('hparams/aerocst/chem_species', None)
        if isinstance(chem_species, str):
            chem_species = chem_species.split(',')

        # Sanity checking `n_chem`
        n_chem = pp_config.get('hparams/aerocst/n_chem', None)
        if n_chem is None:
            assert chem_species is not None
            n_chem = len(chem_species)

        # Unifying the `n_part` and `n_bins` variables
        n_bins = pp_config.pop('hparams/aerocst/n_bins', None)
        n_part_ = pp_config.get('hparams/aerocst/n_part')
        if n_bins is not None:
            assert n_part_ is None, 'Only one of n_bins or n_part must be specified'
            n_part = n_bins
        elif n_part_ is not None:
            assert n_bins is None, 'Only one of n_bins or n_part must be specified'
            n_part = n_part_
        else:
            raise ValueError(f'Either n_bins or n_part must be specified')

        # Sanity checking `rho_chm`: The chemical density values in kg / m^3.
        rho_chm1 = pp_config.get('hparams/aerocst/rho_chm', None)
        chem2rho = get_subdict(pp_config, 'hparams/aerocst/chem2rho', pop=True)
        if len(chem2rho) > 0:
            assert chem_species is not None
            rho_chmnp2 = np.array([chem2rho[chem] for chem in chem_species], dtype=np.float64)
            assert rho_chmnp2.shape == (n_chem,)
            rho_chm2 = torch.from_numpy(rho_chmnp2).to(device=device, dtype=dtype)
            assert rho_chm2.shape == (n_chem,)
        else:
            rho_chm2 = None

        if rho_chm1 is None:
            assert rho_chm2 is not None, 'either chem2rho or rho_chm must be specified'
            rho_chm = rho_chm2
            assert rho_chm.shape == (n_chem,)
        elif torch.is_tensor(rho_chm1):
            rho_chm = rho_chm1.to(device=device, dtype=dtype)
            assert rho_chm.shape == (n_chem,)
            assert (rho_chm2 is None) or rho_chm.allclose(rho_chm2), 'mismatch in the rho_chm inputs'
        elif isinstance(rho_chm1, np.ndarray):
            rho_chm = torch.from_numpy(rho_chm1).to(device=device, dtype=dtype)
            assert rho_chm.shape == (n_chem,)
            assert (rho_chm2 is None) or rho_chm.allclose(rho_chm2), 'mismatch in the rho_chm inputs'
        else:
            raise ValueError(f'undefined rho_chm={rho_chm1}')

        # Sanity checking `kappa_chm`: The chemical kappa values for CCN calculations
        kappa_chm1 = pp_config.get('hparams/aerocst/kappa_chm', None)
        chem2kappa = get_subdict(pp_config, 'hparams/aerocst/chem2kappa', pop=True)
        if len(chem2kappa) > 0:
            assert chem_species is not None
            kappa_chmnp2 = np.array([chem2kappa[chem] for chem in chem_species], dtype=np.float64)
            assert kappa_chmnp2.shape == (n_chem,)
            kappa_chm2 = torch.from_numpy(kappa_chmnp2).to(device=device, dtype=dtype)
            assert kappa_chm2.shape == (n_chem,)
        else:
            kappa_chm2 = None

        if kappa_chm1 is None:
            assert kappa_chm2 is not None, 'either chem2kappa or kappa_chm must be specified'
            kappa_chm = kappa_chm2
            assert kappa_chm.shape == (n_chem,)
        elif torch.is_tensor(kappa_chm1):
            kappa_chm = kappa_chm1.to(device=device, dtype=dtype)
            assert kappa_chm.shape == (n_chem,)
            assert (kappa_chm2 is None) or kappa_chm.allclose(kappa_chm2), 'mismatch in the kappa_chm inputs'
        elif isinstance(kappa_chm1, np.ndarray):
            kappa_chm = torch.from_numpy(kappa_chm1).to(device=device, dtype=dtype)
            assert kappa_chm.shape == (n_chem,)
            assert (kappa_chm2 is None) or kappa_chm.allclose(kappa_chm2), 'mismatch in the kappa_chm inputs'
        else:
            raise ValueError(f'undefined kappa_chm={kappa_chm1}')

        # Sanity checking `len_wvref`: The reference wave-lengths
        len_wvref1 = pp_config.get('hparams/aerocst/len_wvref', None)

        wave_idssrtd, n_waveref = None, None
        wvid2len = get_subdict(pp_config, 'hparams/aerocst/wvid2len', pop=True)
        if len(wvid2len) > 0:
            n_waveref2 = len(wvid2len)
            wave_idssrtd = [wvid for wvid, wvlen in sorted(wvid2len.items(), key=lambda pair: pair[1])]
            len_wvrefnp2 = np.array([wvid2len[wvid] for wvid in wave_idssrtd])
            assert len_wvrefnp2.shape == (n_waveref2,)
            len_wvref2 = torch.from_numpy(len_wvrefnp2).to(device=device, dtype=dtype)
            assert len_wvref2.shape == (n_waveref2,)
        else:
            len_wvref2, n_waveref2 = None, None

        if len_wvref1 is None:
            assert len_wvref2 is not None, 'either wvid2len or len_wvref must be specified'
            len_wvref, n_waveref = len_wvref2, n_waveref2
            assert len_wvref.shape == (n_waveref,)
        elif torch.is_tensor(len_wvref1):
            n_waveref = len_wvref1.numel()
            len_wvref = len_wvref1.to(device=device, dtype=dtype)
            assert len_wvref.shape == (n_waveref,)
            assert (len_wvref2 is None) or len_wvref.allclose(len_wvref2), 'mismatch in the len_wvref inputs'
        elif isinstance(len_wvref1, np.ndarray):
            n_waveref = len_wvref1.size
            len_wvref = torch.from_numpy(len_wvref1).to(device=device, dtype=dtype)
            assert len_wvref.shape == (n_waveref,)
            assert (len_wvref2 is None) or len_wvref.allclose(len_wvref2), 'mismatch in the len_wvref inputs'
        else:
            raise ValueError(f'undefined len_wvref={len_wvref1}')

        # Sanity checking `refr_wvchmref`: The chemical refraction index values for optical property calculations.
        refr_wvchmref1 = pp_config.get('hparams/aerocst/refr_wvchmref', None)
        chem2refr = get_subdict(pp_config, 'hparams/aerocst/chem2refr', pop=True)
        if len(chem2refr) > 0:
            assert chem_species is not None
            refr_wvchmrefnp2 = np.array([
                [complex(chem2refr[f'{wvid}/{chem}']) for chem in chem_species] 
                for wvid in wave_idssrtd]).astype(np.complex128)
            assert refr_wvchmrefnp2.shape == (n_waveref, n_chem)
            refr_wvchmref2 = torch.from_numpy(refr_wvchmrefnp2).to(device=device, dtype=cdtype)
            assert refr_wvchmref2.shape == (n_waveref, n_chem)
        else:
            refr_wvchmref2 = None

        if refr_wvchmref1 is None:
            assert refr_wvchmref2 is not None, 'either chem2refr or refr_wvchmref must be specified'
            refr_wvchmref = refr_wvchmref2
            assert refr_wvchmref.shape == (n_waveref, n_chem)
        elif torch.is_tensor(refr_wvchmref1):
            refr_wvchmref = refr_wvchmref1.to(device=device, dtype=cdtype)
            assert refr_wvchmref.shape == (n_waveref, n_chem)
            assert (refr_wvchmref2 is None) or refr_wvchmref.allclose(refr_wvchmref2), 'mismatch in the refr_wvchmref inputs'
        elif isinstance(refr_wvchmref1, np.ndarray):
            refr_wvchmref = torch.from_numpy(refr_wvchmref1).to(device=device, dtype=cdtype)
            assert refr_wvchmref.shape == (n_waveref, n_chem)
            assert (refr_wvchmref2 is None) or refr_wvchmref.allclose(refr_wvchmref2), 'mismatch in the refr_wvchmref inputs'
        else:
            raise ValueError(f'undefined refr_wvchmref={refr_wvchmref1}')
        
        # Computing the interpolated optical refraction constants for the arbitrary set of wave-lengths
        wave_length = pp_config.get('hparams/aeroopt/wave_length', None)
        if need_optcalc:
            assert wave_length is not None
            assert len_wvref.shape == (n_waveref,)
            assert refr_wvchmref.shape == (n_waveref, n_chem)

            refr_wvchm, len_wv, n_wave = interp_refr(refr_wvchmref, len_wvref, wave_length, 
                chem_species=chem_species)
            
            assert len_wv.shape == (n_wave,)
            assert refr_wvchm.shape == (n_wave, n_chem)
        else:
            n_wave, len_wv, refr_wvchm = None, None, None

        # Sanity checking that everything necessary exists
        n_epshist = pp_config.get('hparams/aeroccn/n_epshist', None)
        n_tmprtr = pp_config.get('hparams/aeroinp/n_tmprtr', None)
        n_bins = pp_config.get('hparams/aerohst/n_bins', None)
        
        if need_hstcalc or need_ccncalc or need_optcalc or need_inpcalc:
            assert rho_chm is not None
        else:
            rho_chm = None
        
        if need_hstcalc:
            assert n_bins is not None
        else:
            n_bins = None
        
        if need_ccncalc:
            assert kappa_chm is not None
            assert n_epshist is not None
        else:
            n_epshist = None
            kappa_chm = None
        
        if need_optcalc:
            assert refr_wvchm is not None
            assert len_wv is not None
            assert n_wave is not None
        else:
            refr_wvchm = None
            len_wv = None
            n_wave = None
        
        if need_inpcalc:
            assert n_tmprtr is not None
        else:
            n_tmprtr = None
        
        pp_config['hparams/aerocst/chem_species'] = chem_species
        pp_config['hparams/aerocst/n_chem'] = n_chem
        pp_config['hparams/aerocst/n_part'] = n_part
        pp_config['hparams/aerocst/rho_chm'] = rho_chm
        pp_config['hparams/aerocst/kappa_chm'] = kappa_chm
        pp_config['hparams/aerocst/n_waveref'] = n_waveref
        pp_config['hparams/aerocst/len_wvref'] = len_wvref
        pp_config['hparams/aerocst/refr_wvchmref'] = refr_wvchmref
        pp_config['hparams/aerocst/n_wave'] = n_wave
        pp_config['hparams/aerocst/len_wv'] = len_wv
        pp_config['hparams/aerocst/refr_wvchm'] = refr_wvchm
        pp_config['hparams/aeroopt/wave_length'] = wave_length
        pp_config['hparams/aeroccn/n_epshist'] = n_epshist
        pp_config['hparams/aeroinp/n_tmprtr'] = n_tmprtr
        pp_config['hparams/aerohst/n_bins'] = n_bins
        pp_config['hparams/output/type'] = diag_types

    @staticmethod
    def get_diagtypes(measures):
        all_ccnmeasures = {'ccn_cdf'}
        all_optmeasures = {'qs_pop', 'qscs_prt', 'qs_prt', 'qa_pop', 'qacs_prt', 'qa_prt'}
        all_inpmeasures = {'frznfrac_tmp', 'logfrznfrac_tmp', 'surf_oin', 'surf_bc'}
        all_hstmeasures = {'m_chmprthst', 'n_prthst', 'm_prthst'}
        all_inputkeys = {'m_chmprthst', 'n_prthst', 'm_prtchm', 'n_prt'}
        all_measures = all_inputkeys | all_ccnmeasures | all_optmeasures | all_inpmeasures | all_hstmeasures
        assert set(measures).issubset(all_measures), dedent(f'''
            The following aerosol measures are undefined: 
                {set(measures) - all_measures}''')
        need_ccncalc = len(measures & all_ccnmeasures) > 0
        need_optcalc = len(measures & all_optmeasures) > 0
        need_inpcalc = len(measures & all_inpmeasures) > 0
        need_hstcalc = len(measures & all_hstmeasures) > 0

        diag_types = []
        if need_hstcalc:
            diag_types.append('hst')
        if need_ccncalc:
            diag_types.append('ccn')
        if need_optcalc:
            diag_types.append('opt')
        if need_inpcalc:
            diag_types.append('inp')
            
        return diag_types

    def forward(self, inputs_mb: dict, full: bool = False):
        """
        Performs a forward transformation on the pre-processing tree.

        Args:
            inputs_mb (dict): The mapping of the input layers to mini-batch
                arrays. The arrays can be either `np.ndarray` or
                `torch.tensor` types.

                Example:
                    ```
                    inputs_mb = {
                        'm_chmprthst': np.array(n_seed, n_mb, n_bins, n_chem),
                        'n_prthst': np.array(n_seed, n_mb, n_bins, 1)
                    }
                    ```

            full (bool): Whether to return the computed mini-batch arrays for all
                the layers (true), or only returning the mini-batch arrays for
                the output layers (false).

        Returns:

            layers_mb: A dictionary of layer names (str) to values (torch.tensor).
        """
        pp_config = dict(self.pp_config)
        tch_device, tch_dtype, tch_cdtype = self.tch_device, self.tch_dtype, self.tch_cdtype

        ###################################################
        ############ Getting the HP Variables #############
        ###################################################
        # The input data type; hst for histograms, and prt for particle data
        input_type = pp_config.pop('hparams/input/type')
        assert input_type in ('hst', 'prt')

        measures = set(pp_config.pop('hparams/measures'))

        # Sanity checking the aero constants, and interpolating the necessary 
        # wave-lengths for the refraction constants.
        self.vrfy_aerocsts(pp_config, measures, tch_device, tch_dtype, tch_cdtype)

        # Extracting the CCN keyword arguments
        eps_histmin = pp_config.pop('hparams/aeroccn/eps_histmin')
        eps_histmax = pp_config.pop('hparams/aeroccn/eps_histmax')
        n_epshist = pp_config.pop('hparams/aeroccn/n_epshist')
        tmprtr_sclr = pp_config.pop('hparams/aeroccn/tmprtr_sclr')
        void_kappa = pp_config.pop('hparams/aeroccn/void_kappa')
        n_rfiters = pp_config.pop('hparams/aeroccn/n_rfiters')
        e_rfiters = pp_config.pop('hparams/aeroccn/e_rfiters')
        strict = pp_config.pop('hparams/aeroccn/strict')
        bs_ccn = pp_config.pop('hparams/aeroccn/eval/bs')
        inptyp_ccn = pp_config.pop('hparams/aeroccn/input/type')
        inptyp_ccn = input_type if inptyp_ccn is None else inptyp_ccn

        # Extracting the optical keyword arguments
        wave_length = pp_config.pop('hparams/aeroopt/wave_length')
        n_max = pp_config.pop('hparams/aeroopt/n_max')
        void_refr = complex(pp_config.pop('hparams/aeroopt/void_refr'))
        trnsps_opt = pp_config.pop('hparams/aeroopt/trnsps')
        bs_opt = pp_config.pop('hparams/aeroopt/eval/bs')
        inptyp_opt = pp_config.pop('hparams/aeroopt/input/type')
        inptyp_opt = input_type if inptyp_opt is None else inptyp_opt
        
        # Extracting the INP keyword arguments
        tmprtr_inpmin = pp_config.pop('hparams/aeroinp/tmprtr_inpmin')
        tmprtr_inpmax = pp_config.pop('hparams/aeroinp/tmprtr_inpmax')
        n_tmprtr = pp_config.pop('hparams/aeroinp/n_tmprtr')
        trnsps_inp = pp_config.pop('hparams/aeroinp/trnsps')
        bs_inp = pp_config.pop('hparams/aeroinp/eval/bs')
        inptyp_inp = pp_config.pop('hparams/aeroinp/input/type')
        inptyp_inp = input_type if inptyp_inp is None else inptyp_inp

        # Extracting the diameter histogram keyword arguments
        n_bins = pp_config.pop('hparams/aerohst/n_bins')
        diam_low = pp_config.pop('hparams/aerohst/diam_low')
        diam_high = pp_config.pop('hparams/aerohst/diam_high')
        diam_logbase = pp_config.pop('hparams/aerohst/diam_logbase')
        bs_hst = pp_config.pop('hparams/aerohst/eval/bs')

        # Extracting the physical aerosol constants
        chem_species = pp_config.pop('hparams/aerocst/chem_species')
        n_chem = pp_config.pop('hparams/aerocst/n_chem')
        n_part = pp_config.pop('hparams/aerocst/n_part')
        rho_chm = pp_config.pop('hparams/aerocst/rho_chm')
        kappa_chm = pp_config.pop('hparams/aerocst/kappa_chm')
        n_waveref = pp_config.pop('hparams/aerocst/n_waveref')
        len_wvref = pp_config.pop('hparams/aerocst/len_wvref')
        refr_wvchmref = pp_config.pop('hparams/aerocst/refr_wvchmref')
        n_wave = pp_config.pop('hparams/aerocst/n_wave')
        len_wv = pp_config.pop('hparams/aerocst/len_wv')
        refr_wvchm = pp_config.pop('hparams/aerocst/refr_wvchm')
        diag_types = pp_config.pop('hparams/output/type')

        assert len(pp_config) == 0, f'unused options: {pp_config}'

        ###################################################
        ########## Getting the Input Variables ############
        ###################################################
        if input_type == 'hst':
            m_chmprthst = inputs_mb['m_chmprthst']
            n_seeds, n_mb = m_chmprthst.shape[:2]
            assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, n_part)

            n_prthst = inputs_mb['n_prthst']
            assert n_prthst.shape == (n_seeds, n_mb, 1, n_part)

            m_input, n_input = m_chmprthst, n_prthst
        elif input_type == 'prt':
            m_prtchm = inputs_mb['m_prtchm']
            n_seeds, n_mb = m_prtchm.shape[:2]
            assert m_prtchm.shape == (n_seeds, n_mb, n_part, n_chem)
            
            n_prt = inputs_mb['n_prt']
            assert n_prt.shape == (n_seeds, n_mb, n_part, 1)

            m_input, n_input = m_prtchm, n_prt
        else:
            raise ValueError(f'undefined input/type={input_type}')

        ###################################################
        ### Getting the Diameter Histogram Measurements ###
        ###################################################
        hst_out = self.get_hstmeasures(m_input, n_input, rho_chm, diam_low, diam_high, 
            diam_logbase, n_seeds, n_mb, n_part, n_chem, n_bins, bs_hst, measures, input_type)

        ###################################################
        ########## Getting the CCN Measurements ###########
        ###################################################
        if (inptyp_ccn == input_type):
            m_inputccn, n_inputccn = m_input, n_input 
        elif (inptyp_ccn == 'hst') and (input_type == 'prt'):
            assert 'm_chmprthst' in hst_out, dedent('''
                you need to specify histogram measurements if the ccn 
                input is going to be histogram data.''')
            m_inputccn, n_inputccn = hst_out['m_chmprthst'], hst_out['n_prthst']
        elif (inptyp_ccn == 'prt') and (input_type == 'hst'):
            raise ValueError('Cannot pass particles to CCN with histogram inputs!')
        else:
            raise ValueError('This should have not been possible')

        ccn_out = self.get_ccnmeasures(m_inputccn, n_inputccn, eps_histmin, eps_histmax, 
            n_epshist, tmprtr_sclr, rho_chm, kappa_chm, chem_species, void_kappa, 
            n_rfiters, e_rfiters, strict, n_seeds, n_mb, n_chem, n_part, bs_ccn, 
            tch_device, tch_dtype, measures, inptyp_ccn)

        ###################################################
        ######## Getting the Optical Measurements #########
        ###################################################
        if (inptyp_opt == input_type):
            m_inputopt, n_inputopt = m_input, n_input 
        elif (inptyp_opt == 'hst') and (input_type == 'prt'):
            assert 'm_chmprthst' in hst_out, dedent('''
                you need to specify histogram measurements if the opt 
                input is going to be histogram data.''')
            m_inputopt, n_inputopt = hst_out['m_chmprthst'], hst_out['n_prthst']
        elif (inptyp_opt == 'prt') and (input_type == 'hst'):
            raise ValueError('Cannot pass particles to opt with histogram inputs!')
        else:
            raise ValueError('This should have not been possible')

        opt_out = self.get_optmeasures(m_inputopt, n_inputopt, rho_chm, refr_wvchm, 
            len_wv, n_max, void_refr, chem_species, n_seeds, n_mb, n_chem, 
            n_part, n_wave, bs_opt, trnsps_opt, tch_device, tch_dtype, tch_cdtype, 
            measures, inptyp_opt)

        ###################################################
        ########## Getting the INP Measurements ###########
        ###################################################
        if (inptyp_inp == input_type):
            m_inputinp, n_inputinp = m_input, n_input 
        elif (inptyp_inp == 'hst') and (input_type == 'prt'):
            assert 'm_chmprthst' in hst_out, dedent('''
                you need to specify histogram measurements if the inp 
                input is going to be histogram data.''')
            m_inputinp, n_inputinp = hst_out['m_chmprthst'], hst_out['n_prthst']
        elif (inptyp_inp == 'prt') and (input_type == 'hst'):
            raise ValueError('Cannot pass particles to inp with histogram inputs!')
        else:
            raise ValueError('This should have not been possible')

        inp_out = self.get_inpmeasures(m_inputinp, n_inputinp, tmprtr_inpmin, 
            tmprtr_inpmax, n_tmprtr, rho_chm, chem_species, n_seeds, n_mb, n_chem, 
            n_part, bs_inp, trnsps_inp, measures, inptyp_inp)
        
        shaper = None
        layers_mb = dict()
        if (input_type == 'hst') and (full or ('m_chmprthst' in measures)):
            layers_mb['m_chmprthst'] = m_chmprthst
        if (input_type == 'hst') and (full or ('n_prthst' in measures)):
            layers_mb['n_prthst'] = n_prthst
        if (input_type == 'prt') and (full or ('m_prtchm' in measures)):
            layers_mb['m_prtchm'] = m_prtchm
        if (input_type == 'prt') and (full or ('n_prt' in measures)):
            layers_mb['n_prt'] = n_prt
        layers_mb.update(ccn_out)
        layers_mb.update(opt_out)
        layers_mb.update(inp_out)
        layers_mb.update(hst_out)

        return layers_mb, shaper
    
    @staticmethod
    @torch.no_grad()
    def get_hstmeasures(m_input, n_input, rho_chm, diam_low, diam_high, diam_logbase,
        n_seeds, n_mb, n_part, n_chem, n_bins, bs_hst, measures, input_type):

        hst_vars = {'m_chmprthst', 'm_prthst', 'n_prthst'}
        if not any(msr in hst_vars for msr in measures):
            return dict()
        
        hst_outlsts = defaultdict(list)
        n_iterhst = math.ceil(n_mb / bs_hst)
        for i_iterhst in range(n_iterhst):
            i1 = bs_hst * i_iterhst
            i2 = min(n_mb, i1 + bs_hst)
            n_mbhst = i2 - i1
            if input_type == 'hst':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbhst, n_chem, n_part)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbhst, 1, n_part)
            elif input_type == 'prt':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbhst, n_part, n_chem)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbhst, n_part, 1)
            else:
                raise ValueError(f'undefined input/type={input_type}')

            if input_type == 'hst':
                m_chmprthst_tmp, n_prthst_tmp = clean_hist(m_inputmb, n_inputmb, n_seeds, 
                    n_mbhst, n_chem, n_bins)
                
                assert m_chmprthst_tmp.shape == (n_seeds, n_mbhst, n_chem, n_bins)
                assert n_prthst_tmp.shape == (n_seeds, n_mbhst, 1, n_bins)
                
                m_prthst_tmp = m_chmprthst_tmp.sum(dim=-2, keepdims=True)
                assert m_prthst_tmp.shape == (n_seeds, n_mbhst, 1, n_bins)
            elif input_type == 'prt':
                hst_info = get_diamhst(m_inputmb, n_inputmb, rho_chm, n_seeds, n_mbhst, n_part, 
                    n_chem, n_bins, diam_low, diam_high, diam_logbase, lib='torch')

                m_chmprthst_tmp = hst_info['m_chmprthst']
                assert m_chmprthst_tmp.shape == (n_seeds, n_mbhst, n_chem, n_bins)
                m_prthst_tmp = hst_info['m_prthst']
                assert m_prthst_tmp.shape == (n_seeds, n_mbhst, 1, n_bins)
                n_prthst_tmp = hst_info['n_prthst']
                assert n_prthst_tmp.shape == (n_seeds, n_mbhst, 1, n_bins)
            else:
                raise ValueError(f'undefined input/type={input_type}')

            if 'm_chmprthst' in measures:
                hst_outlsts['m_chmprthst'].append(m_chmprthst_tmp)
            if 'm_prthst' in measures:
                hst_outlsts['m_prthst'].append(m_prthst_tmp)
            if 'n_prthst' in measures:
                hst_outlsts['n_prthst'].append(n_prthst_tmp)
        
        hst_out = {key: torch.cat(val, dim=1) for key, val in hst_outlsts.items()}
        return hst_out

    @staticmethod
    @torch.no_grad()
    def get_ccnmeasures(m_input, n_input, eps_histmin, eps_histmax, 
        n_epshist, tmprtr_sclr, rho_chm, kappa_chm, chem_species, void_kappa, 
        n_rfiters, e_rfiters, strict, n_seeds, n_mb, n_chem, n_part, 
        bs_ccn, tch_device, tch_dtype, measures, input_type):
        
        ccn_vars =  {'ccn_cdf'}
        if not any(msr in ccn_vars for msr in measures):
            return dict()

        ccn_outlsts = defaultdict(list)
        n_iterccn = math.ceil(n_mb / bs_ccn)
        for i_iterccn in range(n_iterccn):
            i1 = bs_ccn * i_iterccn
            i2 = min(n_mb, i1 + bs_ccn)
            n_mbccn = i2 - i1

            if input_type == 'hst':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbccn, n_chem, n_part)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbccn, 1, n_part)
            elif input_type == 'prt':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbccn, n_part, n_chem)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbccn, n_part, 1)
            else:
                raise ValueError(f'undefined input/type={input_type}')

            ccn_info = get_apprxccncdf(m_inputmb, n_inputmb, 
                eps_histmin, eps_histmax, n_epshist, tmprtr_sclr, rho_chm, kappa_chm, 
                chem_species, void_kappa, n_rfiters, e_rfiters, n_seeds, n_mbccn, n_chem, 
                n_part, tch_device, tch_dtype, input_type, strict=strict)

            n_partset = ccn_info['n_partset']
            assert n_partset == (n_seeds * n_mbccn)
            assert ccn_info['n_part'] == n_part

            eps_histbins = ccn_info['eps_histbins']
            assert eps_histbins.shape == (n_epshist,)
            ccn_cdf = ccn_info['ccn_cdf']
            assert ccn_cdf.shape == (n_partset, n_epshist)

            ccn_cdf2 = ccn_cdf.reshape(n_seeds, n_mbccn, 1, n_epshist)
            assert ccn_cdf2.shape == (n_seeds, n_mbccn, 1, n_epshist)

            if 'ccn_cdf' in measures:
                ccn_outlsts['ccn_cdf'].append(ccn_cdf2)
        
        ccn_out = {key: torch.cat(val, dim=1) for key, val in ccn_outlsts.items()}
    
        return ccn_out

    @staticmethod
    @torch.no_grad()
    def get_optmeasures(m_input, n_input, rho_chm, refr_wvchm, len_wv,
        n_max, void_refr, chem_species, n_seeds, n_mb, n_chem, n_part, n_wave, 
        bs_opt, trnsps_opt, tch_device, tch_dtype, tch_cdtype, measures, input_type):

        inp_vars = {'qs_prt', 'qscs_prt', 'qs_pop', 'qa_prt', 'qacs_prt', 'qa_pop'}
        if not any(msr in inp_vars for msr in measures):
            return dict()

        opt_outlsts = defaultdict(list)
        n_iteropt = math.ceil(n_mb / bs_opt)
        for i_iteropt in range(n_iteropt):
            i1 = bs_opt * i_iteropt
            i2 = min(n_mb, i1 + bs_opt)
            n_mbopt = i2 - i1

            if input_type == 'hst':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbopt, n_chem, n_part)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbopt, 1, n_part)
            elif input_type == 'prt':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbopt, n_part, n_chem)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbopt, n_part, 1)
            else:
                raise ValueError(f'undefined input/type={input_type}')

            opt_info = get_bulkoptical(m_inputmb, n_inputmb, rho_chm, refr_wvchm, len_wv,
                n_max, void_refr, chem_species, n_seeds, n_mbopt, n_chem, n_part, n_wave,
                tch_device, tch_dtype, tch_cdtype, input_type)

            qs_prt = opt_info['qs_prt']
            assert qs_prt.shape == (n_seeds, n_mbopt, n_wave, n_part)
            qscs_prt = opt_info['qscs_prt']
            assert qscs_prt.shape == (n_seeds, n_mbopt, n_wave, n_part)
            qs_pop = opt_info['qs_pop'].unsqueeze(-1)
            assert qs_pop.shape == (n_seeds, n_mbopt, n_wave, 1)
            qa_prt = opt_info['qa_prt']
            assert qa_prt.shape == (n_seeds, n_mbopt, n_wave, n_part)
            qacs_prt = opt_info['qacs_prt']
            assert qacs_prt.shape == (n_seeds, n_mbopt, n_wave, n_part)
            qa_pop = opt_info['qa_pop'].unsqueeze(-1)
            assert qa_pop.shape == (n_seeds, n_mbopt, n_wave, 1)
            
            if 'qs_prt' in measures:
                opt_outlsts['qs_prt'].append(qs_prt.transpose(-1, -2) if trnsps_opt else qs_prt)
            if 'qscs_prt' in measures:
                opt_outlsts['qscs_prt'].append(qscs_prt.transpose(-1, -2) if trnsps_opt else qscs_prt)
            if 'qs_pop' in measures:
                opt_outlsts['qs_pop'].append(qs_pop.transpose(-1, -2) if trnsps_opt else qs_pop)
            if 'qa_prt' in measures:
                opt_outlsts['qa_prt'].append(qa_prt.transpose(-1, -2) if trnsps_opt else qa_prt)
            if 'qacs_prt' in measures:
                opt_outlsts['qacs_prt'].append(qacs_prt.transpose(-1, -2) if trnsps_opt else qacs_prt)
            if 'qa_pop' in measures:
                opt_outlsts['qa_pop'].append(qa_pop.transpose(-1, -2) if trnsps_opt else qa_pop)
        
        opt_out = {key: torch.cat(val, dim=1) for key, val in opt_outlsts.items()}

        return opt_out

    @staticmethod
    @torch.no_grad()
    def get_inpmeasures(m_input, n_input, tmprtr_inpmin, tmprtr_inpmax, 
        n_tmprtr, rho_chm, chem_species, n_seeds, n_mb, n_chem, n_part,
        bs_inp, trnsps_inp, measures, input_type):

        inp_vars = {'frznfrac_tmp', 'logfrznfrac_tmp', 'surf_oin', 'surf_bc'}
        if not any(msr in inp_vars for msr in measures):
            return dict()
        
        inp_outlsts = defaultdict(list)
        n_iterinp = math.ceil(n_mb / bs_inp)
        for i_iterinp in range(n_iterinp):
            i1 = bs_inp * i_iterinp
            i2 = min(n_mb, i1 + bs_inp)
            n_mbinp = i2 - i1

            if input_type == 'hst':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbinp, n_chem, n_part)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbinp, 1, n_part)
            elif input_type == 'prt':
                m_inputmb = m_input[:, i1: i2, :, :]
                assert m_inputmb.shape == (n_seeds, n_mbinp, n_part, n_chem)
                n_inputmb = n_input[:, i1: i2, :, :]
                assert n_inputmb.shape == (n_seeds, n_mbinp, n_part, 1)
            else:
                raise ValueError(f'undefined input/type={input_type}')

            inp_info = get_bulkfrznfrac(m_inputmb, n_inputmb, tmprtr_inpmin, 
                tmprtr_inpmax, n_tmprtr, rho_chm, chem_species, n_seeds, 
                n_mbinp, n_chem, n_part, input_type)

            frznfrac_tmp = inp_info['frznfrac_tmp']
            assert frznfrac_tmp.shape == (n_seeds, n_mbinp, n_tmprtr)
            logfrznfrac_tmp = inp_info['logfrznfrac_tmp']
            assert logfrznfrac_tmp.shape == (n_seeds, n_mbinp, n_tmprtr)
            temprtr_bins = inp_info['temprtr_bins']
            assert temprtr_bins.shape == (n_tmprtr,)
            surf_oin = inp_info['surf_oin']
            assert surf_oin.shape == (n_seeds, n_mbinp, n_part)
            surf_bc = inp_info['surf_bc']
            assert surf_bc.shape == (n_seeds, n_mbinp, n_part)

            frznfrac_tmp2 = frznfrac_tmp.reshape(n_seeds, n_mbinp, 1, n_tmprtr)
            assert frznfrac_tmp2.shape == (n_seeds, n_mbinp, 1, n_tmprtr)
            logfrznfrac_tmp2 = logfrznfrac_tmp.reshape(n_seeds, n_mbinp, 1, n_tmprtr)
            assert logfrznfrac_tmp2.shape == (n_seeds, n_mbinp, 1, n_tmprtr)
            surf_oin2 = surf_oin.reshape(n_seeds, n_mbinp, 1, n_part)
            assert surf_oin2.shape == (n_seeds, n_mbinp, 1, n_part)
            surf_bc2 = surf_bc.reshape(n_seeds, n_mbinp, 1, n_part)
            assert surf_bc2.shape == (n_seeds, n_mbinp, 1, n_part)

            if 'frznfrac_tmp' in measures:
                inp_outlsts['frznfrac_tmp'].append(frznfrac_tmp2)
            if 'logfrznfrac_tmp' in measures:
                inp_outlsts['logfrznfrac_tmp'].append(logfrznfrac_tmp2)
            if 'surf_oin' in measures:
                inp_outlsts['surf_oin'].append(surf_oin2.transpose(-1, -2) if trnsps_inp else surf_oin2)
            if 'surf_bc' in measures:
                inp_outlsts['surf_bc'].append(surf_bc2.transpose(-1, -2) if trnsps_inp else surf_bc2)
        
        inp_out = {key: torch.cat(val, dim=1) for key, val in inp_outlsts.items()}
        return inp_out

    def inverse(self, outputs_mb: dict, shaper: any = None, strict: bool = True, full: bool = False):
        """
        Performs an inverse transformation on the pre-processing tree.

        Args:
            outputs_mb (dict): The mapping of the output layers to mini-batch
                arrays. The arrays can be either `np.ndarray` or
                `torch.tensor` types.

                Example:
                    ```
                    inputs_mb = {
                        'm_chmprthst': np.array(n_seed, n_mb, n_bins, n_chem),
                        'n_prthst': np.array(n_seed, n_mb, n_bins),
                        'm_prthst': np.array(n_seed, n_mb, n_bins)}
                    ```

            shaper (ShapeChecker): The shape checker object holding all the layers'
                shape information such as the shape variables and layer shapes. This
                method will only use it to evaluate the expected shape formulas, and
                will not rely on the forward layer shapes memory of the `shaper` object.

            strict (bool): Whether to enforce the one-to-one properties of the inversion
                map. If the pre-processing transformation is not one-to-one, having this
                option equal to `True` will lead to `ValueError` exceptions.

            full (bool): Whether to return the computed mini-batch arrays for all
                the layers (true), or only returning the mini-batch arrays for
                the output layers (false).

        Returns:
            layers_mb (dict): A mapping of layer names (str) to values (torch.tensor).
        """

        raise ValueError('The aerosol measures cannot define an inverse computations')

    @torch.no_grad()
    def infer(self, data: dict, trn_spltidxs: torch.tensor, n_seeds: int, 
        n_mb: int, mb_order: int = 1, hash_data: any = None, pre_pp: list | tuple | None = None, 
        cache_path: str | None = None):
        """
        This method infers the shift and scale parameters from the
        training portion of the data.

        Since there are no data parameters to be learned, we will not 
        do anything here!
        """

        self.n_seeds = n_seeds

    def get_dims(self, data_dims: dict, types: list, n_dims: int = None, n_seeds: int = 0,
                 n_mb: int = 0, mb_order: int = 1):
        """
        Returns a dictionary of layer names to numerical dimension tuples. This
        method omits the preceding `(n_seeds, n_mb)` part of the shape tuple.
        This function also can check that the number of dimensions is fixed to
        be `n_dims`.

        Args:
            data_dims (dict): A dictionary mapping the input variable names
                (str) to non-batch (i.e., not the `n_seeds` and `n_mb`) shapes.

                Example:
                    ```
                    data_dims = {
                        'n_prthst': (1, n_bins),
                        'm_chmprthst': (n_chem, n_bins)
                    }
                    ```
            types (str | list): Either a string or a list/tuple of strings.

                Example:

                    types = 'input'

                    types = ['input', 'output', 'log']

            n_dims (int): The number of dimensions. Only used for assertion checks.

            n_seeds (int): The number of independent runs with different RNG seeds.
                For the purposes of this function, it can be a meaningless value
                such as zero, since it will be dropped from the resulting shapes
                anyway.

            n_mb (int): The mini-batch size. For the purposes of this function,
                it can be a meaningless value such as zero, since it will be
                dropped from the resulting shapes anyway.

            mb_order (int): The mini-batch dimension index. should be either 0 or 1.
                This determines where `n_mb` is located in the input's shape:

                ```
                if mb_order == 0:
                    assert x.shape == (n_mb, n_seeds, n_channel, *x_spatial_dims)
                elif mb_order == 1:
                    assert x.shape == (n_seeds, n_mb, n_channel, *x_spatial_dims)
                else:
                    raise NotImplementedError
                ```

        Returns:

            dims (dict): a dictionary mapping the layer names to the
                evaluated numerical dimension tuples.

                Example:

                    dims = {'ccn_cdf':  (1, n_epshist), 'qs_pop':  (1, 1), 
                        'qs_prt':  (1, n_bins), 'qa_pop':  (1, 1), 
                        'qa_prt':  (1, n_bins), 'frznfrac_tmp':  (1, n_tmprtr), 
                        'logfrznfrac_tmp':  (1, n_tmprtr), 'surf_oin':  (1, n_bins), 
                        'surf_bc':  (1, n_bins)}
        """
        measures = self.pp_config['hparams/measures']
        n_chem = self.pp_config['hparams/aerocst/n_chem']
        n_part = self.pp_config['hparams/aerocst/n_part']
        n_wave = self.pp_config['hparams/aerocst/n_wave']
        n_epshist = self.pp_config['hparams/aeroccn/n_epshist']
        n_tmprtr = self.pp_config['hparams/aeroinp/n_tmprtr']
        input_type = self.pp_config['hparams/input/type']
        n_bins = self.pp_config['hparams/aerohst/n_bins']
        trnsps_opt = self.pp_config['hparams/aeroopt/trnsps']
        assert input_type in ('hst', 'prt')

        if (n_bins is None) and (input_type == 'hst'):
            n_bins = n_part

        if 'n_chem' in data_dims:
            assert data_dims['n_chem'] == n_chem
        if 'n_bins' in data_dims:
            assert data_dims['n_bins'] == n_part
        if 'n_part' in data_dims:
            assert data_dims['n_part'] == n_part
        if n_dims is not None:
            assert n_dims == 2

        all_dims = {
            'm_chmprthst': (n_chem, n_bins), 'm_prthst': (1, n_bins), 'n_prthst': (1, n_bins), 
            'm_prtchm': (n_part, n_chem), 'n_prt': (n_part, 1), 'ccn_cdf':  (1, n_epshist), 
            'qs_prt':  (n_wave, n_part), 'qscs_prt':  (n_wave, n_part), 'qs_pop':  (n_wave, 1), 
            'qa_prt':  (n_wave, n_part), 'qacs_prt':  (n_wave, n_part), 'qa_pop':  (n_wave, 1), 
            'frznfrac_tmp':  (1, n_tmprtr), 'logfrznfrac_tmp':  (1, n_tmprtr), 
            'surf_oin':  (1, n_part), 'surf_bc':  (1, n_part)}

        if trnsps_opt:
            for key in list(all_dims):
                if key in ('qs_prt', 'qscs_prt', 'qs_pop', 'qa_prt', 
                    'qacs_prt', 'qa_pop', 'surf_oin', 'surf_bc'):
                    all_dims[key] = all_dims[key][::-1]

        types = {types} if isinstance(types, str) else set(types)
        assert types.issubset({'input', 'output', 'log'})

        dims = dict()
        if ('input' in types) and (input_type == 'hst'):
            dims['m_chmprthst'] = (n_chem, n_part)
            dims['n_prthst'] = (1, n_part)
        if ('input' in types) and (input_type == 'prt'):
            dims['m_prtchm'] = (n_part, n_chem)
            dims['n_prt'] = (n_part, 1)
        if 'output' in types:
            for u_node in measures:
                assert u_node in all_dims, f'dim spec for {u_node} undefined'
                dims[u_node] = all_dims[u_node]
        if 'log' in types:
            for u_node in all_dims:
                dims[u_node] = all_dims[u_node]

        return dims
    
    def get_shapevars(self):
        n_chem = self.pp_config['hparams/aerocst/n_chem']
        n_part = self.pp_config['hparams/aerocst/n_part']
        n_bins = self.pp_config['hparams/aerohst/n_bins']
        n_wave = self.pp_config['hparams/aerocst/n_wave']
        n_epshist = self.pp_config['hparams/aeroccn/n_epshist']
        n_tmprtr = self.pp_config['hparams/aeroinp/n_tmprtr']
        shapevars = dict(n_chem=n_chem, n_part=n_part, n_bins=n_bins, 
            n_wave=n_wave, n_epshist=n_epshist, n_tmprtr=n_tmprtr)
        return shapevars

    def get_shaper(self, inputs_shp: dict):
        """
        Runs a simulated forward loop without any calculations and tries to infer the
        layer tensor shapes using first principles about the operations. It will return
        a `ShapeChcker` object that

        Args:
            inputs_shp (dict): The mapping of the input names (str) to mini-batch
                shapes (tuple/list).

                Example:
                    ```
                    inputs_mb = {
                        'm_chmprthst': (n_seed, n_mb, k_bins, n_chem),
                        'n_prthst': (n_seed, n_mb, k_bins),
                        'm_prthst': (n_seed, n_mb, k_bins)
                    }
                    ```
        Returns:
            shaper (ShapeChecker): The shape checker object holding all the layers'
                shape information such as the shape variables and layer output shapes.

        """

        return None

    def save(self):
        """
        Returns the full state of the instance such that it would be
        serializable and completely able to re-create the object on
        its own.

        Returns:
            state (dict): A hierarchical dictionary with all the
                object's information and parameters including the
                layers graph, the parameters, and the hyper-parameter
                definitions.
        """

        pp_config = dict(self.pp_config)
        input_type = pp_config['hparams/input/type']
        n_seeds = self.n_seeds
        input_names = {'hst': ['m_chmprthst', 'n_prthst'], 
            'prt': ['m_prtchm', 'n_prt']}[input_type]

        pp_dims = self.get_dims(dict(), ['output'])
        pp_layers = {u_node: {'type': u_node, 
                'input': input_names, 
                'shape': ['n_seeds', 'n_mb', *u_dims]}
            for u_node, u_dims in pp_dims.items()}

        pp_params = dict()
        for p_name in ('rho_chm', 'kappa_chm', 'refr_wvchmref', 'len_wvref', 'refr_wvchm', 'len_wv'):
            p_var = pp_config.pop(f'hparams/aerocst/{p_name}', None)
            if p_var is not None:
                pp_params[p_name] = p_var[None].expand(n_seeds, *p_var.shape)
        
        assert not any(torch.is_tensor(val) for key, val in pp_config.items())

        state_hie = {'layers': pp_layers, 'params': pp_params, 'hpdefs': pp_config}
        state = deep2hie(state_hie, maxdepth=1, sep='/')
        return state

    def load(self, state: dict):
        """
        Loads the state dictionary into self

        Args:
            state (dict): The hierarchical dictionary with all the
                object's information and parameters including the
                layers graph, the parameters, and the hyper-parameter
                definitions.

        Returns:
            pp (PreProcessor): A new instance of the PreProcessor object
                with all the state variables embedded in it.
        """
        state_hie = hie2deep(deep2hie(state), maxdepth=1, sep='/')
        pp_config = state_hie['hpdefs']
        for p_name, p_val in state_hie['params'].items():
            assert (p_val == p_val[:1, :]).all(), dedent(f'''
                "{p_name}" is assumed to be the same for all seeds.''')
            pp_config[f'hparams/aerocst/{p_name}'] = p_val[0]
        
        self.pp_config.clear()
        self.pp_config.update(pp_config)

    @classmethod
    def make(cls, state: dict, device: torch.device, dtype: torch.dtype, 
        cdtype: torch.dtype):
        """
        Creates a new instance of the PreProcessor Object using a state
        dictionary.

        Args:
            state (dict): The hierarchical dictionary with all the
                object's information and parameters including the
                layers graph, the parameters, and the hyper-parameter
                definitions.

            device (torch.device | str): The pytorch device of the tensors.

            dtype (torch.dtype | str): The pytorch dtype of the tensors.

        Returns:
            pp (PreProcessor): A new instance of the PreProcessor object
                with all the state variables embedded in it.
        """
        pp = cls(None, device, dtype, cdtype)
        pp.load(state)
        return pp


if __name__ == '__main__':
    import netCDF4
    from os.path import exists
    from partnn.io_cfg import PROJPATH

    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tch_device = torch.device(device_name)
    tch_dtype = torch.float64

    print('Performing unit tests for the CRH calculation functions:')

    test_datapath = f'{PROJPATH}/notebooks/16_ccn/01_testprts.nc'
    test_refpath = f'{PROJPATH}/notebooks/16_ccn/02_refcrh.nc'
    assert exists(test_datapath), f'{test_datapath} must be downloaded first.'
    assert exists(test_refpath), f'{test_refpath} must be downloaded first.'

    with netCDF4.Dataset(test_datapath, "r") as fp:
        ################# The Array Dimensions #################
        # The number of particle sets
        n_partset = fp.dimensions['n_partset'].size
        # The number of particles within each set
        n_part = fp.dimensions['n_part'].size
        # The number of chemical specices
        n_chem = fp.dimensions['n_chem'].size

        #################### The Data Arrays ####################
        # The speciated mass of each chemical within each particle
        m_chmprtnp = np.array(fp.variables['m_chmprt'][:])
        assert m_chmprtnp.shape == (n_partset, n_part, n_chem)
        # The number of particles within each diameter bin
        n_prtnp = np.array(fp.variables['n_prt'][:])
        assert n_prtnp.shape == (n_partset, n_part)
        # The particle sets temperature
        tmprtrnp = np.array(fp.variables['tmprtr'][:])
        assert tmprtrnp.shape == (n_partset,)
        # The chemical densities in (kg/m^3)
        rho_chmnp = np.array(fp.variables['rho'][:])
        assert rho_chmnp.shape == (n_chem,)
        # The chemical Kappa values
        kappa_chmnp = np.array(fp.variables['kappa'][:])
        assert kappa_chmnp.shape == (n_chem,)

        ####################### Meta-Data #######################
        # The Chemical Species
        chem_species_str = fp.variables['chem_species'].names
        chem_species = chem_species_str.split(',')
        
    # Converting these arrays into pytorch tensors
    m_chmprt = torch.from_numpy(m_chmprtnp).to(device=tch_device, dtype=tch_dtype)
    assert m_chmprt.shape == (n_partset, n_part, n_chem)
    n_prt = torch.from_numpy(n_prtnp).to(device=tch_device, dtype=tch_dtype)
    assert n_prt.shape == (n_partset, n_part)
    tmprtr = torch.from_numpy(tmprtrnp).to(device=tch_device, dtype=tch_dtype)
    assert tmprtr.shape == (n_partset,)
    rho_chm = torch.from_numpy(rho_chmnp).to(device=tch_device, dtype=tch_dtype)
    assert rho_chm.shape == (n_chem,)
    kappa_chm = torch.from_numpy(kappa_chmnp).to(device=tch_device, dtype=tch_dtype)
    assert kappa_chm.shape == (n_chem,)

    crh_info = calc_crh(m_chmprt, n_prt, tmprtr, rho_chm, kappa_chm, chem_species, 
    n_partset, n_part, n_chem, void_kappa=0.0, n_rfiters=100)

    d_prtdry = crh_info['d_prtdry'] 
    assert d_prtdry.shape == (n_partset, n_part)
    d_prt = crh_info['d_prt'] 
    assert d_prt.shape == (n_partset, n_part)
    kappa_prt = crh_info['kappa_prt'] 
    assert kappa_prt.shape == (n_partset, n_part)
    d_prtcrt = crh_info['d_prtcrt'] 
    assert kappa_prt.shape == (n_partset, n_part)
    crh_prt = crh_info['crh_prt']
    assert crh_prt.shape == (n_partset, n_part)

    with netCDF4.Dataset(test_refpath, "r") as fp:
        crh_prtrefnp = np.array(fp.variables['crh'][:])
        assert crh_prtrefnp.shape == (n_partset, n_part)

        d_prtcrtrefnp = np.array(fp.variables['critical diameter'][:])
        assert d_prtcrtrefnp.shape == (n_partset, n_part)

        d_prtdryrefnp = np.array(fp.variables['dry diameter'][:])
        assert d_prtdryrefnp.shape == (n_partset, n_part)

        kappa_prtrefnp = np.array(fp.variables['kappa'][:])
        assert kappa_prtrefnp.shape == (n_partset, n_part)

    crh_prtref = torch.from_numpy(crh_prtrefnp).to(device=tch_device, dtype=tch_dtype)
    assert crh_prtref.shape == (n_partset, n_part)

    d_prtcrtref = torch.from_numpy(d_prtcrtrefnp).to(device=tch_device, dtype=tch_dtype)
    assert d_prtcrtref.shape == (n_partset, n_part)

    d_prtdryref = torch.from_numpy(d_prtdryrefnp).to(device=tch_device, dtype=tch_dtype)
    assert d_prtdryref.shape == (n_partset, n_part)

    kappa_prtref = torch.from_numpy(kappa_prtrefnp).to(device=tch_device, dtype=tch_dtype)
    assert kappa_prtref.shape == (n_partset, n_part)

    is_prtnonempty = n_prt > 0
    assert is_prtnonempty.shape == (n_partset, n_part)

    for tnsr_val, tnsr_ref, tnsr_name in [
        (d_prtdry,  d_prtdryref,  'Dry Diameter'),
        (kappa_prt, kappa_prtref, 'Particle Kappa'),
        (d_prtcrt,  d_prtcrtref,  'Critical Diameter'),
        (crh_prt,   crh_prtref,   'CRH')]:
        
        tnsr_err = (tnsr_ref - tnsr_val)[is_prtnonempty].abs()
        tnsr_maxerr = tnsr_err.max().item()
        tnsr_avgerr = tnsr_err.mean().item()
        tnsr_mederr = tnsr_err.median().item()
        tnsr_minerr = tnsr_err.min().item()

        print(f'  * {tnsr_name} Error:{" " * (20 - len(tnsr_name))} Minimum={tnsr_minerr:0.04g},' + 
            f' Median={tnsr_mederr:0.04g}, Mean={tnsr_avgerr:0.04g}, Maximum={tnsr_maxerr:0.04g}')
