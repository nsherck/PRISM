#!python
from __future__ import division,print_function
from pyPRISM.core.PairTable import PairTable
from pyPRISM.core.MatrixArray import MatrixArray
from pyPRISM.core.Space import Space
import warnings
import numpy as np

COMPONENT_WARNING = '''
This calculation was derived for a two component system. It is often the case that
these calculations can be generalized for pairs of sites withing multicomponent 
systems. We caution the user when interpreting the data from this calculation 
for more than two components. 
'''

def chi(PRISM,extrapolate=True):
    r'''Calculate the effective interaction parameter, :math:`\chi`

    Parameters
    ----------
    PRISM: pyPRISM.core.PRISM
        A **solved** PRISM object.

    extrapolate: bool, *optional*
        If *True*, only return the chi value extrapolated to :math:`k=0` rather
        than returning :math:`\chi(k)`
    
    Returns
    -------
    chi: pyPRISM.core.PairTable
        PairTable of all :math:`\chi(k)`  or :math:`\chi(k=0)` values
    

    **Mathematical Definition**

    .. math::
        
        \hat{\chi}_{\alpha,\beta}(k)  = \frac{0.5 \rho}{R^{+0.5} \phi_{\alpha} + R^{-0.5} \phi_{\beta}} (R^{-1} \hat{C}_{\alpha,\alpha}(k)
        + R \hat{C}_{\beta,\beta}(k) - 2 \hat{C}_{\alpha,\beta}(k))

    .. math::

        R = v_{\alpha}/v_{\beta}


    **Variable Definitions**

        - :math:`\hat{\chi}_{\alpha,\beta}(k)` 
            Wavenumber dependent effective interaction parameter between site
            types :math:`\alpha` and :math:`\beta`
    
        - :math:`\rho`
            Total system density from the :class:`pyPRISM.core.Density`
            instance stored in the system object (which is stored in the PRISM
            object)
    
        - :math:`\phi_{\alpha},\phi_{\beta}` 
            Volume fraction of site types :math:`\alpha` and :math:`\beta`. 
    
            .. math::
    
                \phi_{\alpha} = \frac{\rho_{\alpha}}{\rho_{\alpha} + \rho_{\beta}}

        - :math:`v_{\alpha},v_{\beta}` 
            Volume of site type :math:`\alpha` and :math:`\beta`


    **Description**

        :math:`\hat{\chi}_{\alpha,\beta}(k)` describes the overall effective
        interactions between site types :math:`\alpha` and :math:`\beta` as a
        single number. While there are many different definitions of
        :math:`\chi`, this is an effective version that takes into account both
        *entropic* and *enthalpic* interactions. In this way, this :math:`\chi`
        is similar to a second virial coefficient. In terms of value,
        :math:`\chi<0` indicates effective attraction and :math:`\chi>0`
        effective repulsion. 

        As most theories do not take into account the (potentially contentious [2,3])
        wavenumber dependence of :math:`\chi`, the zero-wavenumber extrapolation
        is often used when reporting PRISM-based :math:`\chi` values. For
        convenience, the full wavenumber dependent curve can be requested, but
        only the :math:`k=0` values are returned by default. 

    .. warning::
        
        Using standard atomic closures (e.g, PY, HNC, MSA), PRISM theory may
        not predict the correct scaling of spinodal temperatures for phase
        separating systems. While this issue is mitigated by using molecular
        closures,[3] these closures are not currently implemented in pyPRISM.
        For more information, this issue is referenced in the pyPRISM
        paper.[5]. We urge users to do their due diligence in understanding how
        well these closures and PRISM theory perform for their systems of
        interest.

    .. warning::

        The :math:`\chi` calculation is only valid for multicomponent systems
        i.e. systems with more than one defined type. This method will throw an
        exception if passed a 1-component PRISM object. 

    .. warning::

        This calculation is only rigorously defined in the two-component case.
        With that said, pyPRISM allows this method to be called for 
        multicomponent systems in order to calculate pairwise
        :math:`\chi` values. We urge caution when using this method for
        multicomponent systems as it is not clear if this approach is fully
        rigorous.

    .. warning::

        Passing an unsolved PRISM object to this function will still produce
        output based on the default values of the attributes of the PRISM
        object.
    
    References
    ----------

        1.  Schweizer, Curro, Thermodynamics of Polymer Blends, J. Chem. Phys., 1989
            91 (8) 5059, DOI: 10.1063/1.457598 [`link
            <http://dx.doi.org/10.1063/1.457598>`__]

        2.  Zirkel, A., et al., Small-angle neutron scattering investigation of
            the Q-dependence of the Flory-Huggins interaction parameter in a binary
            polymer blend. Macromolecules, 2002. 35(19): p. 7375-7386. [`link
            <http://dx.doi.org/10.1021/ma010576o>`__]

        3.  Cabral, Higgins, Small Angle Neutron Scattering from the
            Highly Interacting Polymer Mixture TMPC/PSd: No Evidence of Spatially
            Dependent chi Parameter. Macromolecules, 2009. 42(24): p. 9528-9536.
            [`link <http://dx.doi.org/10.1021/ma901516v>`__]

        4.  Schweizer, K.S. and A. Yethiraj, POLYMER REFERENCE INTERACTION SITE
            MODEL-THEORY - NEW MOLECULAR CLOSURES FOR PHASE-SEPARATING FLUIDS AND
            ALLOYS. Journal of Chemical Physics, 1993. 98(11): p. 9053-9079. [`link
            <http://dx.doi.org/10.1063/1.464465>`__]

        5.  Martin, T.B.; Gartner, T.E. III;  Jones, R.L.; Snyder, C.R.; Jayaraman,
            A.; pyPRISM: A Computational Tool for Liquid State Theory
            Calculations of Macromolecular Materials, Macromolecules, 2018, 51 (8),
            p2906-2922 [`link <https://dx.doi.org/10.1021/acs.macromol.8b00011>`__]

    Example
    -------
    .. code-block:: python

        import pyPRISM

        sys = pyPRISM.System(['A','B'])

        # ** populate system variables **
        
        PRISM = sys.createPRISM()

        PRISM.solve()

        chi = pyPRISM.calculate.chi(PRISM)

        chi_AB = chi['A','B']
        chi_AA = chi['A','A'] #returns None because self-chi values are not defined

    '''
    
    assert PRISM.sys.rank>1,'The chi calculation is only valid for multicomponent systems'

    if PRISM.sys.rank!=2:
        warnings.warn(COMPONENT_WARNING)
    
    if PRISM.directCorr.space == Space.Real:
        PRISM.domain.MatrixArray_to_fourier(PRISM.directCorr)
        
    
    chi = PairTable(name='chi',types=PRISM.sys.types)
    chi0 = PairTable(name='chi0',types=PRISM.sys.types)
    for i,t1 in enumerate(PRISM.sys.types):
        for j,t2 in enumerate(PRISM.sys.types):
            if i<j:
                C_AA = PRISM.directCorr[t1,t1]
                C_AB = PRISM.directCorr[t1,t2]
                C_BB = PRISM.directCorr[t2,t2]

                v_A = 4.0/3.0 * np.pi * (PRISM.sys.diameter[t1]/2.0)**(3.0)
                v_B = 4.0/3.0 * np.pi * (PRISM.sys.diameter[t2]/2.0)**(3.0)

                rho_A = PRISM.sys.density[t1]
                rho_B = PRISM.sys.density[t2]

                phi_A = rho_A/(rho_A + rho_B)
                phi_B = rho_B/(rho_A + rho_B)

                R = v_A/v_B
                
                chi[t1,t2] = (R**(-0.5)*phi_A + R**(0.5)*phi_B)**(-1.0)*0.5*PRISM.sys.density.total*(R**(-1.0) * C_AA + R*C_BB - 2*C_AB)

                x = PRISM.sys.domain.k[:3]
                y = chi[t1,t2][:3]
                fit = np.poly1d(np.polyfit(x,y,2))
                chi0[t1,t2] = fit(0)

    if extrapolate:
        return chi0
    else:
        return chi
