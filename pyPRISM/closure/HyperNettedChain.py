#!python
from __future__ import division,print_function
from pyPRISM.closure.AtomicClosure import AtomicClosure
import numpy as np
class HyperNettedChain(AtomicClosure):
    r'''HyperNettedChain closure 

    **Mathematial Definition**

        .. math:: c_{\alpha,\beta}(r) = \exp\left(\gamma_{\alpha,\beta}(r)-U_{\alpha,\beta}(r)\right) - 1.0 -  \gamma_{\alpha,\beta}(r)

        .. math:: \gamma_{\alpha,\beta}(r) =  h_{\alpha,\beta}(r) - c_{\alpha,\beta}(r)

    
    **Variables Definitions**

        - :math:`h_{\alpha,\beta}(r)` 
            Total correlation function value at distance :math:`r` between
            sites :math:`\alpha` and :math:`\beta`.
    
        - :math:`c_{\alpha,\beta}(r)`
            Direct correlation function value at distance :math:`r` between
            sites :math:`\alpha` and :math:`\beta`.
    
        - :math:`U_{\alpha,\beta}(r)`
            Interaction potential value at distance :math:`r` between sites
            :math:`\alpha` and :math:`\beta`.
    
    **Description**

        The Hypernetted Chain Closure (HNC) is derived by expanding the
        direct correlation function, :math:`c(r)`, in powers of density shift
        from a reference state. See Reference [1] for a full derivation and
        discussion of this closure.
        
        The change of variables is necessary in order to use potentials with
        hard cores in the computational setting. Written in the standard form,
        this closure diverges with divergent potentials, which makes it
        impossible to numerically solve. 

        Compared to the PercusYevick closure, the HNC closure is a more
        accurate approximation of the full expression for the direct
        correlation function. Depsite this, it can produce inaccurate,
        long-range fluctuations that make it difficult to employ in
        phase-separating systems. The HNC closure performs well for systems
        where there is a disparity in site diameters and is typically used for
        the larger site. 
    
    References
    ----------
    #. Hansen, J.P.; McDonald, I.R.; Theory of Simple Liquids; Chapter 4, Section 4; 
       4th Edition (2013), Elsevier [`link
       <https://www.sciencedirect.com/science/book/9780123870322>`__]

    Example
    -------
    .. code-block:: python

        import pyPRISM

        sys = pyPRISM.System(['A','B'])
        
        sys.closure['A','A'] = pyPRISM.closure.PercusYevick()
        sys.closure['A','B'] = pyPRISM.closure.PercusYevick()
        sys.closure['B','B'] = pyPRISM.closure.HypernettedChain()

        # ** finish populating system object **

        PRISM = sys.createPRISM()

        PRISM.solve()
    
    '''
    def __init__(self,apply_hard_core=False):
        '''Contstructor

        Parameters
        ----------
        apply_hard_core: bool
            If *True*, the total correlation function will be assumed to be -1
            inside the core (:math:`r_{i,j}<(d_i + d_j)/2.0`) and the closure
            will not be applied in this region. Defaults to *True*.
        '''
        self.potential = None
        self.value = None
        self.sigma = None
        self.apply_hard_core = apply_hard_core
        
    def __repr__(self):
        return '<AtomicClosure: HyperNettedChain>'
    
    def calculate(self,gamma):
        '''Calculate direct correlation function based on supplied :math:`\gamma`

        Arguments
        ---------
        gamma: np.ndarray
            array of :math:`\gamma` values used to calculate the direct
            correlation function
        
        '''
        
        assert self.potential is not None,'Potential for this closure is not set!'
        
        assert len(gamma) == len(self.potential),'Domain mismatch!'
        
        
        return self.value

        
        
    def calculate(self,r,gamma):
        '''Calculate direct correlation function based on supplied :math:`\gamma`

        Arguments
        ---------
        r: np.ndarray
            array of real-space values associated with :math:`\gamma`

        gamma: np.ndarray
            array of :math:`\gamma` values used to calculate the direct
            correlation function
        
        '''
        
        assert self.potential is not None,'Potential for this closure is not set!'
        
        assert len(gamma) == len(self.potential),'Domain mismatch!'
        
        if self.apply_hard_core:
            assert self.sigma is not None, 'If apply_hard_core=True, sigma parameter must be set!'

            # apply hard core condition 
            self.value = -1 - gamma

            # calculate closure outside hard core
            mask = r>self.sigma
            self.value[mask] = np.exp(gamma[mask] - self.potential[mask]) - 1.0 - gamma[mask]
        else:
            self.value = np.exp(gamma - self.potential) - 1.0 - gamma

        
        return self.value
        
        
class HNC(HyperNettedChain):
    '''Alias of HyperNettedChain'''
    pass
