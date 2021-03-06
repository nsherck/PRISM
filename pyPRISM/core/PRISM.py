#!python
from __future__ import division,print_function

from pyPRISM.core.Space import Space
from pyPRISM.core.MatrixArray import MatrixArray
from pyPRISM.core.IdentityMatrixArray import IdentityMatrixArray
from pyPRISM.closure.AtomicClosure import AtomicClosure
from pyPRISM.closure.MolecularClosure import MolecularClosure

from scipy.optimize import root

import numpy as np

from copy import deepcopy
import warnings

class PRISM:
    r'''Primary container for a storing a PRISM calculation
    
    Each pyPRISM.PRISM object serves as an encapsulation of a fully specified
    PRISM problem including all inputs needed for the calculation and the
    function to be numerically minimized. 
    
    Attributes
    ----------
    domain: pyPRISM.Domain
        The Domain object fully specifies the Real- and Fourier- space solution
        grids.
    
    directCorr: pyPRISM.MatrixArray
        The direct correlation function for all pairs of site types
        
    omega: pyPRISM.MatrixArray
        The intra-molecular correlation function for all pairs of site types
    
    closure: pyPRISM.core.PairTable of pyPRISM.closure.Closure
        Table of closure objects used to generate the direct correlation
        functions (directCorr)
        
    pairCorr: pyPRISM.MatrixArray
        The *inter*-molecular pair correlation functions for all pairs of
        site types. Also commonly refered to as the radial distribution functions.
    
    totalCorr: pyPRISM.MatrixArray
        The *inter*-molecular total correlation function is simply the pair
        correlation  function y-shifted by 1.0 i.e.  totalCorr = pairCorr - 1.0
        
    potential: pyPRISM.MatrixArray
        Interaction potentials for all pairs of sites
        
    GammaIn,GammaOut: pyPRISM.MatrixArray
        Primary inputs and outputs of the PRISM cost function. Gamma is defined as
        "totalCorr - directCorr" (in Fourier space) and results from a change
        of variables used to remove divergences in the closure relations. 
    
    OC,IOC,I,etc: pyPRISM.MatrixArray
        Various MatrixArrays used as intermediates in the PRISM functional.
        These arrays are pre-allocated and stored for efficiency. 
    
    x,y: float np.ndarray
        Current inputs and outputs of the cost function
    
    pairDensityMatrix: float np.ndarray
        Rank by rank array of pair densities between sites. See :class:`pyPRISM.core.Density`
            
    siteDensityMatrix: float np.ndarray
        Rank by rank array of site densities. See :class:`pyPRISM.core.Density`
    
    Methods
    -------
    cost:
        Primary cost function used to define the criteria of a "converged"
        PRISM solution. The numerical solver will be given this function 
        and will attempt to find the inputs (self.x) that make the outputs
        (self.y) as close to zero as possible. 
        
        
    '''
    def __init__(self,sys):
        self.sys = deepcopy(sys)

        # Need to set the potential for each closure object
        for (i,j),(t1,t2),U in self.sys.potential.iterpairs():
            if isinstance(self.sys.closure[t1,t2],AtomicClosure):
                #only set sigma if not set directly in potential
                if U.sigma is None:
                    U.sigma = self.sys.diameter[t1,t2]
                self.sys.closure[t1,t2].sigma = self.sys.diameter[t1,t2]
                self.sys.closure[t1,t2].potential = U.calculate(self.sys.domain.r) / self.sys.kT
            elif isinstance(self.sys.closure[t1,t2],MolecularClosure):
                raise NotImplementedError('Molecular closures are not fully implemented in this release.')
                #only set sigma if not set directly in potential
                if U.sigma is None:
                    U.sigma = self.sys.diameter[t1,t2]
                self.sys.closure[t1,t2].sigma = self.sys.diameter[t1,t2]
                self.sys.closure[t1,t2].potential = U.calculate_attractive(self.sys.domain.r) / self.sys.kT

        
        #cost function input and output
        self.x         = np.zeros(sys.rank*sys.rank*sys.domain.length)
        self.y         = np.zeros(sys.rank*sys.rank*sys.domain.length)

        # The omega objects must be converted to a MatrixArray of the actual correlation
        # function values rather than a table of OmegaObjects.
        applyFunc = lambda x: x.calculate(sys.domain.k)
        self.omega  = self.sys.omega.apply(applyFunc,inplace=False).exportToMatrixArray(space=Space.Fourier)
        self.omega *= sys.density.site #omega should always be scaled by site density 
        
        # Spaces are set based on when they are used in self.cost(...). In some cases,
        # this is redundant because these array's will be overwritten with copies and
        # then their space will be inferred from their parent MatrixArrays
        self.directCorr = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.directCorrOut = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.totalCorr  = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)
        self.GammaIn    = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)
        self.GammaOut   = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Real,types=sys.types)
        self.OC         = MatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)
        self.I          = IdentityMatrixArray(length=sys.domain.length,rank=sys.rank,space=Space.Fourier,types=sys.types)
        
    def __repr__(self):
        return '<PRISM length:{} rank:{}>'.format(self.sys.domain.length,self.sys.rank)
        
    def cost(self,x):
        r'''Cost function 
        
        There are likely several cost functions that could be imagined using
        the PRISM equations. In this case we formulate a self-consistent 
        formulation where we expect the input of the PRISM equations to be
        identical to the output. 

        .. image:: ../../img/numerical_method.svg
            :width: 300px
        
        The goal of the solve method is to numerically optimize the input (:math:`r \gamma_{in}`) 
        so that the output (:math:`r(\gamma_{in}-\gamma_{out})`) is minimized to zero.
        
        '''
        self.x = x #store input

        # The np.copy is important otherwise x saves state between calls to
        # this function.
        self.GammaIn.data = np.copy(x.reshape((-1,self.sys.rank,self.sys.rank)))
        self.GammaIn     /= self.sys.domain.long_r
        
        # directCorr is calculated directly in Real space but immediately 
        # inverted to Fourier space. We must reset this from the last call.
        self.directCorr.space = Space.Real 
        for (i,j),(t1,t2),closure in self.sys.closure.iterpairs():
            if isinstance(closure,AtomicClosure):
                self.directCorr[t1,t2] = closure.calculate(self.sys.domain.r,self.GammaIn[t1,t2])
            elif isinstance(closure,MolecularClosure):
                raise NotImplementedError('Molecular closures are untested and not fully implemented.')
                self.directCorr[t1,t2] = closure.calculate(self.GammaIn[t1,t2],self.omega[t1,t1],self.omega[t2,t2])
            else:
                raise ValueError('Closure type not recognized')
            
        self.sys.domain.MatrixArray_to_fourier(self.directCorr)
        
        self.OC = self.omega.dot(self.directCorr)
        self.IOC = self.I - self.OC
        self.IOC.invert(inplace=True)
        
        self.totalCorr  = self.IOC.dot(self.OC).dot(self.omega)
        self.totalCorr /= self.sys.density.pair
        
        self.GammaOut  = self.totalCorr - self.directCorr
        
        self.sys.domain.MatrixArray_to_real(self.GammaOut)
        
        #self.y = self.sys.domain.long_r*(self.GammaOut.data - self.GammaIn.data)
        
        
        
        #return self.y.reshape((-1,))
        return [self.sys.domain.long_r*self.GammaOut.data, self.sys.domain.long_r*self.GammaIn.data]
    
    def cost1(self,x):
        r'''Cost function 
        
        There are likely several cost functions that could be imagined using
        the PRISM equations. In this case we formulate a self-consistent 
        formulation where we expect the input of the PRISM equations to be
        identical to the output. 

        .. image:: ../../img/numerical_method.svg
            :width: 300px
        
        The goal of the solve method is to numerically optimize the input (:math:`r \gamma_{in}`) 
        so that the output (:math:`r(\gamma_{in}-\gamma_{out})`) is minimized to zero.
        
        '''
        self.x = x #store input

        # The np.copy is important otherwise x saves state between calls to
        # this function.
        #NSOUT self.GammaIn.data = np.copy(x.reshape((-1,self.sys.rank,self.sys.rank)))
        #NSOUT self.GammaIn     /= self.sys.domain.long_r
        
        self.directCorr.data = np.copy(x.reshape((-1,self.sys.rank,self.sys.rank)))
        self.sys.domain.MatrixArray_to_fourier(self.directCorr)
        
        # do OZ equation
        self.OC = self.omega.dot(self.directCorr)
        self.IOC = self.I - self.OC
        self.IOC.invert(inplace=True)
        
        # get out h(r)
        self.totalCorr  = self.IOC.dot(self.OC).dot(self.omega)
        self.totalCorr /= self.sys.density.pair
        
        # calculate gammain

        self.GammaIn = self.totalCorr - self.directCorr
        self.sys.domain.MatrixArray_to_real(self.GammaIn) # convert self.GammaIn to real matrix

        # directCorr is calculated directly in Real space but immediately 
        # inverted to Fourier space. We must reset this from the last call.
        #NSOUT self.directCorr.space = Space.Real 
        for (i,j),(t1,t2),closure in self.sys.closure.iterpairs():
            if isinstance(closure,AtomicClosure):
                self.directCorrOut[t1,t2] = closure.calculate(self.sys.domain.r,self.GammaIn[t1,t2])
            elif isinstance(closure,MolecularClosure):
                raise NotImplementedError('Molecular closures are untested and not fully implemented.')
                self.directCorrOut[t1,t2] = closure.calculate(self.GammaIn[t1,t2],self.omega[t1,t1],self.omega[t2,t2])
            else:
                raise ValueError('Closure type not recognized')
        
        
        self.sys.domain.MatrixArray_to_real(self.directCorr)
        #self.y = self.sys.domain.long_r*(self.GammaOut.data - self.GammaIn.data)
        print(self.directCorrOut.space)
        print(self.directCorr.space)
        #return self.y.reshape((-1,))
        return [(self.directCorrOut.data).reshape((-1,)), (self.directCorr.data).reshape((-1,))]
        
    def solve(self,guess=None,method='krylov',options=None):
        '''Attempt to numerically solve the PRISM equations
        
        Using the supplied inputs (in the constructor), we attempt to numerically
        solve the PRISM equations using the scheme laid out in :func:`cost`. If the 
        numerical solution process is successful, the attributes of this class
        will contain the solved values for a given input i.e. self.totalCorr will
        contain the numerically optimized (solved) total correlation functions.

        This function also does basic checks to ensure that the results are 
        physical. At this point, this consists of checking to make sure that
        the pair correlation functions are not negative. If this isn't true
        a warning is issued to the user. 
        
        Parameters
        ----------
        guess: np.ndarray, size (rank*rank*length)
            The initial guess of :math:`\gamma` to the numerical solution process.
            The numpy array should be of size rank x rank x length corresponding to 
            the a full flattened MatrixArray. If not specified, an initial guess
            of all zeros is used. 
            
        method: string
            Set the type of optimization scheme to use. The scipy documentation
            for `scipy.optimize.root
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html>`__
            details the possible values for this parameter. 
            

        options: dict
            Dictionary of options specific to the chosen solver method. The
            scipy documentation for `scipy.optimize.root
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html>`__
            details the possible values for this parameter. 
        
        '''
        if guess is None:
            guess = np.zeros(self.sys.rank*self.sys.rank*self.sys.domain.length)
            
        if options is None:
            options = {'disp':True}

        self.minimize_result = root(self.cost,guess,method=method,options=options)

        
        if self.totalCorr.space == Space.Fourier:
            self.sys.domain.MatrixArray_to_real(self.totalCorr)

        tol = 1e-5
        warnstr = 'Pair correlations are negative (value = {:3.2e}) for {}-{} pair!'
        for i,(t1,t2),H in self.totalCorr.iterpairs():
            if np.any(H<-(1.0+tol)):
                val = np.min(H)
                warnings.warn(warnstr.format(val,t1,t2))
        
        return self.minimize_result
        
    
