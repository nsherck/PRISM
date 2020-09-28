#import sys, os
#cwd = os.getcwd()
#sys.path.append(os.path.join(cwd,'pyPRISMpackage/pyPRISM'))
import pyPRISM
import pyPRISM
print(pyPRISM.__file__)

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

sigma = 1.
scale = 1
kbT = 1.
dr = 0.01
power = 10 # 2**power == # grid points; therefore length = dr*2**power

sys = pyPRISM.System(['A'],kT=kbT)
sys.domain = pyPRISM.Domain(dr=dr,length=2**power)
print('r =',sys.domain.r)
print('k = ',sys.domain.k)

sys.diameter['A'] = dr
sys.omega['A','A'] = pyPRISM.omega.SingleSite()

def cust_norm(_GammaIn,_GammaOut):
    _diff = np.subtract(_GammaOut,_GammaIn)
    _diff = np.sum(_diff**2)
    _diff = np.divide(_diff,np.sum(_GammaOut**2))
    return _diff


''' Try with single species. Compare to MD.'''
scales = [1.]
guess_1 = np.zeros(sys.rank*sys.rank*sys.domain.length)
for _i, xfact in enumerate(scales):
    print("Scale Factor: {}".format(xfact))
    
    uAA = 7.07106*xfact
    range = 1.570796
    sys.potential['A','A'] = pyPRISM.potential.Gaussian(epsilon=-1*uAA,alpha=range,sigma=0,high_value=uAA)
    sys.density['A']       = 1.0
    print(sys.density)
    sys.closure['A','A'] = pyPRISM.closure.HyperNettedChain(apply_hard_core=False)

    PRISM = sys.createPRISM()
    #PRISM.solve(guess=guess_1,method='anderson',options={'tol_norm':cust_norm})#,method='Krylov')
    #guess_1 = np.copy(PRISM.x)
    
    _iter = 0
    Norm = 1
    alpha = 0.9
    while Norm > 1e-4: # custom update 
        if _iter == 0:
            gamma_in = guess_1
        out = PRISM.cost(gamma_in)
        print(out[0].shape) # GammaOut
        print(out[1].shape) # GammaIng
        
        # update gamma
        gamma_new = np.add(alpha*out[1],(1-alpha)*out[0])
        #gamma_new = np.multiply(sys.domain.r,gamma_new)
        print(gamma_new.shape)
        Norm = cust_norm(out[1],out[0])
        
        #plt.plot(sys.domain.r,gamma_in,'-r')
        #plt.plot(sys.domain.r,gamma_new,'-b')
        #plt.show()
        #plt.close()
        gamma_in = np.copy(gamma_new)
        print('Iter: {} Norm: {}'.format(_iter,Norm))
        _iter += 1

    #guess_1 = np.copy(PRISM.x)
    #print(guess_1)
    #print(guess_1.shape)

    x = sys.domain.r
    RDF_AA = pyPRISM.calculate.pair_correlation(PRISM)['A','A']
    np.savetxt('RDF_AA.dat',np.column_stack((x,RDF_AA)))
    data = np.loadtxt('A_RDF.txt',delimiter=',')
    data1 = np.loadtxt('RDF_POL_POL_uab_0.0.dat')
    plt.plot(data[:,0],data[:,1],'or')
    plt.plot(data1[:,0]*10,data1[:,1],'ob')
    plt.plot(x, RDF_AA, "-r")
    plt.xlim(0,10)
    plt.savefig('RDF.png')
    plt.show()
    plt.close()

''' Now try for AB Gaussian Fluid '''    
sys = pyPRISM.System(['A','B'],kT=kbT)
sys.domain = pyPRISM.Domain(dr=dr,length=2**power)
print('r =',sys.domain.r)
print('k = ',sys.domain.k)
sys.diameter['A'] = dr
sys.diameter['B'] = dr
uBB = 7.07106
uAB = 7.77817
range = 1.570796
sys.potential['A','A'] = pyPRISM.potential.Gaussian(epsilon=-1*uAA,alpha=range,sigma=0,high_value=uAA)
sys.potential['A','B'] = pyPRISM.potential.Gaussian(epsilon=-1*uAB,alpha=range,sigma=0,high_value=uAB)
sys.potential['B','B'] = pyPRISM.potential.Gaussian(epsilon=-1*uBB,alpha=range,sigma=0,high_value=uBB)
sys.density['A']       = 0.5
sys.density['B']       = 0.5
sys.omega['A','A'] = pyPRISM.omega.SingleSite()
sys.omega['A','B'] = pyPRISM.omega.SingleSite()
sys.omega['B','B'] = pyPRISM.omega.SingleSite()

print(sys.rank)

guess_2 = np.zeros(sys.rank*sys.rank*sys.domain.length)
matrix_in = np.zeros((len(guess_1),2,2))
plt.plot(sys.domain.r,guess_1,'or')
plt.xlim(0,10)
plt.savefig('Gamma.png')
plt.show()
plt.close()
matrix_in[:,0,0] = 0.5*guess_1
matrix_in[:,0,1] = 0.25*guess_1
matrix_in[:,1,0] = 0.25*guess_1
matrix_in[:,1,1] = 0.5*guess_1
#matrix_in[:,1,1] = []
matrix_in = matrix_in.flatten()
#guess_2[0:512] = guess_1
#print(guess_2.shape)
#guess_chk = np.copy(guess_2.reshape((-1,sys.rank,sys.rank)))
#print(guess_chk.shape)


guess_1 = np.zeros(sys.rank*sys.rank*sys.domain.length)
#xfrac_list = [0]
#xfrac_list.extend(np.linspace(0.1,1,10))

for _i,_xfrac in enumerate([1.]):
    print('Using HNC closure. Density: ')
    sys.density['A']       = 0.5
    sys.density['B']       = 0.5
    
    sys.kT = 1
    sys.closure['A','A'] = pyPRISM.closure.HyperNettedChain(apply_hard_core=False)
    sys.closure['A','B'] = pyPRISM.closure.HyperNettedChain(apply_hard_core=False)
    sys.closure['B','B'] = pyPRISM.closure.HyperNettedChain(apply_hard_core=False)
    
    _xfrac = 1.
    print('Using HNC closure. Density: {}'.format(sys.density))
    uAB = (7.77817-uAA)*_xfrac + uAA
    print('uAB {} and uAA {}'.format(uAB,uAA))
    sys.potential['A','B'] = pyPRISM.potential.Gaussian(epsilon=-1*uAB,alpha=range,sigma=0,high_value=uAB)
    PRISM = sys.createPRISM()
    
    _iter = 0
    Norm = 1
    alpha = 0.99
    while Norm > 0.01:
        if _iter == 0:
            gamma_in = guess_1 # tried playing around with seed based off one component
        out = PRISM.cost(gamma_in)
        print(out[0].shape) # GammaOut
        print(out[1].shape) # GammaIn

        gamma_new = np.add(alpha*out[1],(1-alpha)*out[0])
        #gamma_new = np.multiply(sys.domain.r,gamma_new)
        print(gamma_new.shape)
        Norm = cust_norm(out[1],out[0])
        
        #plt.plot(sys.domain.r,gamma_in[0:int(2**power)],'-r')
        #plt.plot(sys.domain.r,gamma_new[0:int(2**power)],'-b')
        #plt.show()
        #plt.close()
        gamma_in = np.copy(gamma_new)
        print('Iter: {} Norm: {}'.format(_iter,Norm))
        _iter += 1
    

    #dirCorr = PRISM.directCorr()
    
    
    #PRISM.solve(guess=guess_1,method='anderson',options={'maxiter':1000,'jac_options':{'M':5},'tol_norm':cust_norm})#guess=matrix_in,method='Krylov')#,options={'maxiter':500})
    
    out = PRISM.cost(guess_1)
    print(out[0].shape)
    print(out[1].shape)
    guess_1 = np.copy(PRISM.x)

    x = sys.domain.r
    RDF_AA = pyPRISM.calculate.pair_correlation(PRISM)['A','A']
    RDF_AB = pyPRISM.calculate.pair_correlation(PRISM)['A','B']
    np.savetxt('RDF_AA.dat',np.column_stack((x,RDF_AA)))
    np.savetxt('RDF_AB.dat',np.column_stack((x,RDF_AB)))
    data = np.loadtxt('A_RDF.txt',delimiter=',')
    data1 = np.loadtxt('RDF_POL_POL_uaa_2.0.dat')
    plt.plot(data[:,0],data[:,1],'or')
    plt.plot(data1[:,0]*10,data1[:,1],'ob')
    plt.plot(x, RDF_AA, "-r")
    plt.plot(x, RDF_AB, "-b")
    plt.xlim(0,10)
    plt.savefig('RDF.png')
    plt.show()
    plt.close()

''' calculate pressure '''
# Left over from doing just single species
xsq = np.multiply(x,x)
xsqsq = np.multiply(xsq,xsq)
integrand = 8*np.pi*range*uAB*np.multiply(np.multiply(xsqsq,np.exp(-1*range*xsq)),RDF_AA)
int = sp.integrate.simps(integrand,x)
Pex = sys.density['A']**2/6.*int
Pid = sys.density['A']
Ptot = Pid + Pex
print("Pressure ideal: {}".format(Pid))
print("Pressure excess: {}".format(Pex))
print("Total Pressure: {}".format(Ptot))