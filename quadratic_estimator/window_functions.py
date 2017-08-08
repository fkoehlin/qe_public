#!/usr/bin/env python
# encoding: UTF8
# Code for estimating maximum likelihood band powers of the lensing power spectrum
# following the "quadratic estimator" method by Hu & White 2001 (ApJ, 554, 67) 
# implementation by Fabian Koehlinger

#import sys
#import os
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.special as special
#import time
import multiprocessing

@np.vectorize
def _sph_jn(n, x, derivative=False):
    
    sph_jn, deriv_sph_jn = special.sph_jn(n, x)
    
    if derivative:
        return deriv_sph_jn    
    else:
        return sph_jn

def sph_jn(n, x, derivative=False):
    
    return _sph_jn(n, x, derivative=derivative)

def unwrap_moment(arg, **kwarg):
    """ Helper function for multiprocessing; converts `f([1,2])` to `f(1,2)` call """
    
    return WindowFunctions.moment_wn(*arg, **kwarg)

# this is only useful if I need many w(l) but very inefficient for just getting one l-value...
# not used in QE any longer!
class WindowFunctions(object):
    """ Calculation of moments n = 0, 4, 8 (currently hard-coded) of window function"""
    
    def __init__(self, sigma=0.1, l_min=1e-2, l_max=1e4, number_nodes=1e4, ncpus=1, for_plot=False):
        
        l = np.logspace(np.log10(l_min), np.log10(l_max), number_nodes)
        #nells = int(l_max - l_min + 1)
        #l = np.linspace(l_min, l_max, nells)
        self.sigma = np.deg2rad(sigma)
        self.norm0, dnorm0 = self.__norm(0)
        self.norm4, dnorm4 = self.__norm(4)
        self.norm8, dnorm8 = self.__norm(8)
        
        # be nice and don't steal all CPUs by default...
        cpus_available = multiprocessing.cpu_count()
        if ncpus == 0:
            take_cpus = cpus_available
        elif ncpus > cpus_available:
            take_cpus = cpus_available
        else:
            take_cpus = ncpus
        
        self.n = 0
        pool = multiprocessing.Pool(processes=take_cpus)
        # self is first argument of function!
        results_w0 = np.asarray(pool.map(unwrap_moment, zip([self] * len(l), l)))
        pool.close()
        
        w0 = results_w0[:, 0]
        dw0 = results_w0[:, 1]
        
        self.n = 4
        pool = multiprocessing.Pool(processes=take_cpus)
        # self is first argument of function!
        results_w4 = np.asarray(pool.map(unwrap_moment, zip([self] * len(l), l)))
        pool.close()
        
        w4 = results_w4[:, 0]
        dw4 = results_w4[:, 1]
        
        self.n = 8
        pool = multiprocessing.Pool(processes=take_cpus)
        # self is first argument of function!
        results_w8 = np.asarray(pool.map(unwrap_moment, zip([self] * len(l), l)))
        pool.close()
        
        w8 = results_w8[:, 0]
        dw8 = results_w8[:, 1]
        
        self.w0 = interpolate.interp1d(l, w0, kind='linear')
        self.w4 = interpolate.interp1d(l, w4, kind='linear')
        self.w8 = interpolate.interp1d(l, w8, kind='linear')
        
        self.l_nodes = l
        
    def __norm(self, n):
    
        if type(n) == int and n > 0:
            val = np.pi
            dval = 0.
        elif type(n) == int:
            val = 2. * np.pi
            dval = 0.
        else:
            # normalization is the integral over the weighting function square:
            val, dval = norm, dnorm = integrate.quad(lambda x: np.cos(n * x)**2, 0., 2. * np.pi, limit=1000)
    
        return val, dval
    
    def moment_wn(self, l):
    
        w, dw = integrate.quad(lambda phi: self.__integrand_wn(phi, l), 0., 2. * np.pi, limit=1000)
        
        # TODO: Check if absolute value here is correct (makes only a difference for n=4; I can reproduce 
        # Hu & White's Fig. 2 only when plotting |w_n(l)|, so I always assumed the absolute value is correct, 
        # but maybe they made a mistake in the label of the Fig.?)        
        
        return np.abs(w), dw
        
    def __integrand_wn(self, phi, l):
    
        w_sqr = self.__window_sqr(l, phi)
    
        return w_sqr * np.cos(self.n * phi)
    
    '''
    def __window(self, l, phi):
    
        x1 = l * self.sigma / 2. * np.cos(phi)
        x2 = l * self.sigma / 2. * np.sin(phi)
    
        return np.sin(x1) * np.sin(x2) / x1 / x2
    '''
    
    def __window_sqr(self, l, phi):
        
        # devision by pi due to sinc(x) = sin(pi x) / (pi x)
        x1 = l * self.sigma / (2. * np.pi) * np.cos(phi)
        x2 = l * self.sigma / (2. * np.pi) * np.sin(phi)
        
        return np.sinc(x1)**2 * np.sinc(x2)**2
    
    def getWindowFunction(self, l, n):
        
        #if self.sigma != sigma:
        #    print 'Fatal error!!!'
        #    exit()
        if n == 0:
            return self.w0(l) / self.norm0
        if n == 4:
            return self.w4(l) / self.norm4
        if n == 8:
            return self.w8(l) / self.norm8
            
    def getArray(self, n):
    
        if n == 0:
            return self.w0(self.l_nodes) / self.norm0
        if n == 4:
            return self.w4(self.l_nodes) / self.norm4
        if n == 8:
            return self.w8(self.l_nodes) / self.norm8
    
    
class WindowFunctionsSimple(object):
    """ Calculation of moments of window function. Kernel is the same as used in class above, but this 
        implementation is much simpler (no precalculations and interpolations) and aimed to be used in 
        calculation of the band window matrix (which requires only a single value for specified l).
    """
    
    def __init__(self, sigma=0.1):
        #l = np.logspace(np.log10(l_min), np.log10(l_max), number_nodes)
        self.sigma = np.deg2rad(sigma)
        
    def __norm(self, n):
    
        if type(n) == int and n > 0:
            val = np.pi
            dval = 0.
        elif type(n) == int:
            val = 2. * np.pi
            dval = 0.
        else:
            # normalization is the integral over the weighting function square:
            val, dval = norm, dnorm = integrate.quad(lambda x: np.cos(n * x)**2, 0., 2. * np.pi, limit=1000)
            
        return val, dval
    
    def moment_wn(self, l, n):
        
        w, dw = integrate.quad(lambda phi: self.__integrand_wn(phi, l, n), 0., 2. * np.pi, limit=1000)
    
        # TODO: Check if absolute value here is correct (makes only a difference for n=4; I can reproduce 
        # Hu & White's Fig. 2 only when plotting |w_n(l)|, so I always assumed the absolute value is correct, 
        # but maybe they made a mistake in the label of the Fig.?)        
        # best guess: Browm et al. (2003, MNRAS 341, 100-118) they write the window function as absolute value before squaring (but no comment on its moments) 
        
        return np.abs(w), dw
        
    def __integrand_wn(self, phi, l, n):
    
        w_sqr = self.__window_sqr(l, phi)
    
        return w_sqr * np.cos(n * phi)

    '''
    def __window_sqr(self, l, phi):
    
        #c = l * self.sigma / 2.
        
        x1 = l * self.sigma / 2. * np.cos(phi)
        x2 = l * self.sigma / 2. * np.sin(phi)
        
        #j0_x1 = special.sph_jn(0, x1)[0][0]
        #j0_x2 = special.sph_jn(0, x2)[0][0]        
        
        j0_x1 = sph_jn(0, x1)
        j0_x2 = sph_jn(0, x2)      
                
        #return np.sin(x1)**2 * np.sin(x2)**2 / x1**2 / x2**2
        return j0_x1**2 * j0_x2**2
    '''
    
    def __window_sqr(self, l, phi):
        
        # devision by pi due to sinc(x) = sin(pi x) / (pi x)
        x1 = l * self.sigma / (2. * np.pi) * np.cos(phi)
        x2 = l * self.sigma / (2. * np.pi) * np.sin(phi)
        
        return np.sinc(x1)**2 * np.sinc(x2)**2
    
    def getWindowFunction(self, l, n):
        
        norm = self.__norm(n)[0]
        
        if isinstance(l, np.ndarray):
            
            window = np.zeros_like(l)
            for index in xrange(l.size):
                window[index] = self.moment_wn(l[index], n)[0] / norm 
        
        else:
            
            window = self.moment_wn(l, n)[0] / norm 
        
        return window 


if __name__ == '__main__':
    
    # some testing:        
    import matplotlib.pyplot as plt    
    plt.style.use('classic')
    
    nells_intp = 1000
    sigma_pix = 0.20
    ell_pix = 2. * np.pi / np.deg2rad(sigma_pix)
    ells_intp = np.logspace(np.log10(20.), np.log10(10. * ell_pix), nells_intp)
    ell_min = 20
    ell_max = int(10 * ell_pix)
    nells = int(ell_max - ell_min + 1)
    ells = np.linspace(ell_min, ell_max, nells)
    #ells = ells_intp
    #print ells
    WF1 = WindowFunctions(sigma=sigma_pix, l_min=ells_intp.min(), l_max=ells_intp.max(), number_nodes=nells_intp, ncpus=1)
    WF2 = WindowFunctionsSimple(sigma=sigma_pix)
    
    w0_1 = WF1.getWindowFunction(ells_intp, 0)    
    w4_1 = WF1.getWindowFunction(ells_intp, 4)
    w8_1 = WF1.getWindowFunction(ells_intp, 8)
    
    w0_2 = WF2.getWindowFunction(ells, 0)
    w4_2 = WF2.getWindowFunction(ells, 4)
    w8_2 = WF2.getWindowFunction(ells, 8)
    
    #print (w0_1 - w0_2) / w0_2
    #print (w4_1 - w4_2) / w4_2
    #print (w8_1 - w8_2) / w8_2    
    
    print w0_2
    print w4_2
    print w8_2    
    
    # this should reproduce Hu & White's Fig. 2:
    plt.plot(ells_intp / ell_pix, w0_1, ls='-', color='black', label=r'$n=0$')
    plt.plot(ells_intp / ell_pix, w4_1, ls='-', color='black', label=r'$n=4$')
    plt.plot(ells_intp / ell_pix, w8_1, ls='-', color='black', label=r'$n=8$')
    plt.plot(ells / ell_pix, w0_2, ls='--', color='red', label=r'$n=0$')
    plt.plot(ells / ell_pix, w4_2, ls='--', color='red', label=r'$n=4$')
    plt.plot(ells / ell_pix, w8_2, ls='--', color='red', label=r'$n=8$')
            
    plt.loglog()
    plt.xlim([1e-2, 10.])
    plt.ylim([1e-4, 1.2])
    plt.xlabel(r'$\ell / \ell_{pix}$')
    plt.ylabel(r'$ |w_n(\ell)| $')
    plt.legend(loc='best', frameon=False)
    plt.show()