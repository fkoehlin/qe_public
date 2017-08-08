#!/usr/bin/env python
# encoding: UTF8
# Code for estimating maximum likelihood band powers of the lensing power spectrum
# following the "quadratic estimator" method by Hu & White 2001 (ApJ, 554, 67) 
# implementation by Fabian Koehlinger

#import sys
#import os
import numpy as np
#import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.special as special
import multiprocessing
#import itertools
import data_reduction as dr
import window_functions as wf
# this is better for timing than "time"
#import time
from timeit import default_timer as timer

# maximum number of substeps in integration routine:
#global LIMIT
LIMIT = 500

# kind of interpolation ('linear' is the only feasible option in current approach!!!)
#global KIND_INTERPOL
KIND_INTERPOL = 'linear'

def unwrap_calculate(arg):
    """ Helper function for multiprocessing; converts `f([1,2])` to `f(1,2)` call """
    
    return SignalMatrix.calculate(*arg)

class SignalMatrix(object):
    """ 
    
    Unfortunately, depends on passing of GLOBAL function method "moment"; can't be defined within this class
    due to the following pickling error:
    
    "PicklingError: Can't pickle <type 'instancemethod'>: attribute lookup __builtin__.instancemethod failed"        
    
    Args:
    
        band_min:   lower l-bound for integration (if integrate=True) 
                    l-value (if integrate=False)
        band_max:   upper l-bound for integration (if integrate=True) 
                    default value as dummy (if integrate=False)
        band:       either 'EE', 'EB' or 'BB' depending on type of power spectrum of interest
        integrate:  boolean, if True: matrix = C,alpha
                             if False: matrix = C,l
        ncpus:     number of CPUs code is allowed to use ("0" means ALL!)
    """
    
    def __init__(self, band_min, band_max=0., band='EE', sigma=0.1, integrate=True, ncpus=0):
        
        #number_nodes=1e4
        self.band_min = band_min
        self.band_max = band_max
        self.band = band
        self.integrate = integrate
        self.ncpus = int(ncpus)
        
        if integrate:
            '''
            l_min = self.band_min - 1 
            l_max = self.band_max + 1    
        
            WF = wf.WindowFunctions(sigma=sigma, l_min=l_min, l_max=l_max, number_nodes=number_nodes, ncpus=self.ncpus)
        
            self.l_nodes = WF.l_nodes
            
            self.w0 = WF.getArray(n=0)
            self.w4 = WF.getArray(n=4)
            '''
            # since we're only summing, pass arrays instead of functions
            WF = wf.WindowFunctionsSimple(sigma=sigma)
            nells = int(self.band_max - self.band_min + 1)
            self.ells = np.linspace(self.band_min, self.band_max, nells)
            self.w0 = WF.getWindowFunction(self.ells, n=0)
            self.w4 = WF.getWindowFunction(self.ells, n=4)
        
        else:
            '''
            WF = wf.WindowFunctionsSimple(sigma=sigma)
            # really stupid but I can't do anything about it because of pickling error (while trying to pass a function...)
            # although I only need l=band_min, I have to pass some more l in order to interpolate...
            self.l_nodes = np.array([band_min - 1, band_min, band_min + 1])
            # don't use np.zeros_like(self.l_nodes) because then w0 & w4 will be integer-arrays!
            w0 = np.zeros(len(self.l_nodes))
            w4 = np.zeros_like(w0)
            for index_l, var_l in enumerate(self.l_nodes):
                w0[index_l] = WF.getWindowFunction(l=var_l, n=0)
                w4[index_l] = WF.getWindowFunction(l=var_l, n=4)
            self.w0 = w0
            self.w4 = w4
            '''
            WF = wf.WindowFunctionsSimple(sigma=sigma)
            self.ell = self.band_min
            self.w0 = WF.getWindowFunction(self.ell, n=0)
            self.w4 = WF.getWindowFunction(self.ell, n=4)
            
    def getSignalMatrix(self, x, y):
        """
        x, y are supposed to be coordinates of (square) full 2D-field and should 
        be provided in radians!
        
        This is brute-force calculation!!!
        """
        
        r, phi = dr.get_distances_and_angles(x, y)
    
        r_flat = r.flatten()
        phi_flat = phi.flatten()

        # create position vector (r,phi):
        pos = np.column_stack((r_flat, phi_flat))
        #t0 = timer()
        print 'Total number of (r, phi):', r_flat.size

        # indices and calculate 11-element etc. separately (i.e. reverse pixel indices with shear-component indices!)
        # be nice and don't steal all CPUs by default...
        cpus_available = multiprocessing.cpu_count()
        if self.ncpus == 0:
            take_cpus = cpus_available
        elif self.ncpus > cpus_available:
            take_cpus = cpus_available
        else:
            take_cpus = self.ncpus
        pool = multiprocessing.Pool(processes=take_cpus)
        #Try to use pathos...didn't work...
        #pool = PathosPool(processes=cpus)
        print 'Started with calculation of matrix elements on {:} cores.'.format(take_cpus)
        t0 = timer()
        # make this independent of GLOBAL variables/functions (i.e. band_min, band_max, band, moment) eventually:
        result = np.asarray(pool.map(unwrap_calculate, zip([self] * len(pos[:, 0]), zip(pos[:, 0], pos[:, 1]))))
        #result = np.asarray(pool.map(unwrap_calculate, itertools.izip(itertools.repeat(self), zip(unique_pos[:,0], unique_pos[:,1]))))
        pool.close()
        print 'Done. Time: {:.2f}s.'.format(timer() - t0)
        matrix_11 = result[:, 0]
        matrix_12 = result[:, 1]
        matrix_22 = result[:, 2]
    
        # put block elements together:
        block_11 = np.asmatrix(matrix_11.reshape(r.shape))
        block_22 = np.asmatrix(matrix_22.reshape(r.shape))
        block_12 = np.asmatrix(matrix_12.reshape(r.shape))
    
        print 'Shape of r (and phi):', r.shape, phi.shape
        print 'Shape of one block in matrix:', np.shape(block_11)
        # block_12 = block_21
        '''
        # TODO: Write out in FITS & solve overwriting!        
        # output for testing
        # file will be overwritten for more than 1 ell- and redshift bin!                    
        # create dummy indices:
        x = range(len(r))
        xx, yy = np.meshgrid(x, x)
        header = 'x, y, dist, angle, block_11, block_12, block_22'        
        savedata = np.column_stack((xx.flatten(), yy.flatten(), r.flatten(), phi.flatten(), np.asarray(block_11).flatten(), np.asarray(block_12).flatten(), np.asarray(block_22).flatten()))
        path = 'cache/' #'/net/delft/data2/Sandbox/neutrinos/common/'
        np.savetxt(path + 'blocks_SLOW.dat', savedata, header=header)
        '''        
        # block_12 = block_21 !!!
        matrix = np.bmat('block_11, block_12; block_12, block_22')
        print 'Shape of matrix:', np.shape(matrix)
    
        return matrix
        
    def getSignalMatrixFAST(self, x, y):
        """
        x, y are supposed to be coordinates of (square) full 2D-field and should 
        be provided in radians!
        
        We use some quick & dirty algorithm for finding unique elements and distributing them then again into full matrix.
        """
        
        r, phi = dr.get_distances_and_angles(x, y)
    
        r_flat = r.flatten()
        phi_flat = phi.flatten()

        # create position vector (r,phi):
        pos = np.column_stack((r_flat, phi_flat))
        #t0 = timer()
        # find unique elements in that position vector:
        unique_pos, index_unique = self.unique_rows(pos)
        #print 'Time solution:', timer()-start
        #print unique_pos, unique_pos.shape
        #print index_unique
        #print 'Unique positions:', unique_pos
        print 'Total number of (r, phi):', r_flat.size
        print 'Number of unique (r, phi):', index_unique.size

        # indices and calculate 11-element etc. separately (i.e. reverse pixel indices with shear-component indices!)
        # be nice and don't steal all CPUs by default...
        cpus_available = multiprocessing.cpu_count()
        if self.ncpus == 0:
            take_cpus = cpus_available
        elif self.ncpus > cpus_available:
            take_cpus = cpus_available
        else:
            take_cpus = self.ncpus
   
        pool = multiprocessing.Pool(processes=take_cpus)
        #Try to use pathos...didn't work...
        #pool = PathosPool(processes=cpus)
        print 'Started with calculation of matrix elements on {:} cores.'.format(take_cpus)
        t0 = timer()
        # make this independent of GLOBAL variables/functions (i.e. band_min, band_max, band, moment) eventually:
        result = np.asarray(pool.map(unwrap_calculate, zip([self] * len(unique_pos[:, 0]), zip(unique_pos[:, 0], unique_pos[:, 1]))))
        #result = np.asarray(pool.map(unwrap_calculate, itertools.izip(itertools.repeat(self), zip(unique_pos[:,0], unique_pos[:,1]))))
        pool.close()
        print 'Done. Time: {:.2f}s.'.format(timer() - t0)
        unique_elements_11 = result[:, 0]
        unique_elements_12 = result[:, 1]
        unique_elements_22 = result[:, 2]
    
        # look-up of unique elements in order to construct the Signal matrix
        # loop over all elements in (r, phi), restore matrix structure later!
        # for sure, that can be done smarter and faster...
        # or at least in parallel?!
        matrix_11 = np.zeros_like(r_flat)
        matrix_22 = np.zeros_like(r_flat)
        matrix_12 = np.zeros_like(r_flat)
        
        t0 = timer()
        '''
        # this is a really stupid bottleneck...
        # maybe use sets...
        for i in range(r_flat.size):
            for j in range(index_unique.size):
                if r_flat[i] == unique_pos[j,0] and phi_flat[i] == unique_pos[j,1]:
                    matrix_11[i] = unique_elements_11[j]
                    matrix_22[i] = unique_elements_22[j]
                    matrix_12[i] = unique_elements_12[j]
        '''
        # this is MUCH faster and should yield same output as nested loop from above:
        for i in range(index_unique.size):
            k = np.where(unique_pos[i, 0] == r_flat)[0]
            l = np.where(unique_pos[i, 1] == phi_flat)[0]
            j = np.intersect1d(k, l)
            matrix_11[j] = unique_elements_11[i]
            matrix_22[j] = unique_elements_22[i]
            matrix_12[j] = unique_elements_12[i]
        #'''
        print 'Time for filling matrix: {:.2f}s.'.format(timer() - t0)
    
        # put block elements together:
        block_11 = np.asmatrix(matrix_11.reshape(r.shape))
        block_22 = np.asmatrix(matrix_22.reshape(r.shape))
        block_12 = np.asmatrix(matrix_12.reshape(r.shape))
    
        print 'Shape of r (and phi):', r.shape, phi.shape
        print 'Shape of one block in matrix:', np.shape(block_11)
        # block_12 = block_21
        '''        
        # TODO: Write out in FITS & solve overwriting!        
        # output for testing
        # file will be overwritten for more than 1 ell- and redshift bin!                    
        # create dummy indices:        
        x = range(len(r))
        xx, yy = np.meshgrid(x, x)
        header = 'x, y, dist, angle, block_11, block_12, block_22'
        savedata = np.column_stack((xx.flatten(), yy.flatten(), r.flatten(), phi.flatten(), np.asarray(block_11).flatten(), np.asarray(block_12).flatten(), np.asarray(block_22).flatten()))
        path = 'cache/' #'/net/delft/data2/Sandbox/neutrinos/common/'
        np.savetxt(path + 'blocks_FAST.dat', savedata, header=header)
        '''        
        # block_12 = block_21 !!!
        matrix = np.bmat('block_11, block_12; block_12, block_22')
        print 'Shape of matrix:', np.shape(matrix)
    
        return matrix     

    def calculate(self, position):
        """
        Functions "matrixelement_<ij>_<band>" depend on GLOBAL passing of Window function, i.e. "moment"
        """
    
        r = position[0]
        phi = position[1]
        #print 'r, phi', r, phi
        
        # this is still suboptimal since interpolation is created for each position per process...
        # but it avoids the pickling issue (with standard multiprocessing...)
        # it's really inefficient if there's no integration required...
        #'''
        #start_interpol = timer()
        
        #self.w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind=KIND_INTERPOL)
        #self.w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind=KIND_INTERPOL)
        
        #print 'Time for interpolation:', timer()-start_interpol
        #'''
        
        if self.integrate:
            
            # instead of evaluating an integral, we are summing now over all multipoles "ell" in the corresponding bin.
            # all defined with integers now!
            '''
            ells_min = self.band_min
            ells_max = self.band_max
            nells = ells_max-ells_min+1
            # these are integer l-values over which we will take the sum used in the convolution with the band window matrix
            self.ells = np.linspace(ells_min, ells_max, nells)
            '''
            if self.band == 'EE':
                unique_elements_11 = self.matrixelement_11_EE(r, phi)
                unique_elements_12 = self.matrixelement_12_EE(r, phi)
                unique_elements_22 = self.matrixelement_22_EE(r, phi)
            elif self.band == 'BB':
                unique_elements_11 = self.matrixelement_11_BB(r, phi)
                unique_elements_12 = self.matrixelement_12_BB(r, phi)
                unique_elements_22 = self.matrixelement_22_BB(r, phi)
            elif self.band == 'EB':
                unique_elements_11 = self.matrixelement_11_EB(r, phi)
                unique_elements_12 = self.matrixelement_12_EB(r, phi)
                unique_elements_22 = self.matrixelement_22_EB(r, phi)
            else:
                print 'No valid mode (EE, BB, EB) specified.'
                exit()
        else:
            #ell = self.band_min
            #print 'no integration, l={:}'.format(l)
            if self.band == 'EE':
                unique_elements_11 = self.integrand_element_11_EE(self.ell, r, phi)
                unique_elements_12 = self.integrand_element_12_EE(self.ell, r, phi)
                unique_elements_22 = self.integrand_element_22_EE(self.ell, r, phi)
            elif self.band == 'BB':
                unique_elements_11 = self.integrand_element_11_BB(self.ell, r, phi)
                unique_elements_12 = self.integrand_element_12_BB(self.ell, r, phi)
                unique_elements_22 = self.integrand_element_22_BB(self.ell, r, phi)
            elif self.band == 'EB':
                unique_elements_11 = self.integrand_element_11_EB(self.ell, r, phi)
                unique_elements_12 = self.integrand_element_12_EB(self.ell, r, phi)
                unique_elements_22 = self.integrand_element_22_EB(self.ell, r, phi)
            else:
                print 'No valid mode (EE, BB, EB) specified.'
                exit()
                
        return unique_elements_11, unique_elements_12, unique_elements_22

    def unique_rows(self, data):
        """ convenience function to find unique elements in an array; there seem to be issues with float precision though"""
        b = np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize*data.shape[1])))
        _, idx = np.unique(b, return_index=True)
    
        return data[idx], idx
    
    def matrixelement_11_EE(self, r, phi):
        
        #element_11, del11 = integrate.quad(lambda l: self.integrand_element_11_EE(l, r, phi), self.band_min, self.band_max, limit=LIMIT)    
        element_11 = np.sum(self.integrand_element_11_EE(self.ells, r, phi), axis=0)
        
        return element_11

    def matrixelement_12_EE(self, r, phi):

        #element_12, del12 = integrate.quad(lambda l: self.integrand_element_12_EE(l, r, phi), self.band_min, self.band_max, limit=LIMIT)
        element_12 = np.sum(self.integrand_element_12_EE(self.ells, r, phi), axis=0)

        return element_12

    def matrixelement_22_EE(self, r, phi):

        #element_22, del22 = integrate.quad(lambda l: self.integrand_element_22_EE(l, r, phi), self.band_min, self.band_max, limit=LIMIT)
        element_22 = np.sum(self.integrand_element_22_EE(self.ells, r, phi), axis=0)

        return element_22

    def matrixelement_11_BB(self, r, phi):

        #element_11, del11 = integrate.quad(lambda l: self.integrand_element_11_BB(l, r, phi), self.band_min, self.band_max, limit=LIMIT) 
        element_11 = np.sum(self.integrand_element_11_BB(self.ells, r, phi), axis=0)
        
        return element_11

    def matrixelement_12_BB(self, r, phi):

        #element_12, del12 = integrate.quad(lambda l: self.integrand_element_12_BB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)
        element_12 = np.sum(self.integrand_element_12_BB(self.ells, r, phi), axis=0)
            
        return element_12

    def matrixelement_22_BB(self, r, phi):

        #element_22, del22 = integrate.quad(lambda l: self.integrand_element_22_BB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)
        element_22 = np.sum(self.integrand_element_22_BB(self.ells, r, phi), axis=0)

        return element_22

    def matrixelement_11_EB(self, r, phi):

        #element_11, del11 = integrate.quad(lambda l: self.integrand_element_11_EB(l, r, phi), self.band_min, self.band_max, limit=LIMIT) 
        element_11 = np.sum(self.integrand_element_11_EB(self.ells, r, phi), axis=0)
        
        return element_11

    def matrixelement_12_EB(self, r, phi):

        #element_12, del12 = integrate.quad(lambda l: self.integrand_element_12_EB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)
        element_12 = np.sum(self.integrand_element_12_EB(self.ells, r, phi), axis=0)

        return element_12

    def matrixelement_22_EB(self, r, phi):

        #element_22, del22 = integrate.quad(lambda l: self.integrand_element_22_EB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)
        element_22 = np.sum(self.integrand_element_22_EB(self.ells, r, phi), axis=0)

        return element_22
    
    def matrixelement_11_EE_BB(self, r, phi):
        
        #element_11, del11 = integrate.quad(lambda l: self.integrand_element_11_EE_BB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)    
        element_11 = np.sum(self.integrand_element_11_EE_BB(self.ells, r, phi), axis=0)
        
        return element_11
        
    def matrixelement_12_EE_BB(self, r, phi):
        
        #element_12, del12 = integrate.quad(lambda l: self.integrand_element_12_EE_BB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)    
        element_12 = np.sum(self.integrand_element_12_EE_BB(self.ells, r, phi), axis=0)
        
        return element_12
    
    def matrixelement_22_EE_BB(self, r, phi):
        
        #element_22, del22 = integrate.quad(lambda l: self.integrand_element_22_EE_BB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)    
        element_22 = np.sum(self.integrand_element_22_EE_BB(self.ells, r, phi), axis=0)
        
        return element_22

    def matrixelement_11_EE_BB_EB(self, r, phi):
        
        #element_11, del11 = integrate.quad(lambda l: self.integrand_element_11_EE_BB_EB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)    
        element_11 = np.sum(self.integrand_element_11_EE_BB_EB(self.ells, r, phi), axis=0)
        
        return element_11
        
    def matrixelement_12_EE_BB_EB(self, r, phi):
        
        #element_12, del12 = integrate.quad(lambda l: self.integrand_element_12_EE_BB_EB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)    
        element_12 = np.sum(self.integrand_element_12_EE_BB_EB(self.ells, r, phi), axis=0)
        
        return element_12
    
    def matrixelement_22_EE_BB_EB(self, r, phi):
        
        #element_22, del22 = integrate.quad(lambda l: self.integrand_element_22_EE_BB_EB(l, r, phi), self.band_min, self.band_max, limit=LIMIT)    
        element_22 = np.sum(self.integrand_element_22_EE_BB_EB(self.ells, r, phi), axis=0)
        
        return element_22        
            
    def integrand_element_11_EE(self, l, r, phi):
    
        #print 'l:', l
        
        # this is not optimal, but somehow "pickle" can't handle the passing of an object...
        # takes also much longer (which makes sense, since the interpolation is constructed at each step of the integration...)
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_11_EE(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand

    def integrand_element_12_EE(self, l, r, phi):
        
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_12_EE(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand
    
    def integrand_element_22_EE(self, l, r, phi):
        
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_22_EE(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand

    def integrand_element_11_BB(self, l, r, phi):
        
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_11_BB(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand

    def integrand_element_12_BB(self, l, r, phi):
        
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_12_BB(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand
    
    def integrand_element_22_BB(self, l, r, phi):
        
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_22_BB(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand

    def integrand_element_11_EB(self, l, r, phi):
        
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_11_EB(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand

    def integrand_element_12_EB(self, l, r, phi):
        
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_12_EB(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand
    
    def integrand_element_22_EB(self, l, r, phi):
    
        #w0_int = interpolate.interp1d(self.l_nodes, self.w0, kind='cubic')
        #w4_int = interpolate.interp1d(self.l_nodes, self.w4, kind='cubic')
        
        I, Q = self.getIQ_22_EB(l * r, phi)
        #w0 = self.w0_int(l) #self.moment(l, 0)
        #w4 = self.w4_int(l) #self.moment(l, 4)
    
        integrand = 0.5 / (l + 1.) * (self.w0 * I + 0.5 * self.w4 * Q)
    
        return integrand
    
    '''
    deprecated:
    def integrand_element_11_EE_BB(self, l, r, phi):
        
        func = self.integrand_element_11_EE(l, r, phi)+self.integrand_element_11_BB(l, r, phi)        
        
        return func
    
    def integrand_element_12_EE_BB(self, l, r, phi):
        
        func = self.integrand_element_12_EE(l, r, phi)+self.integrand_element_12_BB(l, r, phi)        
        
        return func    
        
    def integrand_element_22_EE_BB(self, l, r, phi):
        
        func = self.integrand_element_22_EE(l, r, phi)+self.integrand_element_22_BB(l, r, phi)        
        
        return func    
    
    def integrand_element_11_EE_BB_EB(self, l, r, phi):
        
        func = self.integrand_element_11_EE(l, r, phi)+self.integrand_element_11_BB(l, r, phi)++self.integrand_element_11_EB(l, r, phi)        
        
        return func
    
    def integrand_element_12_EE_BB_EB(self, l, r, phi):
        
        func = self.integrand_element_12_EE(l, r, phi)+self.integrand_element_12_BB(l, r, phi)+self.integrand_element_12_EB(l, r, phi)       
        
        return func    
        
    def integrand_element_22_EE_BB_EB(self, l, r, phi):
        
        func = self.integrand_element_22_EE(l, r, phi)+self.integrand_element_22_BB(l, r, phi)+self.integrand_element_22_EB(l, r, phi)       
        
        return func
    '''
    
    def getIQ_11_EE(self, x, phi):
    
        J0 = special.j0(x)
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        c4 = np.cos(4*phi)
        c8 = np.cos(8*phi)

        I = J0+c4*J4
        Q = J0+2*c4*J4+c8*J8
    
        return I, Q

    def getIQ_12_EE(self, x, phi):
    
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        s4 = np.sin(4*phi)
        s8 = np.sin(8*phi)
    
        I = s4*J4
        Q = s8*J8
    
        return I, Q

    def getIQ_22_EE(self, x, phi):
    
        J0 = special.j0(x)
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        c4 = np.cos(4*phi)
        c8 = np.cos(8*phi)
    
        I = J0-c4*J4
        Q = -J0+2*c4*J4-c8*J8

        return I, Q

    def getIQ_11_BB(self, x, phi):
    
        J0 = special.j0(x)
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        c4 = np.cos(4*phi)
        c8 = np.cos(8*phi)

        I = J0-c4*J4
        Q = -J0+2*c4*J4-c8*J8
    
        return I, Q

    def getIQ_12_BB(self, x, phi):
    
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        s4 = np.sin(4*phi)
        s8 = np.sin(8*phi)
    
        I = -s4*J4
        Q = -s8*J8
    
        return I, Q

    def getIQ_22_BB(self, x, phi):
    
        J0 = special.j0(x)
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        c4 = np.cos(4*phi)
        c8 = np.cos(8*phi)
    
        I = J0+c4*J4
        Q = J0+2*c4*J4+c8*J8

        return I, Q

    def getIQ_11_EB(self, x, phi):
    
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        s4 = np.sin(4*phi)
        s8 = np.sin(8*phi)

        I = -2*s4*J4
        Q = -2*s8*J8
    
        return I, Q

    def getIQ_12_EB(self, x, phi):
    
        J0 = special.j0(x)
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        c4 = np.cos(4*phi)
        c8 = np.cos(8*phi)
    
        I = 2*c4*J4
        Q = 2*J0*+2*c8*J8
    
        return I, Q

    def getIQ_22_EB(self, x, phi):
        
        J4 = special.jn(4, x)
        J8 = special.jn(8, x)
        s4 = np.sin(4*phi)
        s8 = np.sin(8*phi)
        
        I = 2*s4*J4
        Q = 2*s8*J8

        return I, Q
