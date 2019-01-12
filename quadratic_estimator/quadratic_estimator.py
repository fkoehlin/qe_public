#!/usr/bin/python
# encoding: UTF8

# Code for estimating maximum likelihood band powers of the lensing power spectrum
# following the "quadratic estimator" method by Hu & White 2001 (ApJ, 554, 67)
# implementation by Fabian Koehlinger and Wessel Valkenburg

# This is the main script. Call signature: python /path/to/quadratic_estimator.py </path/to/input_parameters.ini>

import os
import gc
import sys
import numpy as np
import h5py
import scipy.interpolate as interpolate
import scipy.sparse as sparse
from scipy.linalg import inv, solve
import multiprocessing
import data_reduction as dr
import signal_matrix as sm
# this is better for timing than "time"
#import time
from timeit import default_timer as timer

def print_dot():

    sys.stdout.write('.')
    sys.stdout.flush()

    return

# single step for parallelized loop
# in cached version I could also directly write out each transposed matrix and so on...
def single_step_compute_inv_covariance_times_deriv_matrices(index):

    #t0 = timer()

    success = False

    # weird bracket notation '[()]' necessary in order to restore sparse matrix
    if ini.use_sparse_matrices:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_index{:03d}.npy'.format(index))[()].toarray()
    else:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_index{:03d}.npy'.format(index))

    # Loading is much faster...
    if ini.do_not_invert_that_matrix:
        covariance = np.load('cache/' + 'covariance.npy')
        result = solve(covariance, deriv_matrix)
    else:
        inv_covariance = np.load('cache/' + 'inv_covariance.npy')
        result = inv_covariance.dot(deriv_matrix)

    # save files directly to cache:
    if ini.use_sparse_matrices:
        matrix = sparse.bsr_matrix(result)
        matrix_T = sparse.bsr_matrix(np.transpose(result))
    else:
        matrix = result
        matrix_T = np.transpose(result)

    np.save('cache/' + 'inv_covariance_times_deriv_matrix_index{:03d}'.format(index), matrix)
    np.save('cache/' + 'inv_covariance_times_deriv_matrix_transposed_index{:03d}'.format(index), matrix_T)

    #print 'Multiplication done in {:.2f}s.'.format(timer()-t0)
    #print_dot()

    success = True

    return success

# single step for parallelized loop
def single_step_compute_inv_covariance_times_deriv_matrices_ell(index):

    #t0 = timer()

    success = False

    if ini.use_sparse_matrices:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_ell_index_lin{:03d}.npy'.format(index))[()].toarray()
    else:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_ell_index_lin{:03d}.npy'.format(index))

    if ini.do_not_invert_that_matrix:
        covariance = np.load('cache/covariance.npy')
        result = solve(covariance, deriv_matrix)
    else:
        inv_covariance = np.load('cache/' + 'inv_covariance.npy')
        result = inv_covariance.dot(deriv_matrix)

    if ini.use_sparse_matrices:
        matrix = sparse.bsr_matrix(np.transpose(result))
    else:
        matrix = np.transpose(result)
    # save files directly to cache:
    # only transpose is needed!
    np.save('cache/' + 'inv_covariance_times_deriv_matrix_ell_transposed_index_ell{:03d}'.format(index), matrix)

    #print 'Multiplication done in {:.2f}s.'.format(timer()-t0)
    #print_dot()

    success = True

    return success

# WV New bottleneck equation:
# precompute C^-1_ij (C,beta)_jk
def compute_inv_covariance_times_deriv_matrices(dim, ncpus=1, for_band_window=False):
    # WV NEW:
    t0 = timer()

    #dataDim = inv_covariance.shape[0]
    #dim = len(deriv_matrices)

    # Using multiprocessing, prepare the iterable array with inv_covariance (whose dot method we need) and deriv[i]:
    #t0_mem = timer()
    #print 'Preparing memory for multiprocessing pool.'

    # weird, but creating a local copy in memory seems to speed up the parallel process by a factor of three.
    #inv_covariance_copy = np.asarray(inv_covariance)
    #deriv_matrices_copy = np.asarray(deriv_matrices)

    #print 'Prepared memory for multiprocessing pool in {:.2f} seconds.'.format(timer()- t0_mem)

    #print 'Shape of poolmem as list:', np.shape(collect_poolmem)

    # this does NOT work or improve things at all...
    # since it's trying to make a copy of that list... whereas a list should hopefully just store pointers...
    #poolmem_alt = np.asarray(poolmem)

    #print 'Shape of poolmem as np.array:', np.shape(poolmem)

    # be nice and don't steal all CPUs by default...
    cpus_available = multiprocessing.cpu_count()
    if ncpus == 0:
        take_cpus = cpus_available
    elif (ncpus > cpus_available):
        take_cpus = cpus_available
    else:
        take_cpus = ncpus

    # set pool size: not larger than dim
    pool_size = take_cpus
    if (take_cpus > dim):
        pool_size = dim

    print 'Running on {:} processes...'.format(pool_size)

    index_array = range(dim)

    # Go compute!
    pool = multiprocessing.Pool(processes = pool_size)
    if for_band_window:
        #index_array = index_map
        loops_were_success = pool.map(single_step_compute_inv_covariance_times_deriv_matrices_ell, index_array)
    else:
        #index_array = range(dim)
        loops_were_success = pool.map(single_step_compute_inv_covariance_times_deriv_matrices, index_array)
    pool.close()
    #print ''

    if np.asarray(loops_were_success).all():
        success = True
    else:
        success = False

    print 'Done pool computation in {:.2f}s.'.format(timer()-t0)

    return success

# single step for parallelized loop
def single_step_get_fisher_matrix(index_pair):

    #t0 = timer()

    if ini.use_sparse_matrices:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_index{:03d}.npy'.format(index_pair[0]))[()].toarray()
        inv_covariance_times_derivative_B_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_index{:03d}.npy'.format(index_pair[1]))[()].toarray()
    else:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_index{:03d}.npy'.format(index_pair[0]))
        inv_covariance_times_derivative_B_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_index{:03d}.npy'.format(index_pair[1]))

    total_sum = np.sum(inv_covariance_times_derivative_A * inv_covariance_times_derivative_B_transposed)
    # NOTE: second matrix is transposed, and this is an ARRAY multiplication (by element), NOT MATRIX.

    #print 'Fisher( X, X ) in', ((timer() - t0)), 'seconds.'
    #print totSum
    # I should make that optional...
    #print_dot()

    #dt = timer()-t0

    return total_sum

def get_fisher_matrix(dim, ncpus=1):

    t0 = timer()

    fisher = np.zeros((dim, dim))

    # prepare the pool's size and memory:
    nfisher = (dim * (dim + 1)) / 2

    # be nice and don't steal all CPUs by default...
    cpus_available = multiprocessing.cpu_count()
    if ncpus == 0:
        take_cpus = cpus_available
    elif (ncpus > cpus_available):
        take_cpus = cpus_available
    else:
        take_cpus = ncpus

    pool_size = take_cpus
    if (take_cpus > nfisher):
        pool_size = nfisher

    print 'Pool size for last step in Fisher computation: {:}'.format(pool_size)

    #t0_mem = timer()
    #print 'Preparing for multiprocessing.'

    # Now new approach:
    # create only an index_array (for loading needed matrices directly in parallel function)
    index_array = []
    for alpha in xrange(dim):
        for beta in xrange(alpha + 1):
            index_array.append([alpha, beta])

    #print 'Prepared memory for multiprocessing pool in {:.2f} seconds.'.format(timer()-t0_mem)

    print 'Collecting Fisher matrix...'

    pool = multiprocessing.Pool(processes = pool_size)
    fisher_as_list = pool.map(single_step_get_fisher_matrix, index_array)
    pool.close()

    #print ''

    index = 0
    for alpha in xrange(dim):
        for beta in xrange(alpha + 1): # 0 < b <= alpha.
            fisher[alpha, beta] = fisher_as_list[index]
            fisher[beta, alpha] = fisher[alpha, beta]
            index += 1

    dt = timer()-t0

    # IMPORTANT: the Fisher matrix calculated here, is actually 2 times larger compared to the definition of the Fisher matrix in Eq. (20) of Hu & White 2001
    # I'm taking care of that factor in every subsequent step for stability reasons of inversion of this matrix!
    return fisher, dt

def single_step_estimate_band_powers(index):

    inv_covariance_times_data_matrix = np.load('cache/' + 'inv_covariance_times_data_matrix.npy')

    if ini.use_sparse_matrices:
        inv_covariance_times_derivative_beta_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_index{:03d}.npy'.format(index))[()].toarray()
    else:
        inv_covariance_times_derivative_beta_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_index{:03d}.npy'.format(index))

    sum_trace = 0.
    sum_trace = np.sum(inv_covariance_times_data_matrix * inv_covariance_times_derivative_beta_transposed)
    sum_trace -= np.trace(inv_covariance_times_derivative_beta_transposed)

    # make this optional
    # print_dot()

    return sum_trace

# bottleneck equation!!!
def estimate_band_powers(data_vector, covariance, fisher, convergence_parameter=1., ncpus=1):
    t0 = timer()
    dim = fisher.shape[0]
    #print dim
    delta_powers = np.zeros((dim))
    # WV NEW:
    # Compute C^-1_{ij} d_j d_k
    dim_data = len(data_vector)
    inv_covariance_times_data_matrix = np.zeros((dim_data, dim_data))
    t0_step1 = timer()

    if ini.do_not_invert_that_matrix:
        inv_covariance_times_data_vector = solve(covariance, data_vector)
    else:
        # we expect that the inv_covariance is passed as covariance!!!
        inv_covariance_times_data_vector = covariance.dot(data_vector)

    for i in xrange(dim_data): # no need for two loops, can be done as soon as we have CInverseTimesdata_vector[i]
        for j in xrange(dim_data):
            inv_covariance_times_data_matrix[i, j] = inv_covariance_times_data_vector[i] * data_vector[j]
    np.save('cache/' + 'inv_covariance_times_data_matrix', inv_covariance_times_data_matrix)
    print 'First step in band powers done in {:.2f}s.'.format(timer() - t0_step1)

    t0_step2 = timer()

    #inv_covarianceTimesDerivMatricesTransposed_copy = np.asarray(inv_covarianceTimesDerivMatricesTransposed)
    #poolmem = []
    index_array = range(dim)

    # be nice and don't steal all CPUs by default...
    cpus_available = multiprocessing.cpu_count()
    if ncpus == 0:
        take_cpus = cpus_available
    elif (ncpus > cpus_available):
        take_cpus = cpus_available
    else:
        take_cpus = ncpus

    pool_size = take_cpus
    if (take_cpus > dim):
        pool_size = dim

    print 'Computing vec_b = Tr[ (d dT - C) ( C^-1 C,b C^-1 )] = Tr[ (C^-1 d dT) (C^-1 C,b) ] - Tr[ (C^-1 C,b) ]...'
    pool = multiprocessing.Pool(processes = pool_size)
    trace_sum_vector = pool.map(single_step_estimate_band_powers, index_array)
    pool.close()
    #print ''

    # IMPORTANT: the Fisher matrix calculated here is two times larger than the definition in Eq. (20) of Hu & White 2001
    # that is why the factor 0.5 in Eq. (19) of Hu & White (2001), implemented below, cancels with (0.5)^(-1) from correct treatment of the inverse of the Fisher matrix
    #delta_powers = convergence_parameter * inv_fisher.dot(trace_sum_vector)
    if ini.do_not_invert_that_matrix:
        delta_powers = convergence_parameter * solve(fisher, trace_sum_vector)
    else:
        # we expect that the inv_fisher is passed as fisher!!!
        delta_powers = convergence_parameter * fisher.dot(trace_sum_vector)

    print 'Second step in band powers done in {:.2f}s.'.format(timer() - t0_step2)

    dt = timer() - t0

    return delta_powers, dt

def get_window_matrix(fisher, trace):

    t0 = timer()

    # IMPORTANT: the Fisher matrix calculated here is two times larger than the definition in Eq. (20) of Hu & White 2001
    # that is why the factor 0.5 in Eq. (21) of Lin et al. (2012), implemented below, cancels with (0.5)^(-1) from correct treatment of the inverse of the Fisher matrix

    if ini.do_not_invert_that_matrix:
        window = solve(fisher, trace)
    else:
        # we expect that the inv_fisher is passed as fisher!!!
        window = fisher.dot(trace)

    dt = timer() - t0

    return window, dt

# I could merge this with "get_Fisher_matrix" by including a check/flag for symmetric matrices...
def get_trace_matrix(dim_alpha, dim_ell, ncpus=1):

    t0 = timer()

    print 'dim(alpha)={:}, dim(ell)={:}'.format(dim_alpha, dim_ell)

    trace = np.zeros((dim_alpha, dim_ell))

    # prepare the pool's size and memory:
    ntrace = dim_alpha * dim_ell

    # be nice and don't steal all CPUs by default...
    cpus_available = multiprocessing.cpu_count()
    if ncpus == 0:
        take_cpus = cpus_available
    elif (ncpus > cpus_available):
        take_cpus = cpus_available
    else:
        take_cpus = ncpus

    pool_size = take_cpus
    if (take_cpus > ntrace):
        pool_size = ntrace

    print 'Pool size for last step in Fisher computation: {:}'.format(pool_size)

    #t0_mem = timer()
    # Now new approach:
    # create only an index_array (for loading needed matrices directly in parallel function)
    index_array = []
    for index_alpha in xrange(dim_alpha):
        for index_ell in xrange(dim_ell):
            index_array.append((index_alpha, index_ell))
    #print 'Prepared memory for multiprocessing pool in {:.2f} seconds.'.format(timer()-t0_mem)

    print 'Collecting trace matrix...'

    pool = multiprocessing.Pool(processes = pool_size)
    trace_as_list = pool.map(single_step_get_trace_matrix, index_array)
    pool.close()

    #print ''

    '''
    # Do I really need this loop or can I just do: Trace = TraceAsList.reshape((dim_alpha, dim_ell))
    index = 0
    for alpha in xrange(dim_alpha):
        for l in xrange(dim_ell): # no symmetries...
            Trace[alpha, ell] = TraceAsList[index]
            index += 1
    '''
    trace = np.asarray(trace_as_list).reshape((dim_alpha, dim_ell))

    # checked this and they are indeed the same!
    #print 'Is Trace.loop = Trace.reshape?', np.allclose(Trace, Trace_alt)
    #print 'Is Trace.reshape = Trace.loop?', np.allclose(Trace_alt, Trace)

    dt = timer() - t0

    return trace, dt

# single step for parallelized loop
# unify with Fisher matrix since it's the same function used there?!...
# but now the function depends implicitly on the correct file names, so better keep it separated!
def single_step_get_trace_matrix(index_pair):

    #t0 = timer()
    if ini.use_sparse_matrices:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_index{:03d}.npy'.format(index_pair[0]))[()].toarray()
        inv_covariance_times_derivative_ell_tranposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_ell_transposed_index_ell{:03d}.npy'.format(index_pair[1]))[()].toarray()
    else:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_index{:03d}.npy'.format(index_pair[0]))
        inv_covariance_times_derivative_ell_tranposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_ell_transposed_index_ell{:03d}.npy'.format(index_pair[1]))

    total_sum = np.sum(inv_covariance_times_derivative_A * inv_covariance_times_derivative_ell_tranposed)
    # NOTE: second matrix is transposed, and this is an ARRAY multiplication (by element), NOT MATRIX.

    #print "Fisher( X, X ) in", ((timer() - ltime)), "seconds."
    #print totSum
    #print_dot()

    #dt = timer()-t0

    return total_sum

# fast, with numpy:
def get_correlation_matrix(covariance_matrix):

    t0 = timer()

    corr = covariance_matrix / np.sqrt(np.asmatrix(np.diag(covariance_matrix)).T * np.asmatrix(np.diag(covariance_matrix)))

    dt = timer() - t0

    return corr, dt

# faster? --> nope, not at least for dim = (28,28); loops but using symmetries!
def get_correlation_matrix_loop(covariance_matrix):

    t0 = timer()

    corr = np.zeros_like(covariance_matrix)

    dim = covariance_matrix.shape[0]

    # no while-loop:
    for i in xrange(dim):
        for j in xrange(i + 1): # 0 < j <= i
            corr[i, j] = covariance_matrix[i, j] / np.sqrt(covariance_matrix[i, i] * covariance_matrix[j, j])
            corr[j, i] = covariance_matrix[i, j]

    dt = timer() - t0

    return corr, dt

def assure_total_power_greater_zero(band_powers, slicing_points_bands, nzbins=2, small_positive_number=1e-9, verbose=False):
    # TODO: Generalize for EB!!!
    # NEW: This is what Hu meant with "total power in signal and noise must be > 0"
    # set this at beginning of iteration, so bp can still become negative!
    # seems to be okay for W2, but produces rubbish for W4...
    # we have to split into EE, BB, and EB and then check:
    # EE_i+BB_i(+EB_i) > 0 --> nothing has to be reset
    # EE_i+BB_i(+EB_i) < 0 --> reset negative values of corresponding band!
    band_powers_EE = band_powers[slicing_points_bands[0][0]:slicing_points_bands[0][1]]
    if 'BB' in ini.bands:
        # B-modes only:
        index_band = 1
        band_powers_BB = band_powers[slicing_points_bands[index_band][0]:slicing_points_bands[index_band][1]]
        index_jump = slicing_offset_redshifts[index_band]
        collect_new_BB = []
        for index_z in np.arange(0, nzbins*(nzbins+1)/2*slicing_offset_redshifts[index_band], slicing_offset_redshifts[index_band]):
            bp_per_zbin = band_powers_BB[index_z:index_jump]
            bp_per_zbin = np.concatenate(([0.], bp_per_zbin))
            index_jump += slicing_offset_redshifts[index_band]
            collect_new_BB.append(bp_per_zbin)
        band_powers_BB = np.asarray(collect_new_BB).flatten()
    else:
        band_powers_EE = np.zeros_like(band_powers_EE)
    if verbose:
        print 'EE band powers: \n', band_powers_EE
        print 'BB band powers: \n', band_powers_BB
    total_power = band_powers_EE+band_powers_BB
    index_EE_below_zero = np.where((band_powers_EE < 0.) & (total_power < 0.) )
    index_BB_below_zero = np.where((band_powers_BB < 0.) & (total_power < 0.))
    if verbose:
        index_total_power_below_zero = np.where(total_power < 0.)
        print 'Total power < 0: \n', total_power[index_total_power_below_zero]
        print 'EE band powers < 0: \n', band_powers_EE[index_EE_below_zero]
        print 'BB band powers < 0: \n', band_powers_BB[index_BB_below_zero]

    band_powers_EE[index_EE_below_zero] = small_positive_number
    band_powers_BB[index_BB_below_zero] = small_positive_number

    # now B-modes are the same length as E-modes:
    if 'BB' in ini.bands:
        # E-mode slicing required!!!
        index_band = 0
        index_jump = slicing_offset_redshifts[index_band]
        collect_new_BB = []
        for index_z in np.arange(0, nzbins*(nzbins+1)/2*slicing_offset_redshifts[index_band], slicing_offset_redshifts[index_band]):
            bp_per_zbin = band_powers_BB[index_z:index_jump]
            index_jump += slicing_offset_redshifts[index_band]
            collect_new_BB.append(bp_per_zbin[1:])
        band_powers_BB = np.asarray(collect_new_BB).flatten()
        band_powers = np.concatenate((np.copy(band_powers_EE), np.copy(band_powers_BB)))
    else:
        band_powers = np.copy(band_powers_EE)

    if verbose:
        print 'Concatenated band powers: \n', band_powers, band_powers.size

    return band_powers

# modified, so that input_parameters are copied and other parameters are appended
# this is actually quite unnecessary if I keep all the regular output...
def write_logfile(filename_in, filename_out, log, field_properties):
    '''
    Since I'm using a parameter file now, many entries here are obsolete...

    @arguments:

    filename    filename for the log-file

    log, field1, field2     dictionaries

    auto                    boolean
                            if True  --> field1 == field2
                            if False --> field1 != field2
    '''

    # some decorations:
    ascii_line = get_ascii_art('-', n=160)

    #keywords for field<1,2>: 'borders_y', 'borders_x', 'field_shape_in_bins', 'npixel', 'field', 'mean_n_eff', 'pixel'

    # 1) make a copy of input file:
    # 'with' takes care of also closing the file again automatically after it is done.
    with open(filename_in) as f:
        with open(filename_out, 'w') as f1:
            f1.write(ascii_line + '\n Copy of input file: \n' + ascii_line + '\n \n')
            for line in f:
                #if "ROW" in line:
                f1.write(line)
            f1.write('\n')

    # 2) append new, important parameters:
    with open(filename_out, 'a') as f:
        f.write(ascii_line + '\n New parameters: \n' + ascii_line + '\n \n')
        f.write('Full path to output files: ' + log['path_output'] + '\n' +
                'Full path to signal matrices: ' + log['path_to_signal_matrices'] + '\n' + '\n' +
                'Number of iterations: {:}/{:}.'.format(log['niterations'], log['max_iterations']) + '\n' + '\n')
        for index_zbin in xrange(len(field_properties)):
            f.write('All following values are w.r.t. whole patch and not to subfield! \n' +
                    '############################ zbin{:} ################################# \n'.format(index_zbin + 1) +
                    'Tangent point for gnomonic projection of patch: RA = {:.5f} deg, Dec = {:.5f} deg'.format(field_properties[index_zbin]['Tangent_point'][0], field_properties[index_zbin]['Tangent_point'][1]) + '\n' +
                    'Area of patch: {:.3f} deg^2'.format(field_properties[index_zbin]['field'][2]) + '\n' +
                    'Length of patch (x direction): {:.3f} deg'.format(field_properties[index_zbin]['field'][0]) + '\n' +
                    'Length of patch (y direction): {:.3f} deg'.format(field_properties[index_zbin]['field'][1]) + '\n' +
                    'Total number of pixels in patch: ' + str(field_properties[index_zbin]['npixel']) + '\n' +
                    'Area of one pixel in patch: {:.4f} deg^2 = {:.3f} arcmin^2'.format(field_properties[index_zbin]['pixel'][2], field_properties[index_zbin]['pixel'][2] * 3600.) + '\n' +
                    'Length of pixel in patch (x direction): {:.3f} deg'.format(field_properties[index_zbin]['pixel'][0]) + '\n' +
                    'Length of pixel in patch (y direction): {:.3f} deg'.format(field_properties[index_zbin]['pixel'][1]) + '\n')
            if 'n_eff' in field_properties[index_zbin]:
                f.write('Effective number density (underestimated): {:.2f} arcmin^(-2)'.format(field_properties[index_zbin]['n_eff']) + '\n')
            if 'N_eff' in field_properties[index_zbin]:
                f.write('Effective number of sources: {:.2f}'.format(field_properties[index_zbin]['N_eff']) + '\n')
            f.write('###################################################################### \n')
        f.write('Number of pixels (after masking): ' + str(log['npixel_after_masking']) + '\n' +
                'Maximal distance in field (after masking): {:.3f} deg'.format(log['r_max']) + '\n' +
                'Maximal range of angular scales allowed by field geometry: ell_field_max = {:.2f} < ell < ell_pix = {:.2f}'.format(log['ell_field_max'], log['ell_pix']) + '\n' +
                'Maximal x-range of angular scales allowed by field geometry: ell_field_max = {:.2f} < ell < ell_pix = {:.2f}'.format(log['ell_field_x'], log['ell_pix']) + '\n' +
                'Maximal y-range of angular scales allowed by field geometry: ell_field_max = {:.2f} < ell < ell_pix = {:.2f}'.format(log['ell_field_y'], log['ell_pix']) + '\n')
        #factor 2 is wrong in case of tomography...
        f.write('Mean value in noise covariance matrix: {:.3e}'.format(log['mean_noise_matrix']) + '\n' +
                'Dimension of matrices: {:.0f}x{:.0f}'.format(log['dim_covariance'][0], log['dim_covariance'][1]) + '\n' + '\n' +
                'Amount of extracted band powers: ' + str(log['initial_guess'].size) + '\n')
        if 'BB' in ini.bands:
            f.write('Lower limits of BB band ranges: \n')
            for i in log['bands_BB_min']:
                f.write('{:.0f}  '.format(i))
            f.write('\n' + 'Upper limits of BB band ranges: \n')
            for i in log['bands_BB_max']:
                f.write('{:.0f}  '.format(i))
        f.write('\n' + 'Guessed initial values for band powers: \n')
        for i in log['initial_guess']:
            f.write('{:.2e}  '.format(i))
        if 'fname_ell_values_band_window' in log:
            f.write('\n' + 'Values for ell-range used in band window function calculation are stored in: ' + str(logdata['fname_ell_values_band_window']))
        if 'time_SM' in log:
            f.write('\n' + 'Time for calculation of signal matrices: {:.2f}min.'.format(log['time_SM'] / 60.) + '\n')
        if 'time_WM' in log:
            f.write('Time for calculation of band window functions: {:.2f}min.'.format(log['time_WM'] / 60.) + '\n')
        f.write('\n' + 'Time for {:} optimization steps: {:.3f}s.'.format(log['niterations'], log['time_opt']) + '\n')
        f.write('Total runtime: {:.2f}min.'.format(logdata['runtime'] / 60.))

    return

def get_initial_guess(bands_min, bands_max, band='EE', LCDM_like=True):
    """ If LCDM_like = True the shape of the initial guess is roughly a LCDM like Cl (with ~10x higher normalization). Otherwise the initial guess starts from 1 (for EE) and 0 (BB, EB).
    """

    ells = (bands_min+bands_max)/2.
    #print 'l:', l

    # hardcoded interpolation range:
    if LCDM_like:
        if band == 'EE':
            bands_initial_min = np.array([20., 40., 280., 500., 750., 1300., 2300.])
            bands_initial_max = np.array([40., 280., 500., 750., 1300., 2300., 5350.])

            initial_guess = np.array([1.5e-5, 3e-5, 5.5e-5, 8e-5, 1.2e-4, 1.8e-4, 2.5e-4])
        else:
            bands_initial_min = np.array([20., 120., 500., 750., 1300., 2300.])
            bands_initial_max = np.array([120., 500., 750., 1300., 2300., 5350.])

            initial_guess = np.array([3.24e-07, 1.296e-06, 3.1304025e-06, 9.75156250e-06, 3.717184e-05, 1.46306250e-04])

        ells_intp = (bands_initial_min+bands_initial_max)/2.
        guessed_PS = interpolate.interp1d(ells_intp, initial_guess)

        #print 'l_intp:', l_intp

        guess = guessed_PS(ells)

    else:
        if band == 'EE':
            guess = np.ones_like(ells) * 1e-6
        else:
            #guess = np.ones_like(ells) * 1e-12
            # zerosare fine, since we always add noise to the main diagonal (and block off-diagonals can be 0, esp. for B-modes!)
            guess = np.zeros_like(ells)

    return guess

def get_input_parameters(filename):
    """ Helper function that reads in required input parameters from file <filename>.
        We always assume that this file is using Python notation for variables and follows the notation and variable names in "default.ini"!
    """

    import imp
    f = open(filename)
    global ini
    ini = imp.load_source('data', '', f)
    f.close()

    # here is the place to include safety checks!
    if ini.max_iterations < 2:
        ini.max_iterations = 2

    return

# dumb implementation but at least it's general...
def get_matrix_string(nzbins, indices):

    matrix = np.zeros((nzbins, nzbins))
    for index_zbin1 in xrange(nzbins):
        for index_zbin2 in xrange(index_zbin1 + 1):
            if index_zbin1 == indices[0] and index_zbin2 == indices[1]:
                matrix[index_zbin1, index_zbin2] = 1
            matrix[index_zbin2, index_zbin1] = matrix[index_zbin1, index_zbin2]

    #print matrix

    row = ''
    for index_row in xrange(nzbins):
        #print matrix[:,index_row]
        for index_elem, elem in enumerate(matrix[:, index_row]):
            if elem == 1:
                if index_elem == nzbins - 1:
                    row += 'signal_block;'
                else:
                    row += 'signal_block,'
            else:
                if index_elem == nzbins - 1:
                    row += 'zero_block;'
                else:
                    row += 'zero_block,'
    # cut off the last "delimiter" ";"
    row = row[:-1]

    return row

# pretty dumb implementation...
#'''
def get_matrix_string_ell(nzbins):

    col = ''

    for index_col in xrange(nzbins):
        if index_col != nzbins - 1:
            col += 'signal_block,'
        else:
            col += 'signal_block;'

    row = ''
    for index_row in xrange(nzbins):
        row += col
    row = row[:-1]

    return row
#'''

# dumb implementation but at least it's general...
def get_matrix_string_noise(nzbins, index):

    matrix = np.zeros((nzbins, nzbins))
    for index_zbin1 in xrange(nzbins):
        for index_zbin2 in xrange(index_zbin1 + 1):
            if index_zbin1 == index and index_zbin2 == index:
                matrix[index_zbin1, index_zbin2] = 1

    #print matrix

    row = ''
    for index_row in xrange(nzbins):
        #print matrix[:,index_row]
        for index_elem, elem in enumerate(matrix[:, index_row]):
            if elem == 1:
                if index_elem == nzbins - 1:
                    row += 'noise_block;'
                else:
                    row += 'noise_block,'
            else:
                if index_elem == nzbins - 1:
                    row += 'zero_block;'
                else:
                    row += 'zero_block,'
    # cut off the last "delimiter" ";"
    row = row[:-1]

    return row

# copy & paste from stackoverflow
def gen_log_space(start, end, n):
    """
    creates n log-spaced integers between (& including) start, end
    without repetitions. In the beginning of the sequence this might
    yield rather a lin-spaced sequence.

    """
    result = [start]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(end) / result[-1])**(1. / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1] + 1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(end) / result[-1])**(1. / (n - len(result)))

    return np.asarray(result).astype('int')

def do_numerical_tests(matrix, inv_matrix, matrix_is_fisher=True):
    """ Function to perform some basic numerical tests on symmetry properties and precision of inversion.
    """

    t0 = timer()

    if matrix_is_fisher:
        type_of_matrix = 'F'
    else:
        type_of_matrix = 'C'

    trans_matrix = matrix.T
    trans_inv_matrix = inv_matrix.T
    identity_matrix = np.eye(matrix.shape[0])

    det_matrix = np.linalg.det(matrix)
    print 'Numerical value of det(' + type_of_matrix + ') = {:.4e}'.format(det_matrix)
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    logsum = 0
    if matrix_is_fisher:
        print 'Are there negative eigenvalues in F?'
    for i in xrange(len(eig_vals)):
        if matrix_is_fisher:
            print 'Eigenvalue {:} = {:.2e}'.format(i + 1, eig_vals[i])
        logsum += np.log10(eig_vals[i])
    print 'log10(det(' + type_of_matrix + ')): ', logsum
    left_product = matrix.dot(inv_matrix)
    right_product = inv_matrix.dot(matrix)
    print 'Is ' + type_of_matrix + ' = ' + type_of_matrix + '^T? ', np.allclose(matrix, trans_matrix)
    print 'Is ' + type_of_matrix + '^T = ' + type_of_matrix + '? ', np.allclose(trans_matrix, matrix)
    print 'Is ' + type_of_matrix + '^{-1} = (' + type_of_matrix + '^{-1})^T? ', np.allclose(inv_matrix, trans_inv_matrix)
    print 'Is (' + type_of_matrix + '^{-1})^T = ' + type_of_matrix + '^{-1}? ', np.allclose(trans_inv_matrix, inv_matrix)
    print 'Is ' + type_of_matrix + '*' + type_of_matrix + '^{-1} = 1? ', np.allclose(left_product, identity_matrix)
    print 'Is ' + type_of_matrix + '^{-1}*' + type_of_matrix + ' = 1? ', np.allclose(right_product, identity_matrix)
    print 'Is 1 = ' + type_of_matrix + '*' + type_of_matrix + '^{-1}? ', np.allclose(identity_matrix, left_product)
    print 'Is 1 = ' + type_of_matrix + '^{-1}*' + type_of_matrix + '? ', np.allclose(identity_matrix, right_product)

    if matrix_is_fisher:
        print 'Are there negative values on diagonal of F^{-1}? \n', np.diag(inv_matrix)
        print 'Are there negative values on diagonal of F? \n', np.diag(matrix)
        print 'WV CHECK: there are negatives if we include off-diagonal block-diagonal noise contributions.'

    dt = timer() - t0
    print 'Time for numerical tests: {:.2f}s.'.format(dt)

    return

def get_ascii_art(symbol, n=80):

    ascii_art = ''

    if n < 80:
        n = 80

    for i in xrange(n):
        ascii_art += symbol

    return ascii_art

def print_ascii_art(text):

    hashtags = get_ascii_art('#', n=len(text))

    print hashtags
    print hashtags + '\n'
    print text + '\n'
    print hashtags
    print hashtags

    return

if __name__ == '__main__':

    # TODO: Make code more modular:
    # 1) init-function (setting up folders, initial guess etc.)
    # 2) function for Newton-Raphson (new external module?)
    # 3) function for BWM (new external module?)
    # However, that's currently only cosmetics for future public release...
    t0_prog = timer()
    # read filename for input parameter file (include some safety checks later on...):
    try:
        fname_input = str(sys.argv[1])
    except:
        print 'No filename for input parameters supplied. \n Loading parameters from \"default.ini\" now.'
        fname_input = 'default.ini'

    # read in input parameters from specified file:
    get_input_parameters(fname_input)
    # create dictionary ofcolumn names for catalog files
    column_names = {'RA': ini.column_ra,
                    'Dec': ini.column_dec,
                    'e1': ini.column_e1,
                    'e2': ini.column_e2,
                    'weight': ini.column_weight,
                    'm1': ini.column_m1,
                    'm2': ini.column_m2,
                    'c1': ini.column_c1,
                    'c2': ini.column_c2
                    }

    # not really necessary any longer...
    if ini.mode in ['clones', 'Clones']:
        mode = 'clones'
    elif ini.mode in ['grfs', 'GRFs', 'GRFS']:
        mode = 'grfs'
    elif ini.mode in ['data', 'Data']:
        mode = 'data'
    elif ini.mode in ['mocks', 'Mocks']:
        mode = 'mocks'

    # dirty way to make script BASH-loopable in case of 184 CFHTLenS Clone mocks or GRFs:
    if ini.make_BASH_loopable:
        ini.identifier_in = str(sys.argv[2])

    # initialize dictionary for logfile-data:
    logdata = {}
    fname_logfile = ini.identifier_in + '.log'
    logdata['max_iterations'] = ini.max_iterations
    path_to_dir_of_run, fname_of_script = os.path.split(os.path.abspath(__file__))
    print 'Script', fname_of_script, ' is being run in', path_to_dir_of_run

    z_min = min(ini.z_min)
    z_max = max(ini.z_max)

    redshift_bins = []
    for index_zbin in xrange(len(ini.z_min)):
        redshift_bin = '{:.2f}z{:.2f}'.format(ini.z_min[index_zbin], ini.z_max[index_zbin])
        redshift_bins.append(redshift_bin)
    # number of z-bins
    nzbins = len(redshift_bins)
    # number of *unique* correlations between z-bins
    nzcorrs = nzbins * (nzbins + 1) / 2

    # create a map of indices, mapping from unique correlation index to redshift bin indices
    index_map = []
    for index_zbin1 in xrange(nzbins):
        for index_zbin2 in xrange(index_zbin1 + 1):
            index_map.append((index_zbin1, index_zbin2))

    # two possible options:
    # a) a list sigma_int_e1 and a list sigma_int_e2 are passed --> symmetrize
    # b) a list (of already symmetrized) sigma_int is passed --> just change to array
    # list is assumed (but not checked for) to have length of z-bins!
    # needs to be a nested try-except-structure!


    if ini.estimate_sigma_int:
        sigma_int = np.ones(nzbins) * -1.
        print 'sigma_int will be estimated from data.'
    else:
        try:
            sigma_int = np.sqrt((np.asarray(ini.sigma_int_e1)**2 + np.asarray(ini.sigma_int_e2)**2) / 2.)
            print 'Symmetrizing sigma_int_e1 and sigma_int_e2 to sigma_int.'
        except:

            try:
                sigma_int = np.asarray(ini.sigma_int)
                print 'Using supplied sigma_int.'
            except:
                print 'Either sigma_int or sigma_int_e1 and sigma_int_e2 must be supplied in input-file!'
                exit()

    # this is just for naming purposes:
    if ini.estimate_sigma_int:
        sigma_int_for_naming = 'sigma_int_est'
    else:
        sigma_int_for_naming = 'sigma_int{:.2f}'.format(sigma_int[0])

    # remove redshift_bin folder?! --> this would create super-long folder-names for more than 3 zbins...
    #if nzbins > 1:
    path_output = ini.root_run + '/{:}/{:.2f}z{:.2f}/{:}zbins/'.format(sigma_int_for_naming, z_min, z_max, nzbins) + '/' + ini.identifier_in + '/'
    path_store = ini.root_store + '/{:}/{:.2f}z{:.2f}/{:}zbins/'.format(sigma_int_for_naming, z_min, z_max, nzbins) + '/' + ini.identifier_in + '/'
    # set subfolder for storing of matrices (does not depend on sigma_int, but depends on nzbins due to different masking of field!):
    path_to_signal = ini.root_signal + '/{:.2f}z{:.2f}/{:}zbins/'.format(z_min, z_max, nzbins)
    path_to_signal_store = ini.root_signal_store + '/{:.2f}z{:.2f}/{:}zbins/'.format(z_min, z_max, nzbins)
    '''
    else:
        path_output = ini.root_run + '/{:}/{:.2f}z{:.2f}/'.format(sigma_int_for_naming, z_min, z_max) + '/' + ini.identifier_in + '/'
        path_store = ini.root_store + '/{:}/{:.2f}z{:.2f}/'.format(sigma_int_for_naming, z_min, z_max) + '/' + ini.identifier_in + '/'
        # set subfolder for storing of matrices:
        path_to_signal = ini.root_signal + '/{:.2f}z{:.2f}/'.format(z_min, z_max)
    '''
    if not os.path.isdir(path_to_signal):
        os.makedirs(path_to_signal)

    if path_to_signal != path_to_signal_store:
        if not os.path.isdir(path_to_signal_store):
            os.makedirs(path_to_signal_store)


    logdata['path_output'] = path_store
    logdata['path_to_signal_matrices'] = path_to_signal_store

    # check if output exists already; if not, create the directory
    if not os.path.isdir(path_output):
        os.makedirs(path_output)
        os.makedirs(path_output + 'cache/')
        os.makedirs(path_output + 'plots/')
        os.makedirs(path_output + 'control_outputs/')

    # TODO: Make this work again...
    # for consistency with BWM calculation:
    if not os.path.isdir(path_output + 'cache/'):
        os.makedirs(path_output + 'cache/')

    if not os.path.isdir(path_output + 'bwm/'):
        os.makedirs(path_output + 'bwm/')

    # switch to new path
    os.chdir(path_output)

    # set paths; root path must point down to split in redshift:
    # TODO: generalize to 'nzbins':
    paths_to_data = []
    for index_zbin, zbin in enumerate(redshift_bins):
        paths_to_data.append(ini.root_path_data + '/' + zbin + '/')
        try:
            # for legacy reasons...
            filename = ini.identifier_in + '_clean.cat'
        except:
            filename = ini.identifier_in + '.fits'

        print 'Loaded catalogs from:'
        print paths_to_data[index_zbin] + filename

    print 'Field ID, sigma_int (per redshift bin):', ini.identifier_in, sigma_int
    print 'Path for saving:', path_output
    print 'Mode is:', mode

    t0_data = timer()
    # looping over 'nzbins' is done internally in function:
    # here we have to pass the sigma_int !!!
    data, shot_noise, field_properties, x_rad_all, y_rad_all = dr.get_data(paths_to_data, filename, names_zbins=redshift_bins, identifier=ini.identifier_in, sigma_int=sigma_int, pixel_scale=ini.side_length,
                                                                      nzbins=nzbins, mode=ini.mode, min_num_elements_pix=ini.minimal_number_elements_per_pixel, column_names=column_names)
    print 'Time for data reduction: {:.4f}s'.format(timer() - t0_data)
    print 'Sigma_int (per z-bin): \n', sigma_int

    # sanity check for that coordinates are the same...:
    # What if not?!
    # This problem is solved now!
    '''
    print 'Is x1=x2?', np.allclose(x_rad_all[0], x_rad_all[1])
    print 'Is y1=y2?', np.allclose(y_rad_all[0], y_rad_all[1])

    print x_rad_all[0]-x_rad_all[1]
    print y_rad_all[0]-y_rad_all[1]
    '''

    # all x_rad, y_rad should be the same per zbin (that's why minimal field and maximal mask), hence:
    x_rad = x_rad_all[0]
    y_rad = y_rad_all[0]

    logdata['npixel_after_masking'] = x_rad.size

    # multiply noise values from both z-bins; this is the diagonal of the diagonal noise matrix!
    # so, noise1 and noise2 must be flattened 1-d arrays!

    # ATTENTION: This is no general implementation but hardcoded for two redshift bins!!!
    # ATTENTION: Noise mustn't be added in case of off-diagonal bins...

    # TODO: Below I assume that the arrays 'data' and 'shot_noise_<e1, e2>' (outer dim=nzbins) exist!
    # construct data_vector and diagonal of noise matrix (assumed to be diagonal anyway) here:
    # noise matrix is assumed to be diagonal!
    inv_N_eff_gal = np.zeros_like(shot_noise)
    for index_zbin in xrange(nzbins):
        # concatenation required because data has two ellipticity components: e1 & e2!
        diag_zbin = np.concatenate((shot_noise[index_zbin], shot_noise[index_zbin]))
        inv_N_eff_gal[index_zbin] = shot_noise[index_zbin] / sigma_int[index_zbin]**2
        if index_zbin == 0:
            diag_noise = diag_zbin
            data_vector = data[index_zbin]
        else:
            diag_noise = np.concatenate((diag_noise, diag_zbin))
            data_vector = np.concatenate((data_vector, data[index_zbin]))

    print 'Mean of noise matrix = {:.4e}'.format(diag_noise.mean())

    # \ell-bands should be set according to these values!
    # I think ell_field is too naive, since in a square field this would basically be the diagonal connections between corners which would only yield 4 modes...
    # Hu & White 2001 seem to assume the side length of the (square) field to be the maximal distance and
    # Lin et al. 2012 even seem to assume half the side length of the (square) field...
    dist_x_max = (x_rad.max() + np.abs(x_rad.min()))
    dist_y_max = (y_rad.max() + np.abs(y_rad.min()))
    dist_max = np.sqrt(dist_x_max**2 + dist_y_max**2)
    ell_field_x = 2. * np.pi / dist_x_max
    ell_field_y = 2. * np.pi / dist_y_max
    ell_field_max = 2. * np.pi / dist_max
    ell_pix = 2. * np.pi / np.deg2rad(ini.side_length)

    logdata['ell_field_x'] = ell_field_x
    logdata['ell_field_y'] = ell_field_y
    logdata['ell_field_max'] = ell_field_max
    logdata['ell_pix'] = ell_pix
    #logdata['side_length'] = ini.side_length
    logdata['r_max'] = np.rad2deg(dist_max)

    print 'Side length of 1 pixel = {:.3f} deg'.format(ini.side_length)
    print 'ell_field_max = {:.2f} < ell < ell_pix = {:.2f}'.format(ell_field_max, ell_pix)
    print 'ell_field_x = {:.2f} < ell < ell_pix = {:.2f}'.format(ell_field_x, ell_pix)
    print 'ell_field_y = {:.2f} < ell < ell_pix = {:.2f}'.format(ell_field_y, ell_pix)

    #print 'Diagonal of noise matrix:', diag_noise
    dim_noise = diag_noise.size
    print 'Dimension of full covariance: {:}'.format(dim_noise)
    # divide by two because of g1, g2!
    print 'Number of unmasked pixels in field: {:}'.format(dim_noise / nzbins / 2)
    # we have to consider that matrices are two times the dimension of the field, but we also divide by the doubled amount of elements!
    mean_diag_noise = diag_noise.mean()
    print 'Mean of noise matrix = {:.4e}'.format(mean_diag_noise)
    logdata['mean_noise_matrix'] = mean_diag_noise
    for index_zbin in xrange(nzbins):
        print 'Effective number density n_eff in patch in zbin{:} (underestimated) = {:.2f}'.format(index_zbin + 1, field_properties[index_zbin]['n_eff'])
        print 'Effective number of sources N_eff in patch in zbin{:} = {:.2f}'.format(index_zbin + 1, field_properties[index_zbin]['N_eff'])
    # ATTENTION: Stop here for data reduction only!
    # TODO: make data reduction optional and code compatible with loading binned catalogs...
    #exit()

    # here, the calculations start:
    # for testing I switch to EE only...
    #logdata['extracted_band'] = ini.bands[0]
    bands_EE_min = np.array(ini.bands_EE_min)
    bands_EE_max = np.array(ini.bands_EE_max)
    # easier:
    '''
    # log-spaced:
    bands_EE = np.logspace(np.log10(30.), np.log10(5100.), 8)
    # linearly spaced
    #dl = 2*l_field
    #bands_EE = np.arange(l_field, 5100., dl)
    #bands_EE_min = np.concatenate((bands_EE[:-2], np.array([2300.])))
    #bands_EE_max = np.concatenate((bands_EE[1:-2], np.array([2300., 5100.])))
    #print bands_EE_min
    #print bands_EE_max
    '''
    bands_BB_min = bands_EE_min[1:]
    bands_BB_max = bands_EE_max[1:]
    bands_EB_min = bands_EE_min[1:]
    bands_EB_max = bands_EE_max[1:]
    bands_min = [bands_EE_min, bands_BB_min, bands_EB_min]
    bands_max = [bands_EE_max, bands_BB_max, bands_EB_max]
    # store BB bands; EE are already specified in input parameters
    if 'BB' in ini.bands:
        logdata['bands_BB_min'] = bands_BB_min
        logdata['bands_BB_max'] = bands_BB_max

    # guessed from Hu & White 2001, Fig. 3 (upper left)
    initial_band_powers_EE = get_initial_guess(bands_EE_min, bands_EE_max, band='EE', LCDM_like=False)
    initial_band_powers_BB = get_initial_guess(bands_BB_min, bands_BB_max, band='BB', LCDM_like=False)
    # initial guess of BB is 0, so we can also use it for EB!
    initial_band_powers_EB = get_initial_guess(bands_EB_min, bands_EB_max, band='BB', LCDM_like=False)

    initial_band_powers_list = [initial_band_powers_EE, initial_band_powers_BB, initial_band_powers_EB]

    # initial_EE_z1z1, initial_EE_z1z2, initial_EE_z2z1, initial_EE_z2z2; initial_BB_z1z1, initial_BB_z1z2, initial_BB_z2z1, initial_BB_z2z2
    # More correct: initial_EE_z1z1, initial_EE_z1z2, initial_EE_z2z2; initial_BB_z1z1, initial_BB_z1z2, initial_BB_z2z2; since initial_BB_z1z2 = initial_BB_z2z1
    for index_band in xrange(len(ini.bands)):
        for index_zcorr in xrange(nzcorrs):
            if index_band == 0 and index_zcorr == 0:
                initial_band_powers = initial_band_powers_list[index_band]
            else:
                initial_band_powers = np.concatenate((initial_band_powers, initial_band_powers_list[index_band]))

    # save initial guess in file for convenience. Do it only once though.
    #                       slicing points for all EE bands
    slicing_points_bands = {'EE': (0, nzcorrs * initial_band_powers_EE.size),
                            # slicing points for all BB bands
                            'BB': (nzcorrs * initial_band_powers_EE.size, nzcorrs * (initial_band_powers_EE.size + initial_band_powers_BB.size)),
                            # slicing points for all EB bands
                            'EB': (nzcorrs * (initial_band_powers_EE.size + initial_band_powers_BB.size), initial_band_powers.size)
                           }
    slicing_offset_redshifts_EE = initial_band_powers_EE.size
    slicing_offset_redshifts_BB = initial_band_powers_BB.size
    slicing_offset_redshifts_EB = initial_band_powers_EB.size
    slicing_offset_redshifts = [slicing_offset_redshifts_EE, slicing_offset_redshifts_BB, slicing_offset_redshifts_EB]
    print slicing_points_bands
    print initial_band_powers
    #print initial_band_powers_EE
    #print initial_band_powers_BB
    for i, band in enumerate(ini.bands):
        print initial_band_powers[slicing_points_bands[band][0]:slicing_points_bands[band][1]], initial_band_powers[slicing_points_bands[band][0]:slicing_points_bands[band][1]].shape
        initial_band_powers_band = initial_band_powers[slicing_points_bands[band][0]:slicing_points_bands[band][1]]
        fname = 'initial_guess_{:}.dat'.format(band)
        if not os.path.isfile(fname):
            savedata = np.column_stack(((bands_min[i] + bands_max[i]) / 2., bands_min[i], bands_max[i], initial_band_powers_band[:slicing_offset_redshifts[i]]))
            header = 'naive bin center, bin_min, bin_max, initial_guess_for_band_powers'
            np.savetxt(fname, savedata, header=header)
        # write also out multipoles in extra file for convenience:
        fname = 'multipoles_{:}.dat'.format(band)
        if not os.path.isfile(fname):
            savedata = np.column_stack(((bands_min[i] + bands_max[i]) / 2., bands_min[i], bands_max[i]))
            header = 'naive bin center, bin_min, bin_max'
            np.savetxt(fname, savedata, header=header)
        # create here some indices for the convergence criterion:
        if band == 'EE':
            bands_to_use_EE = np.ones_like(bands_EE_min)
            # don't use first EE band:
            bands_to_use_EE[0] = 0
            # don't use last EE band:
            bands_to_use_EE[-1] = 0
            # don't use second to last EE band:
            #bands_in_conv_BB[-2] = 0
            all_bands_to_use_EE = []
            for i in xrange(nzbins * (nzbins + 1) / 2):
                all_bands_to_use_EE += bands_to_use_EE.tolist()
        if band == 'BB':
            bands_to_use_BB = np.ones_like(bands_BB_min)
            # don't use last BB band:
            bands_to_use_BB[-1] = 0
            # don't use second to last BB band:
            #bands_in_conv_BB[-2] = 0
            all_bands_to_use_BB = []
            for i in xrange(nzbins * (nzbins + 1) / 2):
                all_bands_to_use_BB += bands_to_use_BB.tolist()
        if band == 'EB':
            bands_to_use_EB = np.ones_like(bands_EB_min)
            # don't use last BB band:
            bands_to_use_EB[-1] = 0
            # don't use second to last BB band:
            #bands_in_conv_EB[-2] = 0
            all_bands_to_use_EB = []
            for i in xrange(nzbins * (nzbins + 1) / 2):
                all_bands_to_use_EB += bands_to_use_EB.tolist()

    # TODO: Expand this to possibly include also EB bands in convergence (but this is a very unlikely use-case as is already the inclusion of BB bands)
    if ini.include_BB_in_convergence:
        all_bands_to_use = all_bands_to_use_EE + all_bands_to_use_BB
    elif not ini.include_BB_in_convergence and 'BB' in ini.bands:
        all_bands_to_use = all_bands_to_use_EE + np.zeros_like(all_bands_to_use_BB).tolist()
    else:
        all_bands_to_use = all_bands_to_use_EE

    # this boolean contains now all indices of bands that should be included in convergence criterion for Newton Raphson!
    indices_bands_to_use = np.where(np.asarray(all_bands_to_use) == 1)[0]

    #print 'All bands to use: \n', all_bands_to_use
    #print 'indices_bands_to_use: \n', indices_bands_to_use

    #exit()

    # call garbage collector explicitly:
    gc.collect()
    # if working with mock data the initial guess for the signal matrix can be shared (if all fields look the same) to save runtime!
    if ini.share_signal_matrices:
        identifier_SM = ini.shared_identifier_SM
    else:
        identifier_SM = ini.identifier_in

    # try to load signal matrices, else calculate them:
    for index_band, band in enumerate(ini.bands):

        print 'band_min:', bands_min[index_band].min()
        print 'band_max:', bands_max[index_band].max()
        #fname = path_to_signal[i]+'signal_matrices_'+band+'_'+str(bands_min[i].min())+'_'+str(bands_max[i].max())+'_'+str(np.round(ini.side_length, decimals=2))
        #print fname
        #if not os.path.isdir(path_to_signal):
        #    os.makedirs(path_to_signal)
            #signal = []
        t0_signal_matrices = timer()
        fname = path_to_signal + 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '.h5'

        #print fname
        #exit()
        if not os.path.isfile(fname): # + '.npy'):
            cmd = 'w'
        else:
            cmd = 'a'

        # calculate (derivative?) signal matrices:
        for index_band_power in xrange(bands_max[index_band].size): #bands_min.size):

            fname_matrix = 'signal_matrix_index_bp{:03d}'.format(index_band_power)

            # check if matrices already exist:
            try:
                with h5py.File(fname, 'r') as hf:
                    matrix_names = hf.keys()

                if fname_matrix in matrix_names:
                    matrix_already_exists = True
                else:
                    matrix_already_exists = False

            except:
                matrix_already_exists = False

            if (cmd == 'w') or ((cmd == 'a') and not matrix_already_exists):
                # these are GLOBAL, together with "band"
                band_min = bands_min[index_band][index_band_power]
                band_max = bands_max[index_band][index_band_power]
                print 'Calculating ' + band + ' signal matrix {:}'.format(index_band_power + 1) + '/{:}'.format(bands_max[index_band].size) + ' with {:.3}'.format(band_min) + ' <= ell <= {:.3}.'.format(band_max)
                print 'Calculating moments of window function.'
                # this is done now in the initialization of SignalMatrix()
                t0_wf = timer()
                dSM = sm.SignalMatrix(band_min=band_min, band_max=band_max, band=band, sigma=ini.side_length, integrate=True, ncpus=ini.ncpus_SM)
                print 'Done. Time: {:.2f}s.'.format(timer() - t0_wf)
                # decide if Fast implementation is used or slow...
                # intensive testing on CFHTLenS data showed now difference in the *end* results between using FAST SM or SLOW SM (the matrices are different though!)
                if ini.calculate_SM_fast:
                    print 'Fast calculation.'
                    signal = np.asarray(dSM.getSignalMatrixFAST(x_rad, y_rad))
                else:
                    print 'Slow calculation.'
                    signal = np.asarray(dSM.getSignalMatrix(x_rad, y_rad))
                print 'Calculation of signal matrix took {:.2f}s.'.format(timer() - t0_wf)
                #fname = path_to_signal + 'signal_matrix_' + band + '_{:.3f}_index_bp{:03d}_'.format(ini.side_length, index_band_power) + identifier_SM
                #np.save(fname, signal)
                #fname = path_to_signal + 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '.h5'

                with h5py.File(fname, cmd) as hf:
                    #group = hf.create_group(band)
                    hf.create_dataset(fname_matrix, data=signal)
                # if it was 'w', set to 'a'!
                cmd = 'a'

        dt_signal_matrices = timer() - t0_signal_matrices
        print 'Calculation of signal matrices took {:.2f}min.'.format(dt_signal_matrices / 60.)

    if ini.calculate_only_signal_matrices_for_derivatives:
        print 'Done with calculation of signal matrices for derivatives.'
        if ini.root_signal != ini.root_signal_store:
            t0_store = timer()
            print 'Copying signal matrix files now to: \n', path_to_signal_store
            import shutil
            for index_band, band in enumerate(ini.bands):
                fname = 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '.h5'
                shutil.move(path_to_signal + fname, path_to_signal_store)
            #os.chdir(path_to_signal_store)
            #shutil.rm(path_to_signal + fname)
            print 'Done in {:.2f}s.'.format(timer() - t0_store)
        exit()

    logdata['time_SM'] = dt_signal_matrices

    # now we have to use this list of list of matrices to construct the needed list of matrices:
    index_signals = 0
    for index_band, band in enumerate(ini.bands):
        # signal_block is now the list [SM_Band_<band><1>...SM_Band_<band><n>]
        #signal_blocks = signals[index_band]
        #fname = path_to_signal[index_band]+'signal_matrices_'+band+'_{:}_{:}_{:.2f}_'.format(int(bands_min[index_band].min()), int(bands_max[index_band].max()), ini.side_length)+ini.identifier_in+'.npy'
        #signal_blocks = np.load(fname)
        #print 'Matrices loaded.'
        #zero_block = np.zeros_like(signal_blocks[0])
        fname_SM = path_to_signal + 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '.h5'
        for index_zcorr in xrange(nzcorrs): #len(redshift_bins)**2):
            for index_band_power in xrange(bands_max[index_band].size):

                # check if derivative matrices exist already (e.g. for BWM run):
                fname_deriv = 'cache/deriv_matrix_index{:03d}.npy'.format(index_signals)

                if os.path.isfile(fname_deriv):
                    # this is only needed once for "dims_single_deriv"
                    if index_signals == 0:
                        matrix = np.load(fname_deriv)
                    index_signals += 1
                # if not, calculate them:
                else:

                    with h5py.File(fname_SM, 'r') as hf:
                        #group = hf.get(band)
                        signal_block = np.array(hf.get('signal_matrix_index_bp{:03d}'.format(index_band_power)))
                    zero_block = np.zeros_like(signal_block)
                    # this is the correct way to build up the list of derivative matrices!
                    # ATTENTION: the ordering here must be consistent with ordering in Fisher matrix!
                    if nzcorrs == 1:
                        # for general compatibility with bracket notation in 'load', save matrix as sparse matrix, although it's not sparse...
                        if ini.use_sparse_matrices:
                            matrix = sparse.bsr_matrix(signal_block)
                        else:
                            matrix = signal_block
                    else:
                        # we make use of sparse matrices --> requires peculiar bracket notation '[()]' in 'load' in order to restore sparse!
                        matrix_string = get_matrix_string(nzbins, index_map[index_zcorr])
                        if ini.use_sparse_matrices:
                            matrix = sparse.bsr_matrix(np.bmat(matrix_string))
                        else:
                            matrix = np.bmat(matrix_string)
                    # save to disk:
                    np.save('cache/' + 'deriv_matrix_index{:03d}'.format(index_signals), matrix)
                    index_signals += 1

    # shape of "matrix" doesn't depend on band type or redshift
    dims_single_deriv = matrix.shape

    dim_signals = index_signals

    # if we want to marginalize over noise parameters, define here the derivatives of the noise matrix
    # currently we only assume a diagonal noise matrix of the form:
    # C_noise_(ij, ab, \mu\nu) = p_noise(z_\mu) / N_i(\mu) d_ij d_ab d_\mu\nu
    if ini.marginalize_over_noise:
        for index_zbin in xrange(nzbins):
            #diag_deriv_noise = np.concatenate((shot_noise[index_zbin], shot_noise[index_zbin]))
            diag_deriv_noise = np.concatenate((inv_N_eff_gal[index_zbin], inv_N_eff_gal[index_zbin]))
            noise_block = np.eye(diag_deriv_noise.size, diag_deriv_noise.size)
            zero_block = np.zeros_like(noise_block)
            index_diag = range(diag_deriv_noise.size)
            noise_block[index_diag, index_diag] = diag_deriv_noise
            matrix_string = get_matrix_string_noise(nzbins, index_zbin)
            deriv_noise_matrix = np.bmat(matrix_string)
            np.save('cache/' + 'deriv_matrix_index{:03d}'.format(dim_signals + index_zbin), deriv_noise_matrix)

        # count up number of signals; for current noise implementation that's just #z-bins:
        dim_signals += nzbins

    print 'Dim(signals):', dim_signals

    # call garbage collector explicitly:
    gc.collect()

    # start of Newton-Raphson:
    # make it a function or modularize more?!
    # counter for number of iterations:
    niterations = 0
    if ini.optimize:
        t0_Newton_Raphson = timer()
        collect_band_powers = []
        collect_delta_band_powers = []
        collect_relative_difference_band_powers = []
        collect_fisher = []
        collect_inv_fisher = []
        collect_covariances = []
        collect_inv_covariances = []
        band_powers = initial_band_powers
        # for noise_marginalization we start with p_noise(z_i) = sigma_int**2(z_i)
        p_noise0 = sigma_int**2 #np.ones(nzbins)
        p_noise = p_noise0
        collect_p_noise = []
        collect_delta_p_noise = []
        collect_relative_difference_p_noise = []
        collect_p_noise.append(p_noise0)
        collect_band_powers.append(initial_band_powers)
        # switch to local variable (because we need to modify it!)
        lambda0 = ini.lambda0
        converged = False
        while not converged:
            t0_loop = timer()
            print 'Lambda = {:}'.format(lambda0)
            print 'Iteration', niterations + 1
            print 'Estimates for band powers: \n', band_powers, band_powers.size
            # now we check that EE+BB+EB > 0., if not we set the corresponding EE < 0, BB < 0, or EB < 0 to a small positive number!
            # this more complicated approach created numerical instabilities (i.e. 'oscillations')
            #band_powers = assure_total_power_greater_zero(band_powers, slicing_points_bands, nzbins=nzbins, small_positive_number=ini.resetting_value)
            if ini.reset_negative_band_powers:
                # now we check that EE+BB+EB > 0., if not we set every EE < 0, BB < 0, or EB < 0 to a small positive number!
                index_of_bp_smaller_zero = np.where(band_powers < 0.)
                print 'Band powers < 0: \n', band_powers[index_of_bp_smaller_zero]
                # 1e-6 was probably too large... (same order of magnitude as the signal...)
                #band_powers_old = np.copy(band_powers)
                # this number could probably still be lower (or even dependent on the band type)
                # or maybe it should be removed entirely (might create negative numbers on diagonal of covariance though...)
                band_powers[index_of_bp_smaller_zero] = float(ini.resetting_value) #1e-9
                #print 'Old band powers: \n', band_powers_old
                print 'Reset band powers: \n', band_powers

            #full_signal = np.zeros_like(signals[0])
            full_signal = np.zeros(dims_single_deriv)
            # first "half" of band_powers is EE, second "half" is BB
            # make this also a multiprocessing "loop"?!:
            t0_mult = timer()
            for index_band_power in xrange(band_powers.size):
                # multiply each band power with its corresponding "derivative matrix"
                # always stored as sparse matrix which requires peculiar bracket notation '[()]' in 'load'
                signal_deriv = np.load('cache/' + 'deriv_matrix_index{:03d}.npy'.format(index_band_power)) #[()].toarray()
                # this is scalar*matrix
                product = band_powers[index_band_power] * signal_deriv
                full_signal += product
            # TODO: the noise matrix is now only a diagonal...
            #full_covariance = np.asarray(full_signal)+np.asarray(noise_matrix)
            full_covariance = np.asarray(full_signal)
            index_diag = range(dims_single_deriv[0])
            # since noise matrix is assumed to be diagonal, only add the diagonals:
            if ini.marginalize_over_noise:
                print 'p_noise(z_mu): \n', p_noise
                # noise matrix is assumed to be diagonal!
                for index_zbin in xrange(nzbins):
                    #if np.abs(p_noise[index_zbin] - p_noise0[index_zbin]) / p_noise0[index_zbin] > 0.5:
                    #    p_noise[index_zbin] = p_noise0[index_zbin]
                    # concatenation required because data has two ellipticity components: e1 & e2!
                    #diag_zbin = np.concatenate((p_noise[index_zbin] * shot_noise[index_zbin], p_noise[index_zbin] * shot_noise[index_zbin]))
                    diag_zbin = np.concatenate((p_noise[index_zbin] * inv_N_eff_gal[index_zbin], p_noise[index_zbin] * inv_N_eff_gal[index_zbin]))
                    if index_zbin == 0:
                        diag_noise = diag_zbin
                        #data_vector = data[index_zbin]
                    else:
                        diag_noise = np.concatenate((diag_noise, diag_zbin))
                        #data_vector = np.concatenate((data_vector, data[index_zbin]))

            full_covariance[index_diag, index_diag] += diag_noise

            dt_mult = timer() - t0_mult
            print 'Time for constructing full covariance: {:.2f}s.'.format(dt_mult)
            logdata['dim_covariance'] = full_covariance.shape

            print 'Dimension of full covariance matrix = {:}x{:}.'.format(full_covariance.shape[0], full_covariance.shape[1])
            t0_inversion = timer()
            # not sure, what .I returns (since method .getI() seems to return pseudo-inverse, if inverse fails...)
            #inv_full_covariance = full_covariance.I
            # Found no difference in using regularization factor or not.
            #mean_diag_cov = np.diag(full_covariance).mean()
            #reg_factor = 2./mean_diag_cov
            #print 'Regularization factor Covariance: {:.4e}'.format(reg_factor)
            #inv_full_covariance = reg_factor*np.linalg.inv(reg_factor*full_covariance)
            inv_full_covariance = inv(full_covariance)
            # Write to disk speeds up multiprocessing:
            np.save('cache/' + 'inv_covariance', inv_full_covariance)
            np.save('cache/' + 'covariance', full_covariance)

            if not ini.do_not_invert_that_matrix:
                print 'Time for matrix inversion: {:.4f}s.'.format(timer() - t0_inversion)

            if ini.numerical_test:
                do_numerical_tests(full_covariance, inv_full_covariance, matrix_is_fisher=False)

            t0_full_fisher = timer()
            print 'First step for Fisher matrix: C^-1 D_A'
            success = compute_inv_covariance_times_deriv_matrices(dim=dim_signals, ncpus=ini.max_ncpus)
            if success:
                print 'Done with first step.'
            # WV NEW asarray:
            if ini.marginalize_over_noise:
                dim_fisher = band_powers.size + p_noise.size #len(inv_covarianceTimesDerivMatrices)
            else:
                dim_fisher = band_powers.size
            fisher, dtf = get_fisher_matrix(dim=dim_fisher, ncpus=ini.max_ncpus)
            print 'Time for second step of calculation of Fisher matrix: {:.2f}s.'.format(dtf)
            print 'Total time for calculation of Fisher matrix: {:.2f}s.'.format(timer() - t0_full_fisher)

            t0_inversion = timer()
            # not sure, what .I returns (since method .getI() seems to return pseudo-inverse, if inverse fails...)
            #inv_fisher = fisher.I
            # Found no difference in using regularization factor or not.
            #mean_diag_fisher = np.diag(Fisher).mean()
            #reg_factor = 2./mean_diag_fisher
            #print 'Regularization factor Fisher: {:.4e}'.format(reg_factor)
            #inv_fisher = reg_factor*la.inv(reg_factor*fisher)
            if not ini.do_not_invert_that_matrix:
                inv_fisher = inv(fisher)
                print 'Time for matrix inversion: {:.4f}s.'.format(timer() - t0_inversion)

            if ini.numerical_test:
                do_numerical_tests(fisher, inv_fisher, matrix_is_fisher=True)
            print 'Started calculation of new band power estimates.'

            if ini.do_not_invert_that_matrix:
                delta_parameters, dt_est = estimate_band_powers(data_vector, full_covariance, fisher, convergence_parameter=lambda0, ncpus=ini.max_ncpus)
            else:
                delta_parameters, dt_est = estimate_band_powers(data_vector, inv_full_covariance, inv_fisher, convergence_parameter=lambda0, ncpus=ini.max_ncpus)

            if ini.marginalize_over_noise:
                delta_band_powers = delta_parameters[:-nzbins]
                delta_p_noise = delta_parameters[-nzbins:]
                p_noise = p_noise + delta_p_noise
            else:
                delta_band_powers = delta_parameters
                delta_p_noise = 0.
                p_noise = 0.

            print 'Time for calculation of new band power estimates: {:2f}s.'.format(dt_est)
            band_powers = band_powers + delta_band_powers
            # TODO: figure out if p_noise should be added to convergence criteria...
            if niterations > 0:
                if ini.marginalize_over_noise:
                    difference_delta_p_noise = np.abs(delta_p_noise + collect_delta_p_noise[niterations - 1]) / np.abs(delta_p_noise)
                    #print collect_delta_band_powers[n-1]
                    print 'Relative cancellation in delta p_noise: \n', difference_delta_p_noise
                    difference_p_noise = np.abs(p_noise - collect_p_noise[niterations - 1]) / np.abs(p_noise)
                    print 'Relative difference in p_noise: \n', difference_p_noise
                    collect_relative_difference_p_noise.append(difference_p_noise)

                difference_delta_band_powers = np.abs(delta_band_powers + collect_delta_band_powers[niterations - 1]) / np.abs(delta_band_powers)
                #print collect_delta_band_powers[n-1]
                print 'Relative cancellation in delta band powers: \n', difference_delta_band_powers
                difference_band_powers = np.abs(band_powers - collect_band_powers[niterations - 1]) / np.abs(band_powers)
                print 'Relative difference in band powers: \n', difference_band_powers
                collect_relative_difference_band_powers.append(difference_band_powers)
                if ini.marginalize_over_noise:
                    mask_bp = (difference_delta_band_powers <= 0.2)
                    mask_noise = (difference_delta_p_noise <= 0.2)
                    if mask_bp.any() or mask_noise.any():
                        lambda0 /= 2.
                else:
                    mask = (difference_delta_band_powers <= 0.2) #& (difference_delta_band_powers >= 0.18)
                    if mask.any():
                        lambda0 /= 2.

                mask = (difference_band_powers[indices_bands_to_use] <= ini.convergence_criterion)
                print 'Relative difference in bands requested for convergence: \n', difference_band_powers[indices_bands_to_use]
                if mask.all():
                    converged = True
                    print 'Convergence within {:.2f}% achieved.'.format(ini.convergence_criterion)

            #band_powers[np.where(band_powers < 0.)] = 1e-7
            print 'New estimates for band powers: \n', band_powers
            if lambda0 < 0.01:
                lambda0 = 0.01

            collect_delta_band_powers.append(delta_band_powers)
            # copy is needed due to resetting of negative band powers!
            collect_band_powers.append(np.copy(band_powers))
            collect_delta_p_noise.append(delta_p_noise)
            # copy is needed due to resetting of negative band powers!
            collect_p_noise.append(np.copy(p_noise))

            # if code has not converged after max_iterations, we stop it nevertheless.
            if niterations == ini.max_iterations - 1:
                converged = True

            print_ascii_art('Time spent in one entire iteration: {:.2f}s.'.format(timer() - t0_loop))
            # increase counter for iterations
            niterations += 1
            # call garbage collector explicitly:
            gc.collect()

        dt_Newton_Raphson = timer() - t0_Newton_Raphson
        logdata['time_opt'] = dt_Newton_Raphson
        print 'Time for Newton-Raphson optimization with {:} iterations for {:} band(s): {:.2f} minutes.'.format(niterations, len(ini.bands), dt_Newton_Raphson / 60.)
        print 'Initial estimate of band powers: \n', collect_band_powers[0]
        print 'Final estimate of band powers: \n', collect_band_powers[-1]
        logdata['initial_guess'] = collect_band_powers[0]
        logdata['niterations'] = niterations
        #print np.shape(collect_band_powers)
        #print collect_band_powers

        # save results and important matrices:
        # TODO: use HDF5 as output!!! --> ONE file to rule them all!
        # TODO: FUNCTION!!!!

        #print collect_inv_Fisher[i], np.shape(collect_inv_Fisher), np.shape(collect_inv_Fisher[i])
        # calculate band power correlation matrix:
        inv_fisher = inv(fisher)
        if ini.marginalize_over_noise:
            print '(p_noise(z_mu) - p_noise0(z_mu)) / p_noise0(z_mu): \n', (p_noise - p_noise0) / p_noise0
            correlation_matrix, dt_correlation_matrix = get_correlation_matrix(inv_fisher)
            fname = 'correlation_matrix_with_noise.dat'
            np.savetxt(fname, correlation_matrix)
            inv_fisher_without_noise = inv_fisher[:-nzbins, :-nzbins]
            fisher_without_noise = fisher[:-nzbins, :-nzbins]
            correlation_matrix, dt_correlation_matrix= get_correlation_matrix(inv_fisher_without_noise)
            fname = 'correlation_matrix.dat'
            np.savetxt(fname, correlation_matrix)
        else:
            correlation_matrix, dt_correlation_matrix= get_correlation_matrix(inv_fisher)
            fname = 'correlation_matrix.dat'
            np.savetxt(fname, correlation_matrix)

        collect_band_powers = np.asarray(collect_band_powers)
        collect_delta_band_powers = np.asarray(collect_delta_band_powers)
        collect_relative_difference_band_powers = np.asarray(collect_relative_difference_band_powers)

        redshift_ids = []
        for index_mu in xrange(nzbins):
            for index_nu in xrange(index_mu+1):
                redshift_ids.append('z{:}xz{:}'.format(index_mu + 1, index_nu + 1))
        for index_band, band in enumerate(ini.bands):
            index_jump = slicing_offset_redshifts[index_band]
            loop_counter = 0
            save_collect_band_powers = collect_band_powers[:,slicing_points_bands[band][0]:slicing_points_bands[band][1]]
            print collect_band_powers.shape
            print save_collect_band_powers.shape

            save_collect_delta_band_powers = collect_delta_band_powers[:, slicing_points_bands[band][0]:slicing_points_bands[band][1]]
            save_collect_relative_difference_band_powers = collect_relative_difference_band_powers[:, slicing_points_bands[band][0]:slicing_points_bands[band][1]]
            for index_z in np.arange(0, nzcorrs * slicing_offset_redshifts[index_band], slicing_offset_redshifts[index_band]):

                fname = 'all_estimates_band_powers_' + band + '_' + redshift_ids[loop_counter] + '.dat'
                np.savetxt(fname, save_collect_band_powers[:, index_z:index_jump])
                fname = 'band_powers_' + band + '_' + redshift_ids[loop_counter] + '.dat'
                np.savetxt(fname, save_collect_band_powers[-1, index_z:index_jump])
                fname = 'all_delta_band_powers_' + band + '_' + redshift_ids[loop_counter] + '.dat'
                np.savetxt(fname, save_collect_delta_band_powers[:, index_z:index_jump])
                fname = 'difference_band_powers_' + band + '_' + redshift_ids[loop_counter] + '.dat'
                np.savetxt(fname, save_collect_relative_difference_band_powers[:, index_z:index_jump])

                index_jump += slicing_offset_redshifts[index_band]
                loop_counter += 1

        if ini.marginalize_over_noise:
            fname = 'all_estimates_p_noise.dat'
            np.savetxt(fname, np.asarray(collect_p_noise))
            fname = 'all_delta_p_noise.dat'
            np.savetxt(fname, np.asarray(collect_delta_p_noise))
            fname = 'difference_p_noise.dat'
            np.savetxt(fname, np.asarray(collect_relative_difference_p_noise))
            fname = 'p_noise.dat'
            header = ''
            for index_zbin in xrange(nzbins):
                header += 'p_noise_z{:}, '.format(index_zbin + 1)
            np.savetxt(fname, np.column_stack(p_noise), header=header[:-1])
            fname = 'last_Fisher_matrix_with_noise.dat'
            # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
            # that is why we multiply it with 0.5 before saving!
            np.savetxt(fname, 0.5 * fisher)
            fname = 'last_Fisher_matrix.dat'
            # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
            # that is why we multiply it with 0.5 before saving!
            np.savetxt(fname, 0.5 * fisher_without_noise)
            fname = 'last_larger_Fisher_matrix.dat'
            # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
            # that is why we multiply it with 0.5 before saving!
            np.savetxt(fname, fisher_without_noise)
            # for the BWM calculation we don't want to keep the noise-parameters in the Fisher matrices!
            if ini.do_not_invert_that_matrix:
                parameter_covariance = fisher_without_noise
            else:
                parameter_covariance = inv_fisher_without_noise
        else:
            fname = 'last_Fisher_matrix.dat'
            # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
            # that is why we multiply it with 0.5 before saving!
            np.savetxt(fname, 0.5 * fisher)
            fname = 'last_larger_Fisher_matrix.dat'
            # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
            # that is why we multiply it with 0.5 before saving!
            np.savetxt(fname, fisher)
            if ini.do_not_invert_that_matrix:
                parameter_covariance = fisher
            else:
                parameter_covariance = inv_fisher

        # this matrix can become quite large, so we use HDF5 just in case:
        fname = 'last_covariance_matrix.h5'
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('covariance', data=np.asarray(full_covariance))

    ### END of Newton-Raphson and saving ###

    # call garbage collector explicitly:
    gc.collect()

    # we implicitly assume in the calculation of band window functions that we always have extracted EE AND BB-modes!!!
    # TODO: Generalize for EE, BB, and EB!
    # thus no loop over bands included (yet)...
    # TODO: make this work with different run- and store paths!
    # if everything is extracted at once, this should work as it is
    # if we extract BWM independent from optimization, I have to copy everything to run path first (if I want to use the SSD and if all files fit on there...)
    # if 'and' condition isn't met, the signal extraction most likely didn't converge and we don't want to waste any time on the BWM then
    if ini.band_window and niterations < ini.max_iterations - 1:
        t0_band_window = timer()
        # here we load everything needed for calculation of BWM under the assumption that optimization was done before.
        # but only if we did not optimize in same run
        if not ini.optimize and not ini.calculate_only_signal_matrices_for_band_window_matrix:
            try:
                # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
                # but for subsequent calculations we expect to load the inverse of the too large Fisher matrix, that is why we load this file below:
                print 'The final Fisher matrix will be used to calculate the band power covariance!'
                fname = 'last_larger_Fisher_matrix.dat'
                last_Fisher = np.loadtxt(fname)
                # load also the last covariance matrix:
                fname = 'last_covariance_matrix.h5'
                with h5py.File(fname, 'r') as hf:
                    full_covariance = np.array(hf.get('covariance'))

                if ini.do_not_invert_that_matrix:
                    parameter_covariance = last_Fisher
                    # write last covariance to cache:
                    np.save('cache/' + 'covariance', full_covariance)
                else:
                    parameter_covariance = inv(last_Fisher)
                    # write last C^-1 to cache:
                    inv_full_covariance = inv(full_covariance)
                    # it must be written into cache:
                    np.save('cache/' + 'inv_covariance', inv_full_covariance)
            except:
                print 'Could not load expected data for calculation of band window matrix.'
                exit()

        # here we try to load signal matrices (in case they've been calculated already in previous runs
        # \ell-nodes are always the same:
        # the '+1' is required because of rounding errors; now ells contains ell_min and ell_max?!
        # at least for ell_min = 30, ell_max = 2600
        # find a better way to create a log-spaced list of integers...
        # log-spaced is really dangerous for low multipoles... (might introduce repeating \ells)
        #ells = np.logspace(np.log10(ini.ell_min+1), np.log10(ini.ell_max+1), ini.nell).astype('int')
        #ells = np.linspace(ini.ell_min, ini.ell_max, ini.nell).astype('int')
        # new function that created log-spaced integers without duplicatons:
        ells = gen_log_space(ini.ell_min, ini.ell_max+1, ini.nell)
        fname = 'multipole_nodes_for_band_window_functions_nell{:}.dat'.format(ini.nell)
        np.savetxt(fname, ells)
        logdata['fname_ell_values_band_window'] = fname

        # new approach: keep bands separate again; doubles runtime unfortunately...
        if not os.path.isdir(path_to_signal):
            os.makedirs(path_to_signal)

        # if working with mock data the initial guess for the signal matrix can be shared (if all fields look the same) to save runtime!
        if ini.share_signal_matrices:
            identifier_SM = ini.shared_identifier_SM
        else:
            identifier_SM = ini.identifier_in

        # using HDF5
        t0_signal_matrices_for_bwm = timer()

        for index_band, band in enumerate(ini.bands):

            fname = path_to_signal + 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '_BWM_nell{:}.h5'.format(ini.nell)

            if not os.path.isfile(fname): # + '.npy'):
                cmd = 'w'
            else:
                cmd = 'a'

            # calculate (derivative) signal matrices:
            for index_ell in xrange(ini.nell):

                fname_matrix = 'signal_matrix_index_ell{:03d}'.format(index_ell)

                # check if matrices already exist:
                try:
                    with h5py.File(fname, 'r') as hf:
                        matrix_names = hf.keys()

                    if fname_matrix in matrix_names:
                        matrix_already_exists = True
                    else:
                        matrix_already_exists = False

                except:
                    matrix_already_exists = False

                if (cmd == 'w') or ((cmd == 'a') and not matrix_already_exists):

                    # these are GLOBAL, together with "band"
                    print 'Calculating matrix of band window functions {:}/{:} with {:} <= ell <= {:}.'.format(index_ell + 1, ini.nell, ini.ell_min, ini.ell_max)
                    print 'Multipole ell = {:}'.format(ells[index_ell])
                    # Unfortunately this is done now at each initialization of SignalMatrix (due to restructuring because of pickle-issue with multiprocessing)
                    print 'Calculating moments of window function.'
                    t0_wf = timer()
                    dSM_ell = sm.SignalMatrix(band_min=ells[index_ell], band_max=0., band=band, sigma=ini.side_length, integrate=False, ncpus=ini.ncpus_BWM)
                    print 'Done. Time: {:.2f}s.'.format(timer() - t0_wf)
                    if ini.calculate_SM_fast:
                        print 'Fast calculation.'
                        #signal_ell.append(dSM_ell.getSignalMatrixFAST(x_rad, y_rad))
                        signal_ell = dSM_ell.getSignalMatrixFAST(x_rad, y_rad)
                    else:
                        print 'Slow calculation.'
                        signal_ell = dSM_ell.getSignalMatrix(x_rad, y_rad)

                    with h5py.File(fname, cmd) as hf:
                    #group = hf.create_group(band)
                        hf.create_dataset(fname_matrix, data=signal_ell)
                    # if it was 'w', set to 'a'!
                    cmd = 'a'

        print 'Time for calculation of {:} signal matrices for band window matrix: {:.2f}min.'.format(len(ini.bands) * ini.nell, (timer() - t0_signal_matrices_for_bwm) / 60.)
        if ini.calculate_only_signal_matrices_for_band_window_matrix:
            print 'Done with calculating signal matrices for band window matrix.'
            if ini.root_signal != ini.root_signal_store:
                t0_store = timer()
                print 'Copying signal matrix files now to: \n', path_to_signal_store
                import shutil
                for index_band, band in enumerate(ini.bands):
                    fname = 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '_BWM_nell{:}.h5'.format(ini.nell)
                    shutil.move(path_to_signal + fname, path_to_signal_store)
                #os.chdir(path_to_signal_store)
                #shutil.rm(path_to_signal + fname)
                print 'Done in {:.2f}s.'.format(timer() - t0_store)
            exit()

        t0_band_window_matrix = timer()

        # included more sophisticated loop in order to get the right shape!
        # or would index-array be more efficient?!
        # save directly to disk in loop!
        #collect_signals_l = []
        # this implementation is stupid...
        # clever indexing and coding up of matrix-multiplication would be much better...
        # new approach with HDF5:
        index_linear = 0
        for index_band, band in enumerate(ini.bands):

            fname_SM = fname = path_to_signal + 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '_BWM_nell{:}.h5'.format(ini.nell)

            for index_zcorr in xrange(nzcorrs): #len(redshift_bins)**2):
                for index_ell in xrange(ini.nell):

                    # check if derivative matrices exist already (e.g. for BWM run):
                    fname_deriv = 'cache/deriv_matrix_ell_index_lin{:03d}.npy'.format(index_linear)

                    if os.path.isfile(fname_deriv):
                        # this is only needed once for "dims_single_deriv"
                        if index_linear == 0:
                            matrix = np.load(fname_deriv)
                        index_linear += 1
                    # if not, calculate them:
                    else:

                        with h5py.File(fname_SM, 'r') as hf:
                            #group = hf.get(band)
                            signal_block = np.array(hf.get('signal_matrix_index_ell{:03d}'.format(index_ell)))
                        zero_block = np.zeros_like(signal_block)
                        # this is the correct way to build up the list of derivative matrices!
                        # ATTENTION: the ordering here must be consistent with ordering in Fisher matrix!
                        if nzcorrs == 1:
                            # for general compatibility with bracket notation in 'load', save matrix as sparse matrix, although it's not sparse...
                            if ini.use_sparse_matrices:
                                matrix = sparse.bsr_matrix(signal_block)
                            else:
                                matrix = signal_block
                        else:
                            # we make use of sparse matrices --> requires peculiar bracket notation '[()]' in 'load' in order to restore sparse!
                            matrix_string = get_matrix_string(nzbins, index_map[index_zcorr])
                            if ini.use_sparse_matrices:
                                matrix = sparse.bsr_matrix(np.bmat(matrix_string))
                            else:
                                matrix = np.bmat(matrix_string)
                        # save to disk:
                        np.save('cache/deriv_matrix_ell_index_lin{:03d}'.format(index_linear), matrix)
                        index_linear += 1

        # now we have to use this list of list of matrices to construct the needed list of matrices:
        # according to revised formula also the derivatives wrt \ell depend on redshift!
        print 'Linear index: {:}'.format(index_linear)
        # call garbage collector explicitly:
        gc.collect()

        #TODO assuming that if index000 exists all other files exist, too:
        if not os.path.isfile('cache/inv_covariance_times_deriv_matrix_index000.npy'):
            print 'Recalculating C^-1 D_alpha.'
            success = compute_inv_covariance_times_deriv_matrices(dim=index_signals, ncpus=ini.max_ncpus, for_band_window=False)

            if success:
                print 'Done with calculation of C^-1 D_alpha.'
        else:
            print 'Skipping calculation of C^-1 D_alpha. Files exist already.'

        if not os.path.isfile('cache/inv_covariance_times_deriv_matrix_ell_transposed_index_ell000.npy'):
            print 'First step for band window matrix: C^-1 D_ell'
            success = compute_inv_covariance_times_deriv_matrices(dim=index_linear, ncpus=ini.max_ncpus, for_band_window=True)

            if success:
                print 'Done with first step.'
        else:
            print 'Skipping calculation of C^-1 D_ell. Files exist already.'

        # call garbage collector explicitly:
        gc.collect()

        print 'Second step for calculation of band window matrix.'

        # much bigger dimension now in order to also get the cross-terms (e.g. EE --> BB):
        # this might be the same as "dim_signals_ell" actually... Indeed it is!
        #dim_ell = int(len(ini.bands) * nzcorrs * ini.nell)
        #print 'dim_ell = {:}'.format(dim_ell)

        # inv_covarianceTimesDerivMatrices MUST be the ones used in Fisher algorithm and NOT inv_covarianceTimesDerivMatrices_l!!!
        trace_matrix, dt_trace = get_trace_matrix(dim_alpha=len(parameter_covariance), dim_ell=index_linear, ncpus=ini.max_ncpus)
        print 'Time for second step of calculation of band window matrix: {:.2f}min.'.format(dt_trace / 60.)

        window_matrix, dt_window_matrix = get_window_matrix(parameter_covariance, trace_matrix)

        fname = 'band_window_matrix_nell{:}.dat'.format(ini.nell)
        np.savetxt(fname, np.asarray(window_matrix))
        print 'Expected shape W_alpha_ell: {:}x{:}'.format(len(parameter_covariance), ini.nell)
        print 'Actual shape W_alpha_ell:', np.shape(window_matrix)
        print 'Time for calculation of matrix of band window functions: {:.2f}min.'.format(dt_window_matrix / 60.)

        dt_band_window = timer() - t0_band_window
        print 'Total time for calculation of band window matrix: {:.2f}min.'.format(dt_band_window / 60.)
        logdata['time_WM'] = dt_band_window
    runtime = timer() - t0_prog
    print 'Done. Total runtime: {:.2f}min.'.format(runtime / 60.)
    logdata['runtime'] = runtime
    # easy way around log-files:
    # that means there's no log-file in case of only extracting the band window function!
    if ini.optimize:
        write_logfile(path_to_dir_of_run + '/' + fname_input, fname_logfile, logdata, field_properties)
    print 'All results were saved to: \n', path_output
    # last step: removal of 'cache'-folder if requested:
    if ini.delete_cache:
        print 'Removing cache folder now.'
        import shutil
        shutil.rmtree('cache/')

    if ini.root_run != ini.root_store:
        t0_store = timer()
        print 'Copying all files now to: \n', path_store
        import shutil
        shutil.copytree(path_output, path_store)
        os.chdir(path_store)
        shutil.rmtree(path_output)
        print 'Done in {:.2f}s.'.format(timer() - t0_store)
    else:
        print 'Done.'