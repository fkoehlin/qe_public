#!/usr/bin/python
# encoding: UTF8
"""
.. module:: quadratic_estimator
    :synopsis: Main module
.. moduleauthor:: Fabian Koehlinger <fabian.koehlinger@ipmu.jp>
.. moduleauthor:: Wessel Valkenburg <valkenburg@lorentz.leidenuniv.nl>

Code for estimating maximum likelihood band powers of the weak gravitational
lensing power spectrum, a.k.a cosmic shear, following the 'quadratic estimator'
method by Hu & White 2001 (ApJ, 554, 67) extended to also take into account
tomographic bins. Moreover, the band window function matrix estimation as
described in Lin et al. 2012 (ApJ, 761, 15) is also included.

This is the main module, call it like this:
   `python /path/to/quadratic_estimator.py /path/to/input_parameters.ini > logfile.log`
"""
from __future__ import print_function
import os
#import gc
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
from utils import print_ascii_art

# Python 2.x - 3.x compatibility: Always use more efficient range function
try:
    xrange
except NameError:
    xrange = range

def compute_inv_covariance_times_deriv_matrices(dim, ncpus=1, for_band_window=False):
    """
    compute_inv_covariance_times_deriv_matrices(dim, ncpus=1, for_band_window=False)

    Wrapper function for calculating the following matrix products:
        a) C^{-1}_{ij} (C,\alpha)_{jk} or
        b) C^{-1}_{ij} (C,\ell)_{jk}
    by selecting the corresponding helper function and by using Python's multiprocessing module.
    Due to potentially large matrix sizes the results will be saved to disk by the helper functions.

    Parameters
    ----------
    dim : int
                The dimension n of the nxn resulting matrix .

    ncpus : int, optional
                The number of CPUs mukltiprocessing will be requesting.

    for_band_windows : bool, optional
                Switch between the corresponding helper functions:
                    True  --> selects option b)
                    False --> selects option a)

    Returns
    -------
    success : bool
                Variable is True if calculation succeeded, False otherwise.
    """

    t0 = timer()

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

    print('Running on {:} processes...'.format(pool_size))

    idx_array = range(dim)

    # Go compute!
    pool = multiprocessing.Pool(processes = pool_size)
    if for_band_window:
        loops_were_success = pool.map(single_step_compute_inv_covariance_times_deriv_matrices_ell, idx_array)
    else:
        loops_were_success = pool.map(single_step_compute_inv_covariance_times_deriv_matrices, idx_array)
    pool.close()

    if np.asarray(loops_were_success).all():
        success = True
    else:
        success = False

    dt = timer() - t0

    print('Computation done in {:.2f}s.'.format(dt))

    return success


def single_step_compute_inv_covariance_times_deriv_matrices(idx):
    """
    single_step_compute_inv_covariance_times_deriv_matrices(idx)

    Helper function for calculating one element of the matrix product:
        C^{-1}_{ij} (C,\alpha)_{jk}.
    Due to potentially large matrix sizes the result will be saved to disk.

    Parameters
    ----------
    idx : int
                Current index of the matrix element to be calculated in this function.

    Returns
    -------
    success : bool
                Variable is True if calculation succeeded, False otherwise.
    """

    #t0 = timer()

    success = False

    # weird bracket notation '[()]' necessary in order to restore sparse matrix
    if ini.use_sparse_matrices:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_idx{:03d}.npy'.format(idx))[()].toarray()
    else:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_idx{:03d}.npy'.format(idx))

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

    np.save('cache/' + 'inv_covariance_times_deriv_matrix_idx{:03d}'.format(idx), matrix)
    np.save('cache/' + 'inv_covariance_times_deriv_matrix_transposed_idx{:03d}'.format(idx), matrix_T)

    #print('Multiplication done in {:.2f}s.'.format(timer()-t0))
    #print_dot()

    success = True

    return success


def single_step_compute_inv_covariance_times_deriv_matrices_ell(idx):
    """
    single_step_compute_inv_covariance_times_deriv_matrices_ell(idx)

    Helper function for calculating one element of the matrix product C^{-1}_{ij} (C,\ell)_{jk}.
    Due to potentially large matrix sizes the result will be saved to disk.

    Parameters
    ----------
    idx : int
                Current index of the matrix element to be calculated in this function.

    Returns
    -------
    success : bool
                Variable is True if calculation succeeded, False otherwise.
    """

    success = False

    if ini.use_sparse_matrices:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_ell_idx_lin{:03d}.npy'.format(idx))[()].toarray()
    else:
        deriv_matrix = np.load('cache/' + 'deriv_matrix_ell_idx_lin{:03d}.npy'.format(idx))

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
    np.save('cache/' + 'inv_covariance_times_deriv_matrix_ell_transposed_idx_ell{:03d}'.format(idx), matrix)

    #print('Multiplication done in {:.2f}s.'.format(timer()-t0))
    #print_dot()

    success = True

    return success


def get_fisher_matrix(dim, ncpus=1):
    """
    get_fisher_matrix(dim, ncpus=1)

    Wrapper function for calculating the Fisher matrix
        F_{\alpha \beta} = Tr[ C^{-1}_{ij} (C,\alpha)_{jk} C^{-1}_{kl} (C,\beta)_{lm} ]

    IMPORTANT: the Fisher matrix calculated here, is actually 2 times larger compared to
               the definition of the Fisher matrix in Eq. (20) of Hu & White (2001)
               This factor is taken care of in all other subsequent equations!

    Parameters
    ----------
    dim : int
                The dimension n of the nxn Fisher matrix .

    ncpus : int, optional
                The number of CPUs mukltiprocessing will be requesting.

    Returns
    -------
    fisher : 2D numpy.ndarray
                Contains final nxn Fisher matrix.

    dt : float
                The time needed for calculating the Fisher matrix.
    """

    t0 = timer()

    fisher = np.zeros((dim, dim))

    # prepare the pool's size:
    nfisher = (dim * (dim + 1)) // 2

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

    print('Number of processes requested for last step in Fisher matrix computation: {:}'.format(pool_size))

    # Now new approach:
    # create only an idx_array (for loading needed matrices directly in parallel function)
    idx_array = []
    for alpha in xrange(dim):
        for beta in xrange(alpha + 1):
            idx_array.append([alpha, beta])

    print('Collecting Fisher matrix...')

    pool = multiprocessing.Pool(processes = pool_size)
    fisher_as_list = pool.map(single_step_get_fisher_matrix, idx_array)
    pool.close()

    # bring list into matrix shape
    idx = 0
    for alpha in xrange(dim):
        for beta in xrange(alpha + 1): # 0 < b <= alpha.
            fisher[alpha, beta] = fisher_as_list[idx]
            fisher[beta, alpha] = fisher[alpha, beta]
            idx += 1

    dt = timer() - t0

    return fisher, dt


def single_step_get_fisher_matrix(idx_pair):
    """
    single_step_get_fisher_matrix(idx_pair)

    Helper function for calculating a single element of the Fisher matrix
        F_{\alpha \beta} = Tr[ C^{-1}_{ij} (C,\alpha)_{jk} C^{-1}_{kl} (C,\beta)_{lm} ]

    Parameters
    ----------
    idx_pair : set of two int
                The unique (\alpha, \beta) index pair of the element to be calculated.

    Returns
    -------
    total_sum : float
                The element F_{\alpha \beta} of the Fisher matrix.
    """

    if ini.use_sparse_matrices:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_idx{:03d}.npy'.format(idx_pair[0]))[()].toarray()
        inv_covariance_times_derivative_B_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_idx{:03d}.npy'.format(idx_pair[1]))[()].toarray()
    else:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_idx{:03d}.npy'.format(idx_pair[0]))
        inv_covariance_times_derivative_B_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_idx{:03d}.npy'.format(idx_pair[1]))

    # NOTE: second matrix is transposed, and this is an ARRAY multiplication (by element), NOT MATRIX.
    total_sum = np.sum(inv_covariance_times_derivative_A * inv_covariance_times_derivative_B_transposed)

    return total_sum


def estimate_band_powers(data_vector_a, data_vector_b, covariance, fisher, convergence_parameter=1., ncpus=1):
    """
    estimate_band_powers(data_vector_a, data_vector_b, covariance, fisher, convergence_parameter=1., ncpus=1)

    Wrapper function for calculating an update \delta_p_\alpha for the band powers p_\alpha:
        \delta_p_\alpha = \lambda \sum_\beta [ F^{-1}_{\alpha \beta} Tr[ (d d^T - C) ( C^{-1} (C,\beta) C^{-1} )]]

    IMPORTANT: the Fisher matrix calculated in the code is two times larger than the definition in Eq. (20) of
               Hu & White (2001) therefore the factor 0.5 in Eq. (19) of Hu & White (2001) does not appear in the
               equation above (as it cancels when propagating the factor through the inverse of the Fisher matrix).

    Parameters
    ----------
    data_vector_a : 1D numpy.ndarray
                The data vector of z-bin <m>.

    data_vector_b : 1D numpy.ndarray
                The data vector of z-bin <n>.

    covariance : 2D numpy.ndarray
                The data covariance matrix stored as an array if ini.do_not_invert_that_matrix = True
                The inverse of the data covariance if ini.do_not_invert_that_matrix = False

    fisher : 2D numpy.ndarray
                The Fisher matrix stored as an array if ini.do_not_invert_that_matrix = True
                The inverse of the Fisher matrix if ini.do_not_invert_that_matrix = False

    convergence_parameter : float, optional
                A scaling factor, \lambda, for the calculated band power corrections.

    ncpus : int, optional
                The number of CPUs mukltiprocessing will be requesting.

    Returns
    -------
    delta_powers : 1D numpy.ndarray
                The elements \delta_p_\alpha of the band power update.

    dt : float
                The time needed for calculating the band power updates.
    """
    t0 = timer()
    dim = fisher.shape[0]
    delta_powers = np.zeros((dim))
    # WV NEW:
    # Compute C^-1_{ij} d_j d_k
    dim_data = len(data_vector_a)
    inv_covariance_times_data_matrix = np.zeros((dim_data, dim_data))
    t0_step1 = timer()

    if ini.do_not_invert_that_matrix:
        inv_covariance_times_data_vector_a = solve(covariance, data_vector_a)
    else:
        # we expect that the inv_covariance is passed as covariance!!!
        inv_covariance_times_data_vector_a = covariance.dot(data_vector_a)

    for i in xrange(dim_data):
        for j in xrange(dim_data):
            inv_covariance_times_data_matrix[i, j] = inv_covariance_times_data_vector_a[i] * data_vector_b[j]
    np.save('cache/' + 'inv_covariance_times_data_matrix', inv_covariance_times_data_matrix)
    print('First step in band powers update done in {:.2f}s.'.format(timer() - t0_step1))

    t0_step2 = timer()

    idx_array = range(dim)

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

    print(r'Computing Tr[ (d d^T - C) ( C^{-1} (C,\beta) C^{-1} )] = Tr[ (C^{-1} d d^T) (C^{-1} (C,\beta)) ] - Tr[ (C^{-1} (C,\beta)) ]...')
    pool = multiprocessing.Pool(processes = pool_size)
    trace_sum_vector = pool.map(single_step_estimate_band_powers, idx_array)
    pool.close()

    if ini.do_not_invert_that_matrix:
        delta_powers = convergence_parameter * solve(fisher, trace_sum_vector)
    else:
        # we expect that the inv_fisher is passed as fisher!!!
        delta_powers = convergence_parameter * fisher.dot(trace_sum_vector)

    print('Second step in band powers update done in {:.2f}s.'.format(timer() - t0_step2))

    dt = timer() - t0

    return delta_powers, dt


def single_step_estimate_band_powers(idx):
    """
    single_step_estimate_band_powers(idx)

    Helper function for calculating one element of:
        Tr[ (d d^T - C) ( C^{-1} (C,\beta) C^{-1} )] = Tr[ (C^{-1} d d^T) (C^{-1} (C,\beta)) ] - Tr[ (C^{-1} (C,\beta)) ]...

    IMPORTANT: the Fisher matrix calculated in the code is two times larger than the definition in Eq. (20) of
               Hu & White (2001) therefore the factor 0.5 in Eq. (19) of Hu & White (2001) does not appear in the
               equation above (as it cancels when propagating the factor through the inverse of the Fisher matrix).

    Parameters
    ----------
    idx : int
                The index, \beta, of the trace element to be calculated.

    Returns
    -------
    sum_trace : float
                The value of the trace element.
    """
    inv_covariance_times_data_matrix = np.load('cache/' + 'inv_covariance_times_data_matrix.npy')

    if ini.use_sparse_matrices:
        inv_covariance_times_derivative_beta_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_idx{:03d}.npy'.format(idx))[()].toarray()
    else:
        inv_covariance_times_derivative_beta_transposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_transposed_idx{:03d}.npy'.format(idx))

    sum_trace = 0.
    sum_trace = np.sum(inv_covariance_times_data_matrix * inv_covariance_times_derivative_beta_transposed)
    sum_trace -= np.trace(inv_covariance_times_derivative_beta_transposed)

    return sum_trace


def get_window_matrix(fisher, trace):
    """
    get_window_matrix(fisher, trace)

    Function for calculating the window matrix of band window functions (cf. Eq. 21 in Lin et al. 2012):
        W_{\alpha \ell} = F^{-1}_{\alpha \beta} C_{\beta \ell}

    IMPORTANT: the Fisher matrix calculated in the code is two times larger than the definition in Eq. (20) of
               Hu & White (2001) therefore the factor 0.5 in Eq. (21) of Lin et al. (2012) does not appear in the
               equation above (as it cancels when propagating the factor through the inverse of the Fisher matrix).

    Parameters
    ----------
    fisher : 2D numpy.ndarray
                The Fisher matrix stored as a 2D array if ini.do_not_invert_that_matrix = True
                The inverse of the Fisher matrix if ini.do_not_invert_that_matrix = False

    trace : 2D numpy.ndarray
                The trace matrix stored as a 2D array.

    Returns
    -------
    window : 2D numpy.ndarray
                Contains final window matrix.

    dt : float
                The time needed for calculating the window matrix.
    """
    t0 = timer()

    if ini.do_not_invert_that_matrix:
        window = solve(fisher, trace)
    else:
        # we expect that the inv_fisher is passed as fisher!!!
        window = fisher.dot(trace)

    dt = timer() - t0

    return window, dt


def get_trace_matrix(dim_alpha, dim_ell, ncpus=1):
    """
    get_trace_matrix(dim_alpha, dim_ell, ncpus=1)

    Wrapper function for calculating the trace matrix needed for the calculation of the
    matrix of band window functions (cf. Eq. 21 in Lin et al. 2012):
        T_{\beta \ell} = Tr[ C^{-1}_{ij} (C,\beta)_{jk} C^{-1}_{kl} (C,\ell)_{lm} ]

    Parameters
    ----------
    dim_alpha : int
                The dimension n of the nxm trace matrix.

    dim_ell : int
                The dimension m of the nxm trace matrix.

    ncpus : int, optional
                The number of CPUs mukltiprocessing will be requesting.

    Returns
    -------
    trace : 2D numpy.array
                Contains final mxn trace matrix.

    dt : float
                The time needed for calculating the trace matrix.
    """
    t0 = timer()

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

    print('Number of processes requested for last step in Fisher matrix computation: {:}'.format(pool_size))

    # Create an idx_array (for loading needed matrices directly into helper function)
    idx_array = []
    for idx_alpha in xrange(dim_alpha):
        for idx_ell in xrange(dim_ell):
            idx_array.append((idx_alpha, idx_ell))

    print('Calculating trace matrix now...')

    pool = multiprocessing.Pool(processes = pool_size)
    trace_as_list = pool.map(single_step_get_trace_matrix, idx_array)
    pool.close()

    trace = np.asarray(trace_as_list).reshape((dim_alpha, dim_ell))

    dt = timer() - t0

    return trace, dt

def single_step_get_trace_matrix(idx_pair):
    """
    single_step_get_trace_matrix(idx_pair)

    Helper function for calculating a single element of the trace matrix needed for the calculation of the
    matrix of band window functions (cf. Eq. 21 in Lin et al. 2012):
        T_{\beta \ell} = Tr[ C^{-1}_{ij} (C,\beta)_{jk} C^{-1}_{kl} (C,\ell)_{lm} ]

    Parameters
    ----------
    idx_pair : set of two int
                The indices \alpha, \ell of the current trace matrix element.

    Returns
    -------
    total_sum : float
                Contains value of trace matrix element \alpha, \ell.
    """
    #t0 = timer()
    if ini.use_sparse_matrices:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_idx{:03d}.npy'.format(idx_pair[0]))[()].toarray()
        inv_covariance_times_derivative_ell_tranposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_ell_transposed_idx_ell{:03d}.npy'.format(idx_pair[1]))[()].toarray()
    else:
        inv_covariance_times_derivative_A = np.load('cache/' + 'inv_covariance_times_deriv_matrix_idx{:03d}.npy'.format(idx_pair[0]))
        inv_covariance_times_derivative_ell_tranposed = np.load('cache/' + 'inv_covariance_times_deriv_matrix_ell_transposed_idx_ell{:03d}.npy'.format(idx_pair[1]))

    # NOTE: second matrix is transposed, and this is an ARRAY multiplication (by element), NOT MATRIX.
    total_sum = np.sum(inv_covariance_times_derivative_A * inv_covariance_times_derivative_ell_tranposed)

    return total_sum


def get_correlation_matrix(covariance_matrix):
    """
    get_correlation_matrix(covariance_matrix)

    Function for calculating the correlation matrix (bounded between -1 and 1) of the
    input covariance matrix:
        Corr_{ij} = Cov_{ij} / sqrt(Cov_{ii} Cov_{jj})

    Parameters
    ----------
    covariance_matrix : 2D numpy.ndarray
                The array containing the covariance matrix.

    Returns
    -------
    corr : 2D numpy.ndarray
                The array containing the correlation matrix.

    dt : float
                The time needed for calculating the trace matrix.
    """
    t0 = timer()

    corr = covariance_matrix / np.sqrt(np.asmatrix(np.diag(covariance_matrix)).T * np.asmatrix(np.diag(covariance_matrix)))

    dt = timer() - t0

    return corr, dt


def initialize_band_powers(bands_min, bands_max, band='EE', constant=True):
    """
    initialize_band_powers(bands_min, bands_max, band='EE', constant=True)

    Function returning an 'initial guess' for the {'EE', 'BB', 'EB'} band
    powers.
    If 'constant=True', the {'BB', 'EB'} band powers are set to zero for all multipoles and 'EE'
    band powers are set to 10^{-6} for all multipoles.
    If 'constant=False', the power spectra file specified in the variable 'theory_for_initial_guess_file'
    in the '.ini' - file is read in and scaled with a random amplitude between 0.1 and 10 times the original
    signal. {'BB', 'EB'} band powers will be drawn from narrow Gaussian centred on zero and scaled between 10^{-16}
    and 10^{-15}.

    Parameters
    ----------
    bands_min : 1D numpy.ndarray
                The array containing the lower limits of the band power multipole intervals.

    bands_max : 1D numpy.ndarray
                The array containing the upper limits of the band power multipole intervals.

    band : string, optional
                The type of band powers, {'EE', 'BB', 'EB'}, for which the initial guess is
                produced.

    constant : bool, optional
                If set to True, the {'BB', 'EB'} band powers are set to zero for all multipoles and 'EE'
                band powers are set to 10^{-6} for all multipoles.
                If set to False, the power spectra file specified in the variable 'theory_for_initial_guess_file'
                in the '.ini' - file is read in and scaled with a random amplitude between 0.1 and 10 times the original
                signal. {'BB', 'EB'} band powers will be drawn from narrow Gaussian centred on zero and scaled between 10^{-16}
                and 10^{-15}.

    Returns
    -------
    guess : 1D numpy.ndarray
                The array containing the initial guess for the selected band powers.

    """
    ells = (bands_min + bands_max) / 2.

    if constant:

        if band == 'EE':
            guess = np.ones_like(ells) * 1e-6
        else:
            #guess = np.ones_like(ells) * 1e-12
            # zeros are fine, since we always add noise to the main diagonal (and block off-diagonals can be 0, esp. for B-modes!)
            guess = np.zeros_like(ells)
            #guess = np.ones_like(ells) * 1e-12
    else:

        if band == 'EE':
            # load a real cosmological signal
            ells_theo, C_ells = np.loadtxt(ini.theory_for_initial_guess_file, usecols=(0, 1), unpack=True)

            a = 0.1
            b = 10.
            amp = (b - a) * np.random.random() + a
            # scale it with a random amplitude a <= amp <= b:
            guessed_PS = interpolate.interp1d(ells_theo, C_ells * amp)
            guess = guessed_PS(ells)
        else:
            a = 1e-16
            b = 1e-15
            # draw from a Gaussian and scale B-modes to positive interval close to zero
            guess = (b - a) * np.random.normal(0., scale=0.1, size=ells.size) + a

    return guess

def get_initial_band_powers(bands_min, bands_max, idx_zbin1, idx_zbin2):
    """
    get_initial_band_powers(bands_min, bands_max, idx_zbin1, idx_zbin2)

    Function supplying the initial guess for the requested {'EE', 'BB', 'EB'} band
    powers for the current redshift bin correlation idx_zbin1 x idx_zbin2.

    Parameters
    ----------
    bands_min : 1D numpy.ndarray
                Array containing the lower limits of the band power multipole intervals.

    bands_max : 1D numpy.ndarray
                Array containing the upper limits of the band power multipole intervals.

    idx_zbin1 : int
                Index of the first z-bin to be used in the analysis.

    idx_zbin2 : int
                Index of the second z-bin to be used in the analysis.

    Returns
    -------
    initial_band_powers : 1D numpy.ndarray
                Array containing the initial guesses for all requested band powers

    indices_bands_to_use : 1D numpy.ndarray
                Array containing the indices (with respect to 'initial_band_powers') for the bands
                that will be checked for convergence.

    slicing_points_bands : dict
                Dictionary containing the sets of indices at which to split 'initial_band_powers' into
                the single EE, BB, EB components.
    """
    # use a read-in power spectrum with random but realistic amplitude scaling:
    initial_band_powers_EE = initialize_band_powers(bands_EE_min, bands_EE_max, band='EE', constant=False)
    # set BB and EB to constant zero for now!
    initial_band_powers_BB = initialize_band_powers(bands_BB_min, bands_BB_max, band='BB', constant=True)
    initial_band_powers_EB = initialize_band_powers(bands_EB_min, bands_EB_max, band='EB', constant=True)

    initial_band_powers_list = [initial_band_powers_EE, initial_band_powers_BB, initial_band_powers_EB]

    # More correct: initial_EE_z1z1, initial_EE_z1z2, initial_EE_z2z2; initial_BB_z1z1, initial_BB_z1z2, initial_BB_z2z2; since initial_BB_z1z2 = initial_BB_z2z1
    for idx_band in xrange(len(ini.bands)):
        if idx_band == 0:
            initial_band_powers = initial_band_powers_list[idx_band]
        else:
            initial_band_powers = np.concatenate((initial_band_powers, initial_band_powers_list[idx_band]))

    # slicing points for all EE bands
    slicing_points_bands = {'EE': (0, initial_band_powers_EE.size),
                            # slicing points for all BB bands
                            'BB': (initial_band_powers_EE.size, initial_band_powers_EE.size + initial_band_powers_BB.size),
                            # slicing points for all EB bands
                            'EB': (initial_band_powers_EE.size + initial_band_powers_BB.size, initial_band_powers.size)
                           }

    for i, band in enumerate(ini.bands):
        initial_band_powers_band = initial_band_powers[slicing_points_bands[band][0]:slicing_points_bands[band][1]]
        fname = 'initial_guess_{:}_z{:}xz{:}.dat'.format(band, idx_zbin2 + 1, idx_zbin1 + 1)
        if not os.path.isfile(fname):
            savedata = np.column_stack(((bands_min[i] + bands_max[i]) / 2., bands_min[i], bands_max[i], initial_band_powers_band))
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
            all_bands_to_use_EE = bands_to_use_EE.tolist()
        if band == 'BB':
            bands_to_use_BB = np.ones_like(bands_BB_min)
            # don't use last BB band:
            bands_to_use_BB[-1] = 0
            # don't use second to last BB band:
            #bands_in_conv_BB[-2] = 0
            all_bands_to_use_BB = bands_to_use_BB.tolist()
        if band == 'EB':
            bands_to_use_EB = np.ones_like(bands_EB_min)
            # don't use last BB band:
            bands_to_use_EB[-1] = 0
            # don't use second to last BB band:
            #bands_in_conv_EB[-2] = 0
            all_bands_to_use_EB = bands_to_use_EB.tolist()

    # TODO: Expand this to possibly include also EB bands in convergence (but this is a very unlikely use-case as is already the inclusion of BB bands)
    if ini.include_BB_in_convergence:
        all_bands_to_use = all_bands_to_use_EE + all_bands_to_use_BB
    elif not ini.include_BB_in_convergence and 'BB' in ini.bands:
        all_bands_to_use = all_bands_to_use_EE + np.zeros_like(all_bands_to_use_BB).tolist()
    else:
        all_bands_to_use = all_bands_to_use_EE

    # this boolean contains now all indices of bands that should be included in convergence criterion for Newton Raphson!
    indices_bands_to_use = np.where(np.asarray(all_bands_to_use) == 1)[0]

    return initial_band_powers, indices_bands_to_use, slicing_points_bands

def get_signal_matrices_per_band(path_to_signal, path_to_signal_store, bands_min, bands_max, x_rad, y_rad):
    """
    get_signal_matrices_per_band(path_to_signal, path_to_signal_store, bands_min, bands_max, x_rad, y_rad)

    Function supplying all signal matrices for the requested band power intervals and band types. The matrices
    are all stored to disk in an h5-file and will be loaded as needed.

    Parameters
    ----------
    path_to_signal : string
                           Path to temporary storage place of the h5 file containing
                           all signal matrices.

    path_to_signal_store : string
                           Path to the final storage place of the h5 file containing
                           all signal matrices.

    bands_min : 1D numpy.ndarray
                           Array containing the lower limits of the band power multipole intervals.

    bands_max : int
                           Array containing the upper limits of the band power multipole intervals.

    x_rad : 1D numpy.ndarray
                           Array containing the x-coordinates (in radians) of all shear pixels.

    y_rad : 1D numpy.ndarray
                           Array containing the y-coordinates (in radians) of all shear pixels.

    Returns
    -------
    identifier_SM : string
                           Unique identifying name for the h5-file containing all signal matrices.
    """
    # if working with mock data the initial guess for the signal matrix can be shared (if all fields look the same) to save runtime!
    if ini.share_signal_matrices:
        identifier_SM = ini.shared_identifier_SM
    else:
        identifier_SM = ini.identifier_in

    # try to load signal matrices, else calculate them:
    for idx_band, band in enumerate(ini.bands):

        print('band_min:', bands_min[idx_band].min())
        print('band_max:', bands_max[idx_band].max())
        #fname = path_to_signal[i]+'signal_matrices_'+band+'_'+str(bands_min[i].min())+'_'+str(bands_max[i].max())+'_'+str(np.round(ini.side_length, decimals=2))
        #print(fname)
        #if not os.path.isdir(path_to_signal):
        #    os.makedirs(path_to_signal)
            #signal = []
        t0_signal_matrices = timer()
        fname = path_to_signal + 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '.h5'

        #print(fname)
        #exit()
        if not os.path.isfile(fname): # + '.npy'):
            cmd = 'w'
        else:
            cmd = 'a'

        # calculate (derivative?) signal matrices:
        for idx_band_power in xrange(bands_max[idx_band].size): #bands_min.size):

            fname_matrix = 'signal_matrix_idx_bp{:03d}'.format(idx_band_power)

            # check if matrices already exist:
            try:
                with h5py.File(fname, 'r') as hf:
                    # we need to create the list explicitly!
                    matrix_names = [key for key in hf.keys()]
                    
                if fname_matrix in matrix_names:
                    matrix_already_exists = True
                else:
                    matrix_already_exists = False

            except:
                matrix_already_exists = False

            if (cmd == 'w') or ((cmd == 'a') and not matrix_already_exists):
                # these are GLOBAL, together with "band"
                band_min = bands_min[idx_band][idx_band_power]
                band_max = bands_max[idx_band][idx_band_power]
                print('Calculating ' + band + ' signal matrix {:}'.format(idx_band_power + 1) + '/{:}'.format(bands_max[idx_band].size) + ' with {:.3}'.format(band_min) + ' <= ell <= {:.3}.'.format(band_max))
                print('Calculating moments of window function.')
                # this is done now in the initialization of SignalMatrix()
                t0_wf = timer()
                dSM = sm.SignalMatrix(band_min=band_min, band_max=band_max, band=band, sigma=ini.side_length, integrate=True, ncpus=ini.ncpus_SM)
                print('Done. Time: {:.2f}s.'.format(timer() - t0_wf))
                # decide if Fast implementation is used or slow...
                # intensive testing on CFHTLenS data showed now difference in the *end* results between using FAST SM or SLOW SM (the matrices are different though!)
                if ini.calculate_SM_fast:
                    print('Fast calculation.')
                    signal = np.asarray(dSM.getSignalMatrixFAST(x_rad, y_rad))
                else:
                    print('Slow calculation.')
                    signal = np.asarray(dSM.getSignalMatrix(x_rad, y_rad))
                print('Calculation of signal matrix took {:.2f}s.'.format(timer() - t0_wf))
                #fname = path_to_signal + 'signal_matrix_' + band + '_{:.3f}_idx_bp{:03d}_'.format(ini.side_length, idx_band_power) + identifier_SM
                #np.save(fname, signal)
                #fname = path_to_signal + 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '.h5'

                with h5py.File(fname, cmd) as hf:
                    #group = hf.create_group(band)
                    hf.create_dataset(fname_matrix, data=signal)
                # if it was 'w', set to 'a'!
                cmd = 'a'

        dt_signal_matrices = timer() - t0_signal_matrices
        print('Calculation of signal matrices took {:.2f}min.'.format(dt_signal_matrices / 60.))

        if ini.calculate_only_signal_matrices_for_derivatives:
            print('Done with calculation of signal matrices for derivatives.')
            if ini.root_signal != ini.root_signal_store:
                t0_store = timer()
                print('Copying signal matrix files now to: \n', path_to_signal_store)
                import shutil
                for idx_band, band in enumerate(ini.bands):
                    fname = 'signal_matrices_' + band + '_sigma_pix{:.3f}_'.format(ini.side_length) + identifier_SM + '.h5'
                    shutil.move(path_to_signal + fname, path_to_signal_store)
                #os.chdir(path_to_signal_store)
                #shutil.rm(path_to_signal + fname)
                print('Done in {:.2f}s.'.format(timer() - t0_store))
            exit()

    return identifier_SM


def get_signal_matrices_per_multipole(path_to_signal, path_to_signal_store, x_rad, y_rad):
    """
    get_signal_matrices_per_multipole(path_to_signal, path_to_signal_store, bands_min, bands_max, x_rad, y_rad)

    Function supplying all signal matrices for the requested multipoles and band types (used in the band window
    matrix calculation). The matrices are all stored to disk in one h5-archive and will be loaded as needed.

    Parameters
    ----------
    path_to_signal : string
                           Path to temporary storage place of the h5 file containing
                           all signal matrices.

    path_to_signal_store : string
                           Path to the final storage place of the h5 file containing
                           all signal matrices.

    bands_min : 1D numpy.ndarray
                           Array containing the lower limits of the band power multipole intervals.

    bands_max : int
                           Array containing the upper limits of the band power multipole intervals.

    x_rad : 1D numpy.ndarray
                           Array containing the x-coordinates (in radians) of all shear pixels.

    y_rad : 1D numpy.ndarray
                           Array containing the y-coordinates (in radians) of all shear pixels.

    Returns
    -------
    identifier_SM : string
                           Unique identifying name for the h5-file containing all signal matrices.
    """
    # new function that created log-spaced integers without duplicatons:
    ells = generate_log_spaced_multipoles(ini.ell_min, ini.ell_max + 1, ini.nell)
    fname = 'multipole_nodes_for_band_window_functions_nell{:}.dat'.format(ini.nell)
    np.savetxt(fname, ells)

    # using HDF5
    t0_signal_matrices_for_bwm = timer()

    # if working with mock data the initial guess for the signal matrix can be shared (if all fields look the same) to save runtime!
    if ini.share_signal_matrices:
        identifier_SM = ini.shared_identifier_SM
    else:
        identifier_SM = ini.identifier_in

    for idx_band, band in enumerate(ini.bands):

        fname = os.path.join(path_to_signal, 'signal_matrices_{:}_sigma_pix{:.3f}_{:}_BWM_nell{:}.h5'.format(band, ini.side_length, identifier_SM, ini.nell))

        if not os.path.isfile(fname):
            cmd = 'w'
        else:
            cmd = 'a'

        # calculate (derivative) signal matrices:
        for idx_ell in xrange(ini.nell):

            fname_matrix = 'signal_matrix_idx_ell{:03d}'.format(idx_ell)

            # check if matrices already exist:
            try:
                with h5py.File(fname, 'r') as hf:
                    # we need to create the list explicitly!
                    matrix_names = [key for key in hf.keys()]

                if fname_matrix in matrix_names:
                    matrix_already_exists = True
                else:
                    matrix_already_exists = False

            except:
                matrix_already_exists = False

            if (cmd == 'w') or ((cmd == 'a') and not matrix_already_exists):

                print('Calculating matrix of band window functions {:}/{:} with {:} <= ell <= {:}.'.format(idx_ell + 1, ini.nell, ini.ell_min, ini.ell_max))
                print('Multipole ell = {:}'.format(ells[idx_ell]))
                # Unfortunately this is done now at each initialization of SignalMatrix (due to restructuring because of pickle-issue with multiprocessing)
                print('Calculating moments of window function.')
                t0_wf = timer()
                dSM_ell = sm.SignalMatrix(band_min=ells[idx_ell], band_max=0., band=band, sigma=ini.side_length, integrate=False, ncpus=ini.ncpus_BWM)
                print('Done. Time: {:.2f}s.'.format(timer() - t0_wf))
                if ini.calculate_SM_fast:
                    print('Fast calculation.')
                    signal_ell = dSM_ell.getSignalMatrixFAST(x_rad, y_rad)
                else:
                    print('Slow calculation.')
                    signal_ell = dSM_ell.getSignalMatrix(x_rad, y_rad)

                with h5py.File(fname, cmd) as hf:
                #group = hf.create_group(band)
                    hf.create_dataset(fname_matrix, data=signal_ell)
                # if it was 'w', set to 'a'!
                cmd = 'a'

    print('Time for calculation of {:} signal matrices for band window matrix: {:.2f}min.'.format(len(ini.bands) * ini.nell, (timer() - t0_signal_matrices_for_bwm) / 60.))

    if ini.calculate_only_signal_matrices_for_band_window_matrix:
        print('Done with calculating signal matrices for band window matrix.')
        if ini.root_signal != ini.root_signal_store:
            t0_store = timer()
            print('Copying signal matrix files now to: \n', path_to_signal_store)
            import shutil
            for idx_band, band in enumerate(ini.bands):
                fname = 'signal_matrices_{:}_sigma_pix{:.3f}_{:}_BWM_nell{:}.h5'.format(band, ini.side_length, identifier_SM, ini.nell)
                shutil.move(path_to_signal + fname, path_to_signal_store)
            #os.chdir(path_to_signal_store)
            #shutil.rm(path_to_signal + fname)
            print('Done in {:.2f}s.'.format(timer() - t0_store))
        exit()

    return identifier_SM

def optimize_with_Newton_Raphson(initial_band_powers, indices_bands_to_use, slicing_points_bands, data_vector_a, data_vector_b, diag_noise, inv_N_eff_gal, dim_signals, sigma_int, idx_zbin1, idx_zbin2):
    """
    optimize_with_Newton_Raphson(initial_band_powers, indices_bands_to_use, data_vector_a, data_vector_b, diag_noise, inv_N_eff_gal, dim_signals, sigma_int, idx_zbin1, idx_zbin2)

    Main function of the algorithm which determines the requested band power types
    from the shear catalogs via a Newton-Raphson iteration to the maximum likelihood
    solution based on an initial guess.

    Parameters
    ----------
    initial_band_powers : 1D numpy.ndarray
                Array containing the initial guesses for all requested band powers

    indices_bands_to_use : 1D numpy.ndarray
                Array containing the indices (with respect to 'initial_band_powers') for the bands
                that will be checked for convergence.

    slicing_points_bands : dict
                Dictionary containing the sets of indices at which to split 'initial_band_powers' into
                the single EE, BB, EB components.

    data_vector_a : 1D numpy.ndarray
                The data vector of z-bin <m>.

    data_vector_b : 1D numpy.ndarray
                The data vector of z-bin <n>.

    diag_noise : 1D numpy.ndarray
                Array containing the diagonal elements of the noise matrix.

    inv_N_eff_gal : 1D numpy.ndarray
                Array containing the inverse of the diagonal elements and having
                devided out \sigma_{int}^2.

    dim_signals : int
                Dimension n for all nxn signal matrices per band.

    sigma_int : 1D numpy.ndarray
                Contains the internal shear dispersion values per z-bin

    idx_zbin1 : int
                Index of the first z-bin to be used in the analysis.

    idx_zbin2 : int
                Index of the second z-bin to be used in the analysis.

    Returns
    -------
    niterations : int
                The number of iterations needed until convergence.
    """
    t0_Newton_Raphson = timer()
    niterations = 0

    if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
        # for noise_marginalization we start with p_noise(z_i) = sigma_int**2(z_i)
        p_noise0 = np.array([sigma_int[idx_zbin1]**2])
        p_noise = p_noise0
        collect_p_noise = []
        collect_delta_p_noise = []
        collect_relative_difference_p_noise = []
        collect_p_noise.append(p_noise0)

    collect_band_powers = []
    collect_delta_band_powers = []
    collect_relative_difference_band_powers = []
    band_powers = initial_band_powers
    collect_band_powers.append(initial_band_powers)
    # switch to local variable (because we need to modify it!)
    lambda0 = ini.lambda0
    converged = False
    while not converged:
        t0_loop = timer()
        print('Lambda = {:}'.format(lambda0))
        print('Iteration', niterations + 1)
        print('Estimates for band powers: \n', band_powers, band_powers.size)
        # now we check that EE+BB+EB > 0., if not we set the corresponding EE < 0, BB < 0, or EB < 0 to a small positive number!
        # this more complicated approach created numerical instabilities (i.e. 'oscillations')
        #band_powers = assure_total_power_greater_zero(band_powers, slicing_points_bands, nzbins=nzbins, small_positive_number=ini.resetting_value)
        #'''
        if ini.reset_negative_band_powers:
            idx_of_bp_smaller_zero = np.where(band_powers < 0.)
            print('Band powers < 0: \n', band_powers[idx_of_bp_smaller_zero])
            #band_powers[idx_of_bp_smaller_zero] = float(ini.resetting_value)
            #a = float(ini.resetting_value)
            #b = 10. * float(ini.resetting_value)
            #print((b - a) * np.random.random(len(idx_of_bp_smaller_zero)) + a)
            #print(idx_of_bp_smaller_zero.size)
            #band_powers[idx_of_bp_smaller_zero] = (b - a) * np.random.random(band_powers[idx_of_bp_smaller_zero].size) + a
            # reset negative band powers to their initial value:
            band_powers[idx_of_bp_smaller_zero] = initial_band_powers[idx_of_bp_smaller_zero]
            print('Reset band powers: \n', band_powers)
        #'''
        full_signal = np.zeros((diag_noise.size, diag_noise.size))
        t0_mult = timer()
        for idx_bp in xrange(band_powers.size):
            # multiply each band power with its corresponding "derivative matrix"
            signal_deriv = np.load('cache/' + 'deriv_matrix_idx{:03d}.npy'.format(idx_bp)) #[()].toarray()
            # this is scalar * matrix
            product = band_powers[idx_bp] * signal_deriv
            full_signal += product

        full_covariance = np.asarray(full_signal)

        if ini.marginalize_over_noise:
            # noise matrix is assumed to be diagonal!
            if idx_zbin1 == idx_zbin2:
                print('p_noise(z{:}) = {:.3f}, sqrt(p_noise(z{:})) = {:.3f}'.format(idx_zbin1 + 1, p_noise[0], idx_zbin1 + 1, np.sqrt(p_noise[0])))
                diag_noise = np.concatenate((p_noise * inv_N_eff_gal, p_noise * inv_N_eff_gal))
            else:
                diag_noise = np.zeros(inv_N_eff_gal.size * 2)

        idx_diag = range(diag_noise.size)
        # since noise matrix is assumed to be diagonal, only add the diagonals:
        full_covariance[idx_diag, idx_diag] += diag_noise

        '''
        if ini.reset_negative_band_powers:
            diag_full_cov = np.copy(np.diag(full_covariance))
            idx_of_diag_elems_smaller_zero = np.where(diag_full_cov < 0.)
            print('Diagonal elements < 0: \n', diag_full_cov[idx_of_diag_elems_smaller_zero])
            a = float(ini.resetting_value)
            b = 10. * float(ini.resetting_value)
            diag_full_cov[idx_of_diag_elems_smaller_zero] = (b - a) * np.random.random(diag_full_cov[idx_of_diag_elems_smaller_zero].size) + a
            print('Reset diagonal elements: \n', diag_full_cov)
            np.fill_diagonal(full_covariance, diag_full_cov)
            print('Diagonal of covariance matrix: \n', np.diag(full_covariance))
        '''
        dt_mult = timer() - t0_mult
        print('Time for constructing full covariance: {:.2f}s.'.format(dt_mult))
        #logdata['dim_covariance'] = full_covariance.shape

        print('Dimension of full covariance matrix = {:}x{:}.'.format(full_covariance.shape[0], full_covariance.shape[1]))
        t0_inversion = timer()
        # not sure, what .I returns (since method .getI() seems to return pseudo-inverse, if inverse fails...)
        #inv_full_covariance = full_covariance.I
        # Found no difference in using regularization factor or not.
        #mean_diag_cov = np.diag(full_covariance).mean()
        #reg_factor = 2./mean_diag_cov
        #print('Regularization factor Covariance: {:.4e}'.format(reg_factor))
        #inv_full_covariance = reg_factor*np.linalg.inv(reg_factor*full_covariance)
        inv_full_covariance = inv(full_covariance)
        # Write to disk speeds up multiprocessing:
        np.save('cache/' + 'inv_covariance', inv_full_covariance)
        np.save('cache/' + 'covariance', full_covariance)

        if not ini.do_not_invert_that_matrix:
            print('Time for matrix inversion: {:.4f}s.'.format(timer() - t0_inversion))

        if ini.numerical_test:
            test_numerics(full_covariance, inv_full_covariance, matrix_is_fisher=False)

        t0_full_fisher = timer()
        print(r'First step for Fisher matrix: C^{-1}_{ij} (C,\alpha)_{jk}')
        success = compute_inv_covariance_times_deriv_matrices(dim=dim_signals, ncpus=ini.max_ncpus)
        if success:
            print('Done with first step.')

        if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
            dim_fisher = band_powers.size + 1
        else:
            dim_fisher = band_powers.size

        fisher, dtf = get_fisher_matrix(dim=dim_fisher, ncpus=ini.max_ncpus)
        print('Time for second step of calculation of Fisher matrix: {:.2f}s.'.format(dtf))
        print('Total time for calculation of Fisher matrix: {:.2f}s.'.format(timer() - t0_full_fisher))

        t0_inversion = timer()
        # not sure, what .I returns (since method .getI() seems to return pseudo-inverse, if inverse fails...)
        #inv_fisher = fisher.I
        # Found no difference in using regularization factor or not.
        #mean_diag_fisher = np.diag(Fisher).mean()
        #reg_factor = 2./mean_diag_fisher
        #print('Regularization factor Fisher: {:.4e}'.format(reg_factor))
        #inv_fisher = reg_factor*la.inv(reg_factor*fisher)
        if not ini.do_not_invert_that_matrix:
            inv_fisher = inv(fisher)
            print('Time for matrix inversion: {:.4f}s.'.format(timer() - t0_inversion))

        if ini.numerical_test:
            test_numerics(fisher, inv_fisher, matrix_is_fisher=True)
        print('Started calculation of new band power estimates.')

        if ini.do_not_invert_that_matrix:
            delta_parameters, dt_est = estimate_band_powers(data_vector_a, data_vector_b, full_covariance, fisher, convergence_parameter=lambda0, ncpus=ini.max_ncpus)
        else:
            delta_parameters, dt_est = estimate_band_powers(data_vector_a, data_vector_b, inv_full_covariance, inv_fisher, convergence_parameter=lambda0, ncpus=ini.max_ncpus)

        if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
            delta_band_powers = delta_parameters[:-1]
            delta_p_noise = delta_parameters[-1:]
            p_noise = p_noise + delta_p_noise
            #print(p_noise)
        else:
            delta_band_powers = delta_parameters
            delta_p_noise = np.array([0.])
            p_noise = np.array([0.])

        print('Time for calculation of new band power estimates: {:2f}s.'.format(dt_est))
        band_powers = band_powers + delta_band_powers
        # TODO: figure out if p_noise should be added to convergence criteria...
        if niterations > 0:
            if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
                difference_delta_p_noise = np.abs(delta_p_noise + collect_delta_p_noise[niterations - 1]) / np.abs(delta_p_noise)
                #print(collect_delta_band_powers[n-1])
                print('Relative cancellation in delta p_noise: \n', difference_delta_p_noise)
                difference_p_noise = np.abs(p_noise - collect_p_noise[niterations - 1]) / np.abs(p_noise)
                print('Relative difference in p_noise: \n', difference_p_noise)
                collect_relative_difference_p_noise.append(difference_p_noise)

            difference_delta_band_powers = np.abs(delta_band_powers + collect_delta_band_powers[niterations - 1]) / np.abs(delta_band_powers)
            #print(collect_delta_band_powers[n-1])
            print('Relative cancellation in delta band powers: \n', difference_delta_band_powers)
            difference_band_powers = np.abs(band_powers - collect_band_powers[niterations - 1]) / np.abs(band_powers)
            print('Relative difference in band powers: \n', difference_band_powers)
            collect_relative_difference_band_powers.append(difference_band_powers)
            if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
                mask_bp = (difference_delta_band_powers <= 0.2)
                mask_noise = (difference_delta_p_noise <= 0.2)
                if mask_bp.any() or mask_noise.any():
                    lambda0 /= 2.
            else:
                mask = (difference_delta_band_powers <= 0.2) #& (difference_delta_band_powers >= 0.18)
                if mask.any():
                    lambda0 /= 2.

            mask = (difference_band_powers[indices_bands_to_use] <= ini.convergence_criterion)
            print('Relative difference in bands requested for convergence: \n', difference_band_powers[indices_bands_to_use])
            if mask.all():
                converged = True
                print('Convergence within {:.2f}% achieved.'.format(ini.convergence_criterion) * 100.)

        #band_powers[np.where(band_powers < 0.)] = 1e-7
        print('New estimates for band powers: \n', band_powers)
        if lambda0 < 0.01:
            lambda0 = 0.01

        collect_delta_band_powers.append(delta_band_powers)
        # copy is needed due to resetting of negative band powers!
        collect_band_powers.append(np.copy(band_powers))
        if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
            collect_delta_p_noise.append(delta_p_noise)
            # copy is needed due to resetting of negative band powers!
            collect_p_noise.append(np.copy(p_noise))

        # if code has not converged after max_iterations, we stop it nevertheless.
        if niterations == ini.max_iterations - 1:
            converged = True

        print_ascii_art('Time spent in one single Newton-Raphson iteration: {:.2f}s.'.format(timer() - t0_loop))
        # increase counter for iterations
        niterations += 1

    dt_Newton_Raphson = timer() - t0_Newton_Raphson
    print('Time for Newton-Raphson optimization with {:} iterations for {:} band(s): {:.2f} minutes.'.format(niterations, len(ini.bands), dt_Newton_Raphson / 60.))
    print('Initial estimate of band powers: \n', collect_band_powers[0])
    print('Final estimate of band powers: \n', collect_band_powers[-1])

    # save results and important matrices:
    inv_fisher = inv(fisher)
    if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
        print('Final estimate of p_noise(z{0:}) = {1:.3f}, sqrt(p_noise(z{0:})) = {2:.3f}'.format(idx_zbin + 1, p_noise[0], np.sqrt(p_noise[0])))
        print('(p_noise(z_mu) - p_noise0(z_mu)) / p_noise0(z_mu): \n', (p_noise - p_noise0) / p_noise0)
        correlation_matrix, dt_correlation_matrix = get_correlation_matrix(inv_fisher)
        fname = 'correlation_matrix_z{:}xz{:}_with_noise.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, correlation_matrix)
        fisher_without_noise = fisher[:-1, :-1]
        inv_fisher_without_noise = inv(fisher_without_noise)
        correlation_matrix, dt_correlation_matrix= get_correlation_matrix(inv_fisher_without_noise)
        fname = 'correlation_matrix_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, correlation_matrix)
    else:
        correlation_matrix, dt_correlation_matrix = get_correlation_matrix(inv_fisher)
        fname = 'correlation_matrix_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, correlation_matrix)

    collect_band_powers = np.asarray(collect_band_powers)
    collect_delta_band_powers = np.asarray(collect_delta_band_powers)
    collect_relative_difference_band_powers = np.asarray(collect_relative_difference_band_powers)

    for idx_band, band in enumerate(ini.bands):
        #idx_jump = slicing_offset[idx_band]
        save_collect_band_powers = collect_band_powers[:,slicing_points_bands[band][0]:slicing_points_bands[band][1]]
        save_collect_delta_band_powers = collect_delta_band_powers[:, slicing_points_bands[band][0]:slicing_points_bands[band][1]]
        save_collect_relative_difference_band_powers = collect_relative_difference_band_powers[:, slicing_points_bands[band][0]:slicing_points_bands[band][1]]

        # flip indices in filensames for standard nomenclature (e.g. 1x4 instead of 4x1):
        fname = 'all_estimates_band_powers_{:}_z{:}xz{:}.dat'.format(band, idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, save_collect_band_powers) #[:, idx_z:idx_jump])
        fname = 'band_powers_{:}_z{:}xz{:}.dat'.format(band, idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, save_collect_band_powers[-1, :]) #idx_z:idx_jump])
        fname = 'all_delta_band_powers_{:}_z{:}xz{:}.dat'.format(band, idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, save_collect_delta_band_powers) #[:, idx_z:idx_jump])
        fname = 'difference_band_powers_{:}_z{:}xz{:}.dat'.format(band, idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, save_collect_relative_difference_band_powers) #[:, idx_z:idx_jump])

    # TODO: get this to work again eventually
    if ini.marginalize_over_noise and idx_zbin1 == idx_zbin2:
        fname = 'all_estimates_p_noise_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, np.asarray(collect_p_noise))
        fname = 'all_delta_p_noise_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, np.asarray(collect_delta_p_noise))
        fname = 'difference_p_noise_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, np.asarray(collect_relative_difference_p_noise))
        fname = 'p_noise_z{:}.dat'.format(idx_zbin1 + 1)
        np.savetxt(fname, p_noise)
        fname = 'last_Fisher_matrix_z{:}xz{:}_with_noise.dat.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
        # that is why we multiply it with 0.5 before saving!
        np.savetxt(fname, 0.5 * fisher)
        fname = 'last_Fisher_matrix_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
        # that is why we multiply it with 0.5 before saving!
        np.savetxt(fname, 0.5 * fisher_without_noise)
        fname = 'last_larger_Fisher_matrix_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
        # that is why we multiply it with 0.5 before saving!
        np.savetxt(fname, fisher_without_noise)
        # for the BWM calculation we don't want to keep the noise-parameters in the Fisher matrices!
        '''
        if ini.do_not_invert_that_matrix:
            parameter_covariance = fisher_without_noise
        else:
            parameter_covariance = inv_fisher_without_noise
        '''
    else:
        fname = 'last_Fisher_matrix_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
        # that is why we multiply it with 0.5 before saving!
        np.savetxt(fname, 0.5 * fisher)
        fname = 'last_larger_Fisher_matrix_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
        np.savetxt(fname, fisher)
        '''
        if ini.do_not_invert_that_matrix:
            parameter_covariance = fisher
        else:
            parameter_covariance = inv_fisher
        '''
    # this matrix can become quite large, so we use HDF5 just in case:
    fname = 'last_covariance_matrix_z{:}xz{:}.h5'.format(idx_zbin2 + 1, idx_zbin1 + 1)
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('covariance', data=np.asarray(full_covariance))

    return niterations

def get_derivative_matrices_per_band(path_to_signal, bands_max, identifier_SM, inv_N_eff_gal, idx_zbin1, idx_zbin2):
    """
    get_derivative_matrices_per_band(path_to_signal, bands_max, identifier_SM, inv_N_eff_gal, idx_zbin1, idx_zbin2)

    Function supplying all signal matrices for the requested multipoles and band types (used in the band window
    matrix calculation). The matrices are all stored to disk in one h5-archive and will be loaded as needed.

    Parameters
    ----------
    path_to_signal : string
                Path to temporary storage place of the h5 file containing all signal matrices.

    bands_max : int
                Array containing the upper limits of the band power multipole intervals.

    inv_N_eff_gal : 1D numpy.ndarray
                Array containing the inverse of the diagonal elements and having
                devided out \sigma_{int}^2.

    idx_zbin1 : int
                Index of the first z-bin to be used in the analysis.

    idx_zbin2 : int
                Index of the second z-bin to be used in the analysis.

    Returns
    -------
    dim_signals : int
                Dimension n for all nxn signal matrices per band.
    """
    idx_signals = 0
    for idx_band, band in enumerate(ini.bands):

        fname_SM = os.path.join(path_to_signal, 'signal_matrices_{:}_sigma_pix{:.3f}_{:}.h5'.format(band, ini.side_length, identifier_SM))

        for idx_band_power in xrange(bands_max[idx_band].size):

            # check if derivative matrices exist already (e.g. for BWM run):
            fname_deriv = 'cache/deriv_matrix_idx{:03d}.npy'.format(idx_signals)

            if os.path.isfile(fname_deriv):
                # this is only needed once for "dims_single_deriv"
                if idx_signals == 0:
                    deriv_matrix = np.load(fname_deriv)
                idx_signals += 1
            # if not, calculate them:
            else:

                with h5py.File(fname_SM, 'r') as hf:
                    #group = hf.get(band)
                    signal_matrix = np.array(hf.get('signal_matrix_idx_bp{:03d}'.format(idx_band_power)))
                zero_matrix = np.zeros_like(signal_matrix)

                if idx_zbin1 == idx_zbin2:
                    deriv_matrix = signal_matrix
                else:
                    deriv_matrix = zero_matrix
                # save to disk:
                np.save(os.path.join('cache/', 'deriv_matrix_idx{:03d}'.format(idx_signals)), deriv_matrix)
                idx_signals += 1

    dim_signals = idx_signals

    # currently we only assume a diagonal noise matrix of the form:
    # C_noise_(ij, ab, \mu\nu) = p_noise(z_\mu) / N_i(\mu) d_ij d_ab d_\mu\nu
    if ini.marginalize_over_noise:
        if idx_zbin1 == idx_zbin2:
            noise_matrix = np.eye(inv_N_eff_gal.size * 2)
            np.fill_diagonal(noise_matrix, np.concatenate((inv_N_eff_gal, inv_N_eff_gal)))
            deriv_noise_matrix = noise_matrix
        else:
            deriv_noise_matrix = np.zeros_like(deriv_matrix)
        np.save('cache/' + 'deriv_matrix_idx{:03d}'.format(dim_signals), deriv_noise_matrix)
        dim_signals += 1

    #print('Dim(signals):', dim_signals)

    return dim_signals

def get_derivative_matrices_per_multipole(path_to_signal, identifier_SM, idx_zbin1, idx_zbin2):
    """
    get_derivative_matrices_per_multipole(path_to_signal, identifier_SM, idx_zbin1, idx_zbin2)

    Function supplying all signal matrices for the requested multipoles and band types (used in the band window
    matrix calculation). The matrices are all stored to disk in one h5-archive and will be loaded as needed.

    Parameters
    ----------
    path_to_signal : string
                Path to temporary storage place of the h5 file containing all signal matrices.

    identifier_SM : string
                Unique identifying name for the h5-file containing all signal matrices.

    idx_zbin1 : int
                Index of the first z-bin to be used in the analysis.

    idx_zbin2 : int
                Index of the second z-bin to be used in the analysis.

    Returns
    -------
    idx_linear : int
                Linear index labelling each derivative matrix with respect to multipoles.
    """
    # new approach with HDF5:
    idx_linear = 0
    for idx_band, band in enumerate(ini.bands):

        fname_SM = os.path.join(path_to_signal, 'signal_matrices_{:}_sigma_pix{:.3f}_{:}_BWM_nell{:}.h5'.format(band, ini.side_length, identifier_SM, ini.nell))

        for idx_ell in xrange(ini.nell):

            # check if derivative matrices exist already (e.g. for BWM run):
            fname_deriv = 'cache/deriv_matrix_ell_idx_lin{:03d}.npy'.format(idx_linear)

            if os.path.isfile(fname_deriv):
                # this is only needed once for "dims_single_deriv"
                if idx_linear == 0:
                    deriv_matrix = np.load(fname_deriv)
                idx_linear += 1
            # if not, calculate them:
            else:

                with h5py.File(fname_SM, 'r') as hf:
                    #group = hf.get(band)
                    signal_matrix = np.array(hf.get('signal_matrix_idx_ell{:03d}'.format(idx_ell)))
                zero_matrix = np.zeros_like(signal_matrix)

                if idx_zbin1 == idx_zbin2:
                    deriv_matrix = signal_matrix
                else:
                    deriv_matrix = zero_matrix
                # save to disk:
                np.save('cache/deriv_matrix_ell_idx_lin{:03d}'.format(idx_linear), deriv_matrix)
                idx_linear += 1

    #print('Linear idx: {:}'.format(idx_linear))

    return idx_linear

def get_input_parameters(filename):
    """
    get_input_parameters(filename)

    Function returning an 'initial guess' for the {'EE', 'BB', 'EB'} band
    powers.

    Helper function that reads in required input parameters from file 'filename'.
    We always assume that this file is using Python notation for variables and
    follows the notation and variable names in 'default.ini'!

    Parameters
    ----------
    filename : string
                The name (inlcuding the path) for the '.ini'-file to be read in.

    Returns
    -------

    Nothing, but makes all parameters listed in '.ini'-file and declares them
    global variables in the namespace 'ini'.
    """

    global ini
    
    # for Python2.X:
    if sys.version_info[0] < 3:
        import imp
        
        f = open(filename)
        ini = imp.load_source('data', '', f)
        f.close()
    
    # for Python3.X:
    else:
        import importlib.machinery as imp
    
        ini = imp.SourceFileLoader(filename, filename).load_module()
        
    # here is the place to include safety checks!
    if ini.max_iterations < 2:
        ini.max_iterations = 2

    return

# copy & paste from stackoverflow
def generate_log_spaced_multipoles(start, end, n):
    """
    generate_log_spaced_multipoles(start, end, n)

    Function creating n log-spaced integers between (& including) start, end
    without repetitions. In the beginning of the sequence this might
    yield rather a lin-spaced sequence.

    Parameters
    ----------
    start : float
            The start of the requested log-spaced multipole interval (inclusive).

    end : float
            The end of the requested log-spaced multipole interval (inclusive).

    n : int
            The amount of requested log-spaced multipoles.

    Returns
    -------

    log_spaced_multipoles : 1D numpy.ndarray
            The array containing the requested log-spaced multipole interval.
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

    log_spaced_multipoles = np.asarray(result).astype('int')

    return log_spaced_multipoles

def test_numerics(matrix, inv_matrix, matrix_is_fisher=True):
    """
    test_numerics(matrix, inv_matrix, matrix_is_fisher=True)

    Function to perform some basic numerical tests on symmetry properties and precision of inversion.

    Parameters
    ----------
    matrix : 2D numpy.ndarray

    inv_matrix : 2D numpy.ndarray

    matrix_is_fisher : bool, optional

    Returns
    -------
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
    print('Numerical value of det(' + type_of_matrix + ') = {:.4e}'.format(det_matrix))
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    logsum = 0
    if matrix_is_fisher:
        print('Are there negative eigenvalues in F?')
    for i in xrange(len(eig_vals)):
        if matrix_is_fisher:
            print('Eigenvalue {:} = {:.2e}'.format(i + 1, eig_vals[i]))
        logsum += np.log10(eig_vals[i])
    print('log10(det(' + type_of_matrix + ')): ', logsum)
    left_product = matrix.dot(inv_matrix)
    right_product = inv_matrix.dot(matrix)
    print('Is ' + type_of_matrix + ' = ' + type_of_matrix + '^T? ', np.allclose(matrix, trans_matrix))
    print('Is ' + type_of_matrix + '^T = ' + type_of_matrix + '? ', np.allclose(trans_matrix, matrix))
    print('Is ' + type_of_matrix + '^{-1} = (' + type_of_matrix + '^{-1})^T? ', np.allclose(inv_matrix, trans_inv_matrix))
    print('Is (' + type_of_matrix + '^{-1})^T = ' + type_of_matrix + '^{-1}? ', np.allclose(trans_inv_matrix, inv_matrix))
    print('Is ' + type_of_matrix + '*' + type_of_matrix + '^{-1} = 1? ', np.allclose(left_product, identity_matrix))
    print('Is ' + type_of_matrix + '^{-1}*' + type_of_matrix + ' = 1? ', np.allclose(right_product, identity_matrix))
    print('Is 1 = ' + type_of_matrix + '*' + type_of_matrix + '^{-1}? ', np.allclose(identity_matrix, left_product))
    print('Is 1 = ' + type_of_matrix + '^{-1}*' + type_of_matrix + '? ', np.allclose(identity_matrix, right_product))

    if matrix_is_fisher:
        print('Are there negative values on diagonal of F^{-1}? \n', np.diag(inv_matrix))
        print('Are there negative values on diagonal of F? \n', np.diag(matrix))
        print('WV CHECK: there are negatives if we include off-diagonal block-diagonal noise contributions.')

    dt = timer() - t0
    print('Time for numerical tests: {:.2f}s.'.format(dt))

    return


if __name__ == '__main__':

    # fix random seed so that code converges to same solution for repeated runs!
    np.random.seed(42)

    # TODO: Make code more modular:
    # 1) init-function (setting up folders, initial guess etc.)
    t0_prog = timer()
    # read filename for input parameter file (include some safety checks later on...):
    try:
        fname_input = str(sys.argv[1])
    except:
        print('No filename for input parameters supplied. \n Loading parameters from \"default.ini\" now.')
        fname_input = 'default.ini'

    # read in input parameters from specified file:
    get_input_parameters(fname_input)
    # create dictionary of column names for catalog files
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

    fname_logfile = ini.identifier_in + '.log'
    path_to_dir_of_run, fname_of_script = os.path.split(os.path.abspath(__file__))
    print('Script', fname_of_script, ' is being run in', path_to_dir_of_run)

    z_min = min(ini.z_min)
    z_max = max(ini.z_max)

    redshift_bins = []
    for idx_zbin in xrange(len(ini.z_min)):
        redshift_bin = '{:.2f}z{:.2f}'.format(ini.z_min[idx_zbin], ini.z_max[idx_zbin])
        redshift_bins.append(redshift_bin)
    # number of z-bins
    nzbins = len(redshift_bins)
    # number of *unique* correlations between z-bins
    nzcorrs = nzbins * (nzbins + 1) // 2

    # two possible options:
    # a) a list sigma_int_e1 and a list sigma_int_e2 are passed --> symmetrize
    # b) a list (of already symmetrized) sigma_int is passed --> just change to array
    # list is assumed (but not checked for) to have length of z-bins!
    # needs to be a nested try-except-structure!

    if ini.estimate_sigma_int_avg:
        sigma_int = np.ones(nzbins) * -1.
        print('Intrinsic ellipticity dispersion, sigma_int, will be estimated from catalogue (i.e. averaged over full catalogue per z-bin).')
    elif ini.estimate_sigma_int_pix:
        sigma_int = np.ones(nzbins) * -2.
        print('Intrinsic ellipticity dispersion, sigma_int, will be estimated per shear pixel.')
    else:
        try:
            sigma_int = np.sqrt((np.asarray(ini.sigma_int_e1)**2 + np.asarray(ini.sigma_int_e2)**2) / 2.)
            print('Symmetrizing sigma_int_e1 and sigma_int_e2 to sigma_int.')
        except:

            try:
                sigma_int = np.asarray(ini.sigma_int)
                print('Using supplied sigma_int.')
            except:
                print('Either sigma_int or sigma_int_e1 and sigma_int_e2 must be supplied in input-file!')
                exit()

    # this is just for naming purposes:
    if ini.estimate_sigma_int_avg:
        sigma_int_for_naming = 'sigma_int_est_avg'
    elif ini.estimate_sigma_int_pix:
        sigma_int_for_naming = 'sigma_int_est_pix'
    else:
        sigma_int_for_naming = 'sigma_int{:.2f}'.format(sigma_int[0])

    path_output = ini.root_run + '/{:}/pix{:.3f}/{:.2f}z{:.2f}/{:}zbins/'.format(sigma_int_for_naming, ini.side_length, z_min, z_max, nzbins) + '/' + ini.identifier_in + '/'
    path_store = ini.root_store + '/{:}/pix{:.3f}/{:.2f}z{:.2f}/{:}zbins/'.format(sigma_int_for_naming, ini.side_length, z_min, z_max, nzbins) + '/' + ini.identifier_in + '/'
    # set subfolder for storing of matrices (does not depend on sigma_int, but depends on nzbins due to different masking of field!):
    path_to_signal = ini.root_signal + '/{:.2f}z{:.2f}/{:}zbins/'.format(z_min, z_max, nzbins)
    path_to_signal_store = ini.root_signal_store + '/{:.2f}z{:.2f}/{:}zbins/'.format(z_min, z_max, nzbins)

    if not os.path.isdir(path_to_signal):
        os.makedirs(path_to_signal)

    if path_to_signal != path_to_signal_store:
        if not os.path.isdir(path_to_signal_store):
            os.makedirs(path_to_signal_store)

    # check if output exists already; if not, create the directory
    if not os.path.isdir(path_output):
        os.makedirs(path_output)
        os.makedirs(path_output + 'cache/')
        # not needed for now:
        #os.makedirs(path_output + 'plots/')
        os.makedirs(path_output + 'control_outputs/')

    # for consistency with BWM calculation:
    if not os.path.isdir(path_output + 'cache/'):
        os.makedirs(path_output + 'cache/')

    if not os.path.isdir(path_output + 'bwm/'):
        os.makedirs(path_output + 'bwm/')

    # switch to new path
    os.chdir(path_output)

    # set paths; root path must point down to split in redshift:
    paths_to_data = []
    for idx_zbin, zbin in enumerate(redshift_bins):
        paths_to_data.append(ini.root_path_data + '/' + zbin + '/')
        try:
            # for legacy reasons...
            filename = ini.identifier_in + '_clean.cat'
        except:
            filename = ini.identifier_in + '.fits'

        print('Loaded catalogs from:')
        print(paths_to_data[idx_zbin] + filename)

    print('Field ID, sigma_int (per redshift bin):', ini.identifier_in, sigma_int)
    print('Path for saving:', path_output)
    print('Mode is:', mode)

    t0_data = timer()
    # looping over 'nzbins' is done internally in function:
    # here we have to pass the sigma_int !!!
    data, shot_noise, field_properties, x_rad_all, y_rad_all = dr.get_data(paths_to_data, filename, names_zbins=redshift_bins, identifier=ini.identifier_in, sigma_int=sigma_int, pixel_scale=ini.side_length,
                                                                      nzbins=nzbins, mode=ini.mode, min_num_elements_pix=ini.minimal_number_elements_per_pixel, column_names=column_names)
    print('Time for data reduction: {:.4f}s'.format(timer() - t0_data))

    # sanity check for that coordinates are the same...:
    # What if not?!
    # This problem is solved now!
    '''
    print('Is x1=x2?', np.allclose(x_rad_all[0], x_rad_all[1]))
    print('Is y1=y2?', np.allclose(y_rad_all[0], y_rad_all[1]))

    print(x_rad_all[0]-x_rad_all[1])
    print(y_rad_all[0]-y_rad_all[1])
    '''

    # all x_rad, y_rad should be the same per zbin (that's why minimal field and maximal mask), hence:
    x_rad = x_rad_all[0]
    y_rad = y_rad_all[0]

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

    print('Side length of 1 pixel = {:.3f} deg'.format(ini.side_length))
    print('ell_field_max = {:.2f} < ell < ell_pix = {:.2f}'.format(ell_field_max, ell_pix))
    print('ell_field_x = {:.2f} < ell < ell_pix = {:.2f}'.format(ell_field_x, ell_pix))
    print('ell_field_y = {:.2f} < ell < ell_pix = {:.2f}'.format(ell_field_y, ell_pix))

    for idx_zbin in xrange(nzbins):
        print('Effective number density n_eff in patch in zbin{:} (underestimated) = {:.2f}'.format(idx_zbin + 1, field_properties[idx_zbin]['n_eff']))
        print('Effective number of sources N_eff in patch in zbin{:} = {:.2f}'.format(idx_zbin + 1, field_properties[idx_zbin]['N_eff']))

    # ATTENTION: Stop here for data reduction only!
    # TODO: make data reduction optional and code compatible with loading binned catalogs...
    #exit()

    # basic setup of band power dimensions:
    bands_EE_min = np.array(ini.bands_EE_min)
    bands_EE_max = np.array(ini.bands_EE_max)
    bands_BB_min = bands_EE_min[1:]
    bands_BB_max = bands_EE_max[1:]
    bands_EB_min = bands_EE_min[1:]
    bands_EB_max = bands_EE_max[1:]

    bands_min = [bands_EE_min, bands_BB_min, bands_EB_min]
    bands_max = [bands_EE_max, bands_BB_max, bands_EB_max]

    # create list of signal matrices:
    identifier_SM = get_signal_matrices_per_band(path_to_signal, path_to_signal_store, bands_min, bands_max, x_rad, y_rad)

    # perform Newton-Raphson optimization of band powers
    if ini.optimize:

        t0_optimize = timer()

        for idx_zbin1 in xrange(nzbins):
            for idx_zbin2 in xrange(idx_zbin1 + 1):

                t0 = timer()
                print_ascii_art('Optimizing band powers for z{:}xz{:}.'.format(idx_zbin2 + 1, idx_zbin1 + 1))

                if idx_zbin1 == idx_zbin2 and ini.subtract_noise_power:
                    # concatenation required because data has two ellipticity components: e1 & e2!
                    diag_noise = np.concatenate((shot_noise[idx_zbin1], shot_noise[idx_zbin2]))
                    inv_N_eff_gal = shot_noise[idx_zbin1] / sigma_int[idx_zbin2]**2
                else:
                    # no noise on diagonal for redshift cross-correlations or no noise at all:
                    diag_noise = np.zeros(shot_noise[idx_zbin1].size * 2)

                print('Mean of noise matrix = {:.4e}'.format(diag_noise.mean()))

                initial_band_powers, indices_bands_to_use, slicing_points_bands = get_initial_band_powers(bands_min, bands_max, idx_zbin1, idx_zbin2)

                data_vector_a = data[idx_zbin1]
                data_vector_b = data[idx_zbin2]

                dim_signals = get_derivative_matrices_per_band(path_to_signal, bands_max, identifier_SM, inv_N_eff_gal, idx_zbin1, idx_zbin2)

                # start Newton-Raphson optimization of band powers:
                niterations = optimize_with_Newton_Raphson(initial_band_powers, indices_bands_to_use, slicing_points_bands, data_vector_a, data_vector_b, diag_noise, inv_N_eff_gal, dim_signals, sigma_int, idx_zbin1, idx_zbin2)

                dt = timer() - t0
                print_ascii_art('Time spent for {:} iterations for z{:}xz{:}: {:.2f}s.'.format(niterations, idx_zbin2 + 1, idx_zbin1 + 1, dt))

        dt = (timer() - t0_optimize) / 60.
        print_ascii_art('Total time spent for extracting {:} bands in {:} correlations: {:.2f}min.'.format(len(ini.bands), nzcorrs, dt))

    if ini.band_window:

        t0_band_window = timer()

        if not os.path.isdir(path_to_signal):
            os.makedirs(path_to_signal)

        identifier_SM = get_signal_matrices_per_multipole(path_to_signal, path_to_signal_store, x_rad, y_rad)

        for idx_zbin1 in xrange(nzbins):
            for idx_zbin2 in xrange(idx_zbin1 + 1):

                print_ascii_art('Calculating now matrix of band window functions for correlation z{:}xz{:}.'.format(idx_zbin2 + 1, idx_zbin1 + 1))

                # here we load everything needed for calculation of BWM under the assumption that optimization was done before.
                # but only if we did not optimize in same run
                if not ini.calculate_only_signal_matrices_for_band_window_matrix:

                    try:
                        # ATTENTION: the Fisher matrix calculated in the code is two times larger than in the definition of Eq. (20) of Hu & White (2001)
                        # but for subsequent calculations we expect to load the inverse of the too large Fisher matrix, that is why we load this file below:
                        print('The final Fisher matrix will be used to calculate the band power covariance!')
                        fname = 'last_larger_Fisher_matrix_z{:}xz{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1)
                        last_Fisher = np.loadtxt(fname)
                        # load also the last covariance matrix:
                        fname = 'last_covariance_matrix_z{:}xz{:}.h5'.format(idx_zbin2 + 1, idx_zbin1 + 1)
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
                        print('Could not load expected data for calculation of band window matrix.')
                        exit()

                t0_band_window_matrix = timer()

                # write all derivative matrices to cache:
                dim_signals = get_derivative_matrices_per_band(path_to_signal, bands_max, identifier_SM, inv_N_eff_gal, idx_zbin1, idx_zbin2)
                idx_linear = get_derivative_matrices_per_multipole(path_to_signal, identifier_SM, idx_zbin1, idx_zbin2)

                print(r'Calculating C^{{-1}}_{{ij}} (C,\alpha)_{{jk}} for current z{:}xz{:} band window matrix.'.format(idx_zbin2 + 1, idx_zbin1 + 1))
                success = compute_inv_covariance_times_deriv_matrices(dim=dim_signals, ncpus=ini.max_ncpus, for_band_window=False)

                if success:
                    print(r'Done with calculation of C^{-1}_{ij} (C,\alpha)_{jk}.')

                print('First step for band window matrix: C^{-1}_{ij} (C,\ell)_{jk}')
                success = compute_inv_covariance_times_deriv_matrices(dim=idx_linear, ncpus=ini.max_ncpus, for_band_window=True)

                if success:
                    print('Done with first step.')

                print('Second step for calculation of band window matrix.')

                # inv_covarianceTimesDerivMatrices MUST be the ones used in Fisher algorithm and NOT inv_covarianceTimesDerivMatrices_l!!!
                trace_matrix, dt_trace = get_trace_matrix(dim_alpha=len(parameter_covariance), dim_ell=idx_linear, ncpus=ini.max_ncpus)
                print('Time for second step of calculation of band window matrix: {:.2f}min.'.format(dt_trace / 60.))

                window_matrix, dt_window_matrix = get_window_matrix(parameter_covariance, trace_matrix)

                fname = 'band_window_matrix_z{:}xz{:}_nell{:}.dat'.format(idx_zbin2 + 1, idx_zbin1 + 1, ini.nell)
                np.savetxt(fname, np.asarray(window_matrix))
                print('Time for calculation of matrix of band window functions for correlation z{:}xz{:}: {:.2f}min.'.format(idx_zbin2 + 1, idx_zbin1 + 1, dt_window_matrix / 60.))

        dt_band_window = timer() - t0_band_window
        print('Total time for calculation of all band window matrices: {:.2f}min.'.format(dt_band_window / 60.))

    runtime = timer() - t0_prog
    print('Done. Total runtime: {:.2f}min.'.format(runtime / 60.))
    print('All results were saved to: \n', path_output)
    # last step: removal of 'cache'-folder if requested:
    if ini.delete_cache:
        print('Removing cache folder now.')
        import shutil
        shutil.rmtree('cache/')

    if ini.root_run != ini.root_store:
        t0_store = timer()
        print('Copying all files now to: \n', path_store)
        import shutil
        shutil.copytree(path_output, path_store)
        os.chdir(path_store)
        shutil.rmtree(path_output)
        print('Done in {:.2f}s.'.format(timer() - t0_store))
    else:
        print('Done.')