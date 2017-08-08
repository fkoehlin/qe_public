#!/usr/bin/env python
# encoding: UTF8
# script plots band powers
# This is the main script. Call signature: python /path/to/plot_band_powers.py </path/to/input_parameters.plt>
# see default.plt for parameters!

import os
import sys
import numpy as np
import scipy.interpolate as interpolate
# for avoiding type 3 fonts:
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import time

def get_input_parameters(filename):
    ''' Helper function that reads in required input parameters from file <filename>.
        We always assume that this file is using Python notation for variables and follows the in notation and variable names "default.ini"! 
    '''
    import imp
    f = open(filename)
    global para
    para = imp.load_source('data', '', f)
    f.close()
    
    # here is the place to include safety checks!
    
    return

def get_correlation_matrix(covariance_matrix):
    
    start = time.time()
    
    cov = covariance_matrix / np.sqrt(np.asmatrix(np.diag(covariance_matrix)).T * np.asmatrix(np.diag(covariance_matrix)))
    
    t = time.time() - start
    
    return cov, t

def get_noise_power_spectrum(p_noise):
    
    ells = np.logspace(-4, 4, 300)
    
    return ells, ells * (ells + 1.) / 2. / np.pi * p_noise

def cm2inch(value):
    return value/2.54

def arcmin2rad(x):
    return np.deg2rad(x/60.)

def theta_to_ell(theta, unit='arcmin'):    
    
    if unit in ['arcmin']:
        theta_deg = theta/60.
    elif unit in ['deg', 'degree']:
        theta_deg = theta
    else:
        print 'Specify valid unit for input (arcmin or deg)!'
        print 'Input is assumed to be in arcmin by default!'
        theta_deg = theta/60.
        
    theta_rad = np.deg2rad(theta_deg)
    ell = 2.*np.pi/theta_rad
    
    return ell

def ell_to_theta(ell, unit='arcmin'):
    
    theta_rad = 2.*np.pi/ell
    
    if unit in ['arcmin']:
        theta_arcmin = np.rad2deg(theta_rad)*60.
        theta_out = theta_arcmin 
    elif unit in ['deg', 'degree']:
        theta_deg = np.rad2deg(theta_rad)
        theta_out = theta_deg
    else:
        print 'Specify valid unit for output (arcmin or deg)!'
        print 'Output is in arcmin by default!'
        theta_arcmin = np.rad2deg(theta_rad)*60.
        theta_out = theta_arcmin 
    
    return theta_out

def get_theoretical_power_spectrum(path, index_corr):
    
    #print index_corr    
    
    fname = path   
    
    data = np.loadtxt(fname)
    ells = data[:, 0]
    
    # new files only contain unique cross-spectra (ij = ji)!
    D_ell = data[:, index_corr + 1]
    
    return ells, D_ell

def get_shot_noise(ell_min, ell_max, nell=300, nzbins=1):

    fname = para.path_to_noise_parameters
    # columns: zmin, zmax, sigma_e1, sigma_e2, neff, nobj
    noise_data = np.loadtxt(fname)
    if nzbins > 1:
        sigma_e1 = noise_data[:, 2]
        sigma_e2 = noise_data[:, 3]
        n_eff = noise_data[:, 4]
    else:
        sigma_e1 = noise_data[2]
        sigma_e2 = noise_data[3]
        n_eff = noise_data[4]

    sigma_int_sqr = (sigma_e1**2 + sigma_e2**2) / 2.
    n_eff_per_sr = n_eff 
    shot_noise = sigma_int_sqr * arcmin2rad(1.)**2 / n_eff
    if nzbins == 1:
        shot_noise = np.asarray(shot_noise)
    #print shot_noise
    ells = np.linspace(ell_min, ell_max, nell)
    if nzbins > 1:
        p_noise = ells * (ells + 1.) / (2. * np.pi) * shot_noise[:, np.newaxis]
    else:
        p_noise = ells * (ells + 1.) / (2. * np.pi) * shot_noise
    #print p_noise.shape
    return ells, p_noise, shot_noise, n_eff

def get_slicing_points_and_offset(root_path, band_type, nzcorrs):
     # initial guesses are the same if patches == 'all'!
    
    # we load all common files from first entry in list:
    path_in = root_path + para.patches[0] + '/'
    print path_in
    
    try:
        fname = path_in + 'initial_guess_' + band_type + '.dat'
        #print fname
        ells, bands_min, bands_max, initial_band_powers = np.loadtxt(fname, unpack=True)
        print 'Loaded borders of bands and initial guess for band powers.' 
        print bands_min
        print bands_max
    except:
        print 'Could not load borders of bands and initial guess.'
        exit()
    
    if band_type == 'BB':
        initial_band_powers_EE = np.loadtxt(path_in + 'initial_guess_EE.dat', usecols=(3,))
        #slicing_points_EE = [0, initial_band_powers_EE.size]
        # factor 4 because of 2 redshift_bins per band
        # TODO: generally these factors should be changed to nzbins(nzbins+1)/2 instead of nzbins^2:
        slicing_points = [nzcorrs * initial_band_powers_EE.size, nzcorrs * (initial_band_powers.size + initial_band_powers_EE.size)]
    elif band_type == 'EE':
        # factor 4 because of 2 redshift_bins per band
        slicing_points = [0, nzcorrs * initial_band_powers.size]
   
    # EE: e.g. 7; BB: e.g. 6
    band_offset = initial_band_powers.size 
    
    return slicing_points, band_offset, ells, bands_min, bands_max

def collect_outputs(root_path, path_out, band_type, correlation, slicing_points, band_offset, index_corr, nzcorr):
    
    #index_zbin1, index_zbin2 = correlation_indices[index_corr][0], correlation_indices[index_corr][1]    
    
    # some collector lists:
    collect_last_powers = []
    #collect_variances = []
    collect_inv_variances = []
    collect_variances_ext = []
    collect_weights_ext = []
    collect_convergence = []
    collect_theory = []
    collect_bwm = []
    if para.plot_band_correlations and index_corr == 0:
        fig = plt.figure(figsize=(cm2inch(29.7), cm2inch(21.0)))#, tight_layout=True)
    for i in range(len(para.patches)):
        # loading files now:
        path_in = root_path + para.patches[i] + '/'
        #identifier = 'sigma{:.2f}_'.format(para.sigma_int) + para.patches[i]
        redshift_id = correlation #'z{:}xz{:}'.format(index_zbin1+1, index_zbin2+1)
        print redshift_id        
        fname = path_in + 'all_estimates_band_powers_' + band_type + '_' + redshift_id + '.dat'
        #print fname
        powers_of_run = np.loadtxt(fname)
        #print powers_of_run[-1]
        fname = path_in + 'last_Fisher_matrix.dat'
        # We have to load the Fisher matrix and invert it here!
        Fisher = np.load(fname)
        # avoid any unnecessary matrix inversion for the sake of less numerical noise...
        # we treat the diagonal of the fisher matrix as inv_variance!
        # I should also invert the sliced Fisher matrix, again due to minimize numerical noise...
        inv_Fisher = np.linalg.inv(Fisher)
        #print inv_Fisher.shape
        
        if para.plot_band_window_functions:
            fname = path_in + 'band_window_matrix_nell{:}.dat'.format(int(para.ell_nodes))
            band_window_matrix = np.loadtxt(fname)
            fname = path_in + 'multipole_nodes_for_band_window_functions_nell{:}.dat'.format(int(para.ell_nodes))
            l_intp_bwm = np.loadtxt(fname)
        else:
            # just carry around some "empty" variables
            band_window_matrix = 0.
            l_intp_bwm = 0.
        collect_bwm.append(band_window_matrix)
        # convergence criteria are only for EE bands!
        fname = path_in + 'difference_band_powers_EE_' + redshift_id + '.dat'
        conv = np.loadtxt(fname)
        #print np.shape(conv)
        # this slices between bands (currently EE, BB)
        print band_type
        #print slicing_points[0], slicing_points[-1], band_offset
        #variances_bands = np.diag(inv_Fisher)[slicing_points[0]:slicing_points[-1]]
        diag_Fisher = np.diag(Fisher)
        inv_variances_bands = diag_Fisher[slicing_points[0]:slicing_points[-1]]
        print 'full diag(Fisher): \n', np.diag(Fisher)
        print 'sliced diag(Fisher): \n', inv_variances_bands
        # this is pure laziness... I'll always use the same external covariance for all patches at the moment
        # TODO: Make this more general
        # this won't work with KiDS450 data...        
        if para.external_covariance:
            if para.type_ext_cov == 'clones':
                fname = para.root_in_ext_cov + '/clones/good/sigma_int{:.2f}/'.format(para.sigma_int) + para.redshift_cut + '/M0_to_M183/covariance_all_z_clones_EE_' + para.patches[i][-2:] + '.npy'
            elif para.type_ext_cov == 'grfs':
                fname = para.root_in_ext_cov + '/grfs_' + para.patches[i][-2:] + '_flip/good/sigma_int{:.2f}/'.format(para.sigma_int) + para.redshift_cut + '/G1_to_G184_' + para.patches[i][-2:] + '/covariance_all_z_grfs_EE_' + para.patches[i][-2:] + '.npy'
            elif para.type_ext_cov == 'stitched':
                fname = para.root_in_ext_cov + '/stitched_covariances/full/covariance_all_z_stitched_EE_' + para.patches[i][-2:] + '.npy'    
            # TODO: for now I'm using the inverse Fisher for B-modes!!!
            elif para.type_ext_cov == 'kids450':
                # full path must be supplied down to filename (because of blinding!!!)  
                if band_type == 'EE':
                    fname = para.root_in_ext_cov + 'covariance_all_z_' + band_type + '.dat'
                elif band_type == 'BB':
                    #fname = para.root_in_ext_cov + 'inv_Fisher_avg_BB.npy'
                    fname = para.root_in_ext_cov + 'covariance_all_z_' + band_type + '.dat'
            cov_ext = np.loadtxt(fname)
            # take thefull diagonal (for BB we currently use the external EE covariance!)
            variances_ext_bands = np.diag(cov_ext)#[slicing_points[0]:slicing_points[-1]]
        else:
            # just carry zeros around for convenience and less ifs:
            variances_ext_bands = np.zeros_like(diag_Fisher)
        # TODO: this is hard-linked to current KiDS450 approach (i.e. noise-covariances are saved in data-folders)...
        # TODO: Now we're only using the area to scale everything...
        if para.external_weights:
            '''
            if band_type == 'EE':            
                fname = path_in + 'covariance_all_z_noise.npy'
            elif band_type == 'BB':
                fname = path_in + 'covariance_all_z_BB.npy'
            weight_matrix = np.load(fname)
            weights_ext_bands = 1. / np.diag(weight_matrix)
            '''
            # we only need the e.g. "G9W"-part...
            fname = para.path_to_weights + 'effective_area_' + para.patches[i][5:-2] + '.txt'
            A_eff = np.loadtxt(fname, usecols=[0], delimiter=', ')            
            weights_ext_bands = np.ones_like(diag_Fisher) * A_eff
        else:
            # just carry zeros around for convenience and less ifs:
            weights_ext_bands = np.zeros_like(diag_Fisher)
        #exit()
        # this will slice between redshift bins:
        # don't ever change the order and make it more general at some point!
        # treat BB-bands here as well!
        # NEW ORDERING HERE DUE TO REVISED FISHER MATRICES!!!
        print inv_variances_bands, inv_variances_bands.shape
        print variances_ext_bands, variances_ext_bands.shape 
        print band_type, correlation
        # TODO: Generalize this for arbitrary number of correlations!        
        if band_type == 'BB':
            skip_first_band = 1
        elif band_type == 'EE':
            skip_first_band = 0  
        
        # create list of slicing points for slicing of z-correlations:
        index_pairs = []
        index_jump = band_offset
        for index_z in np.arange(0, nzcorrs * band_offset, band_offset):            
            index_pairs.append((index_z, index_jump))
            index_jump += band_offset
        
        inv_variances = inv_variances_bands[index_pairs[index_corr][0]:index_pairs[index_corr][1]]
        weights_ext = weights_ext_bands[index_pairs[index_corr][0]:index_pairs[index_corr][1]]
        
        if para.type_ext_cov == 'kids450':
            variances_ext = variances_ext_bands[index_pairs[index_corr][0]:index_pairs[index_corr][1]]
        else:
            # TODO: DON'T EVER USE E-MODE COVARIANCE FOR B-MODES!!!
            # more complicated, because we use EE errors for BB-modes, too, but have to skip the first band:        
            variances_ext = variances_ext_bands[index_corr * (band_offset + skip_first_band) + skip_first_band:(index_corr + 1) * (band_offset + skip_first_band)]
        # sanity check:
        '''
        for i in range(3):
            mult1 = i
            mult2 = mult1+1
            variances_ext = variances_ext_bands[mult1*(band_offset+skip_first_band)+skip_first_band:mult2*(band_offset+skip_first_band)]        
            print 'New var_ext: \n', variances_ext
        print band_offset
        print variances_ext_bands, len(variances_ext_bands)
        print 'Old var_ext z1xz1: \n', variances_ext_bands[skip_first_band:band_offset+skip_first_band]
        print 'Old var_ext z1xz2: \n', variances_ext_bands[band_offset+skip_first_band+skip_first_band:2*(band_offset+skip_first_band)]
        print 'Old var_ext z2xz2: \n', variances_ext_bands[2*(band_offset+skip_first_band)+skip_first_band:]
        for i in range(3):
            inv_variances = inv_variances_bands[index_pairs[i][0]:index_pairs[i][1]] 
            print 'New inv_var_ext: \n', inv_variances        
        print inv_variances_bands, inv_variances_bands.shape 
        print 'Old inv_var z1xz1: \n', inv_variances_bands[:band_offset]
        print 'Old inv_var z1xz2: \n', inv_variances_bands[band_offset:2*band_offset]
        print 'Old inv_var z2xz2: \n', inv_variances_bands[2*band_offset:]
        exit()    
        '''
        
        # TODO: Make this more general, so that I can plot it for data, too (e.g. later for best fit PS etc...)
        if para.plot_convolved_theory and band_type == 'EE':
            # TODO: allow all theories to be convolved...
            D_ell = get_convolved_theory(path_in, para.path_to_theory[0], correlation, para.patches[i], band_offset, index_corr)
        else:
            D_ell = np.zeros((band_offset))
        #print variances_bands[:band_offset]
        #print variances_bands[3*band_offset:]
        #print variances
        #print variances.shape
        #errors = np.sqrt(variances)
        # only collect band powers after last iteration
        collect_last_powers.append(powers_of_run[-1])
        #collect_errors.append(errors)
        collect_inv_variances.append(inv_variances)
        collect_variances_ext.append(variances_ext)
        collect_weights_ext.append(weights_ext)
        # calculate convergence in percent
        collect_convergence.append(conv[-1])
        #all_powers.append(last_powers)
        collect_theory.append(D_ell)
        
        #calculate/load normed covariances:
        # that is actually the correlation matrix here...
        fname = path_in + 'correlation_matrix_' + band_type + '.dat'
        # TODO: Remove 'bla'?!        
        if os.path.isfile(fname + 'bla'):
            correlation_matrix = np.loadtxt(fname)
        else:    
            correlation_matrix, dt = get_correlation_matrix(inv_Fisher)
            np.savetxt(fname, correlation_matrix)
            
        if para.plot_band_correlations and index_corr == 0:
            npatches = len(para.patches)
            if npatches > 1.:
                a, b = get_xy_for_subplots(npatches)
                ax = fig.add_subplot(a, b, i + 1)
            else:
                ax = fig.add_subplot(1, 1, i + 1)
            #q = normed_covariance
            # correlations range from [-1, 1]:
            #img = ax.imshow(q, interpolation='None', origin='lower', aspect='auto', cmap='YlGnBu', vmin=vmin, vmax=vmax)
            img = ax.matshow(correlation_matrix, cmap='seismic', vmin=-1., vmax=1.)
            ax.tick_params(axis='both', labelsize=7)
            ax.set_title(para.patches[i], fontsize=12)
            
            # magic numbers for correct scaling of colorbar
            c = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)#, label='$\log_{10}(\chi^2)$')
            cbytick_obj = plt.getp(c.ax.axes, 'yticklabels')          
            plt.setp(cbytick_obj, fontsize=7)
    
    if para.plot_band_correlations and index_corr == 0:
        for file_type in para.file_types:
            fig.savefig(path_out + 'plots/band_correlations.' + file_type)
            print 'Plot saved to:', path_out + 'plots/band_correlations.' + file_type
    
    # this will write out the convergence to a file:         
    low = 0
    high = np.shape(collect_convergence)[1]
    percent_convergence = np.asarray(collect_convergence)[:, low:high]
    print 'Percentage of achieved convergence in bands {:} to {:}: \n'.format(low + 1, high), percent_convergence
    header = 'Band ' + str(low + 1)
    for i in np.arange(low + 2, high + 1, 1):
        header += ', Band ' + str(i) 
    #print header
    fname = path_out + 'percentages_of_convergence_' + correlation + '_' + band_type + '.dat'
    np.savetxt(fname, percent_convergence, header=header)

    # most important parameters:
    last_powers = np.asarray(collect_last_powers)
    inv_variances = np.asarray(collect_inv_variances)  
    variances_ext = np.asarray(collect_variances_ext)  
    convolved_theory = np.asarray(collect_theory)
    weights_ext = np.asarray(collect_weights_ext)  
    
    return last_powers, inv_variances, variances_ext, convolved_theory, l_intp_bwm, collect_bwm, weights_ext

def get_xy_for_subplots(n):
    """ Very complicated function to get n subplots distributed on a (x, y) grid """
    
    prime_factors = get_prime_factors(n)
    #print prime_factors
    nprimes = len(prime_factors) 
    
    if nprimes > 2:
        x = prime_factors[0]
        i = 1
        while i < nprimes - 1:
            x *= prime_factors[1]
            i += 1
        y = prime_factors[-1]
    elif nprimes == 1:
        #'''
        #print prime_factors[0]
        if prime_factors[0] > 3:
            x, y = get_xy_for_subplots(prime_factors[0] + 1)
        else:
        #'''
            x, y = 1, prime_factors[0]
    else:        
        x, y = prime_factors[0], prime_factors[1]
    
    #print x, y
    return x, y

def get_prime_factors(n):
    """ Stackoverflow """
    
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    else:
        factors.append(1)

    return factors
    
def get_averages_and_errors(band_powers, inv_variances, variances_ext, convolved_theory, band_type, weights_ext):

    band_mean = np.zeros(band_powers.shape[1])
    band_avg1 = np.zeros(band_powers.shape[1])
    band_avg2 = np.zeros(band_powers.shape[1])
    err_run_to_run = np.zeros_like(band_mean)
    #err_fisher_avg = np.zeros_like(band_mean)
    #err_fisher_alt = np.zeros_like(band_mean)    
    err_avg1 = np.zeros_like(band_mean)    
    err_avg2 = np.zeros_like(band_mean)    
    conv_theo_avg = np.zeros_like(band_mean)    
    
    if band_type == 'BB':
        bands_to_plot = para.bands_to_plot[1:]
    else:
        bands_to_plot = para.bands_to_plot
    # set masked values in data and theory vector (over all z-correlations) to 0:
    index1 = np.where(np.asarray(bands_to_plot) == 1)[0]
    
    # i runs over number of bands (7 so far)
    for i in range(band_powers.shape[1]):
        
        # What about the factor 1/sqrt(N)?
        sqrt_N = np.sqrt(len(para.patches))
        
        band = band_powers[:, i]
        inv_var = inv_variances[:, i]
        #print inv_var
        #exit()
        #var = 1. / inv_var
        var_ext = variances_ext[:, i]
        theo = convolved_theory[:, i]
        #print var_ext.shape
        # I should apply here the debiasing to the inverse of var_ext though, shouldn't I?!
        # Eq. 16 from Heymans et al. 2013 (arxiv:1303.1808v1)
        # this should be the total number of bands extracted in order to create the covariance (irrespective of ignoring some of its rows or columns later on)
        # discussion with Massimo: this should be the number of bands considered:
        p = len(index1) #self.nbands_tot
        debias_factor = (para.nruns_ext - p - 2.) / (para.nruns_ext - 1.)
        # TODO: allow also external weights! (needed for KiDS450...)        
        if para.external_weights:
            weight1 = weights_ext[:, i]
        else:
            weight1 = (1. / var_ext) * debias_factor # np.ones_like(inv_var) * 13. * inv_var[0]
        weight2 = inv_var #1./var, np.ones_like(inv_var) * 13. * inv_var[0]
        band_avg1[i] = np.sum(weight1 * band) / np.sum(weight1)
        band_avg2[i] = np.sum(weight2 * band) / np.sum(weight2)
        band_mean[i] = band.mean()
        err_run_to_run[i] = band.std() / sqrt_N
        
        if para.fisher_errors and not para.external_covariance:            
            conv_theo_avg[i] = np.sum(weight2 * theo) / np.sum(weight2)
        else:
            conv_theo_avg[i] = np.sum(weight1 * theo) / np.sum(weight1)
            
        # old:
        # I'm mis-using now err_run_to_run for propagating error estimates from an external covariance (e.g. as estimated from clones etc.)
        # However: taking the mean should have no effect, since for all patches W<i>, I currently load the same external covariance
        #err_run_to_run[i] = np.sqrt(var_ext.sum()) / sqrt_N
        #err_run_to_run[i] = np.sqrt(var_ext.mean()) #band.std(ddof=0)/sqrt_N 
        # new:
        # after discussion with HENK:
        #err_fisher_avg[i] = np.sqrt(var.sum()) / sqrt_N
        # this should be identical to:
        #err_fisher_alt[i] = np.sqrt(var.mean())
        if para.external_weights:
            # mean is correct here, because we supply the same covariance for each patch just for not breaking the whole structure here
            # then the naive mean should just yield the original value from the diagonal again!!!
            err_avg1[i] = np.sqrt(np.mean(var_ext))
        else:
            # this applies only to inverse-variance weighting!
            err_avg1[i] = np.sqrt(1. / np.sum(weight1))
        err_avg2[i] = np.sqrt(1. / np.sum(weight2))
    #print 'Average band powers (per band): \n', band_mean
    #print 'Average errors from run to run: \n', err_run_to_run
    #print 'Average Fisher-errors: \n', err_fisher_avg
    #print 'Weighted errors on weighted average of bands: \n', err_avg1
    #print 'ALTERNATIVE Average Fisher-errors: \n', err_fisher_alt
    #print 'Average band powers + run-to-run errors: \n', band_mean+err_run_to_run
    #print 'Average band powers - run-to-run errors: \n', band_mean-err_run_to_run
    print '### Averaging method 1 (external covariance) ###'
    print 'Weighted average band powers: \n', band_avg1
    print 'Err_avg: \n', err_avg1
    print 'Weighted average band powers + err_avg: \n', band_avg1 + err_avg1
    print 'Weighted average band powers - err_avg: \n', band_avg1 - err_avg1
    
    print '### Averaging method 2 (Fisher errors) ###'
    print 'Weighted average band powers: \n', band_avg2
    print 'Err_avg: \n', err_avg2
    print 'Weighted average band powers + err_avg: \n', band_avg2 + err_avg2
    print 'Weighted average band powers - err_avg: \n', band_avg2 - err_avg2
    
    print '### Naive mean and std/sqrt(N) ###'
    print 'Weighted average band powers: \n', band_mean
    print 'Err_avg: \n', err_run_to_run
    print 'Weighted average band powers + err_avg: \n', band_mean + err_run_to_run
    print 'Weighted average band powers - err_avg: \n', band_mean - err_run_to_run    
    
    print 'Mean over theory convolved with BWM: \n', conv_theo_avg
    #exit()
    
    return band_avg1, band_avg2, band_mean, err_avg1, err_avg2, err_run_to_run, conv_theo_avg

def plot_band_powers_single(root_path, path_out, band_type, l, band_avg1, band_avg2, band_mean, bands_min, bands_max, err_avg1, err_avg2, err_run_to_run, conv_theo_avg, correlation, index_corr): # , band_powers): 
    
    #index_zbin1, index_zbin2 = correlation_indices[0], correlation_indices[1]
    
    #label_ext_cov = {'grfs': r'$GRF \ errors$', 'clones': r'$Clone \ errors$', 'stitched': r'$GRF \& Clone errors$'}
    
    if band_type == 'BB':
        bands_to_plot = para.bands_to_plot[1:]
    else:
        bands_to_plot = para.bands_to_plot
    # set masked values in data and theory vector (over all z-correlations) to 0:
    index1 = np.where(np.asarray(bands_to_plot) == 1)[0]
    
    # set some colors:
    if correlation == 'z1z1':
        color_Fisher = 'red'
    elif correlation == 'z2z2':
        color_Fisher = 'blue'
    else:
        color_Fisher = 'grey'
   
    # I should actually just load the file I have created instead of passing all variables... Not possible, since file doesn't contain "bands_<min, max>"...
    
    # TODO: Fix this...
    shot_noise, shot_noise_nominal, field_area_deg, n_eff_tot, load_fail = calculate_shot_noise(root_path)  
    
    fig1 = plt.figure(figsize=(cm2inch(29.7),cm2inch(21.0))) #, tight_layout=True)
    #fig1 = plt.figure(figsize=(cm2inch(18./3.), cm2inch(18./3.)*3./4.), tight_layout=True)
    ax1 = fig1.add_subplot(111)
    # make a twin axis for theta!
    ax2 = ax1.twiny()
    # some additional labelling...
    fake_point = (1e-6, 1e-6)
    if band_type in ['EE', 'BB']:
        if load_fail:
            title = r'$ \ z_{:}  \times \, z_{:}$'.format(index_zbin1, index_zbin2)
            ax1.text(.5 ,.9, title, horizontalalignment='center', transform=ax1.transAxes, fontsize=para.fontsize_legend)
        else:
            title = r'$ \ z_{:}  \times \, z_{:}$'.format(index_zbin1, index_zbin2)
            ax1.text(.5 ,.9, title, horizontalalignment='center', transform=ax1.transAxes, fontsize=para.fontsize_legend)
            ax1.plot(fake_point, ls='None', label=r'$A_{eff} \, \leq \, $'+r'${:.4}$'.format(field_area_deg)+r'$\, \mathrm{deg^2}$'+'\n'+
                                                  r'$n_{eff} \, = \,$'+r'${:.2f}$'.format(n_eff_tot)+r'$ \, \mathrm{arcmin^{-2}}$')
            
        text_label_Fisher = r'$Fisher \ errors$'
        text_label_run = r'$Run-to-Run \ errors$'
        text_label_initial = r'$Initial \ guess$'
    else:
        text_label_Fisher = None
        text_label_run = None
        text_label_initial = None
    
    if para.box_plot:
        # Fisher boxes/ external covariance boxes:
        wx = bands_max[index1]-bands_min[index1]
        if para.external_covariance and not para.fisher_errors:
            hy = 2.*err_avg1[index1]
            ax1.bar(bands_min[index1], hy, bottom=np.abs(band_avg1[index1])-err_avg1[index1], width=wx, align='edge', facecolor=color_Fisher, edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
        elif not para.external_covariance and para.fisher_errors:
            hy = 2.*err_avg2[index1]
            ax1.bar(bands_min[index1], hy, bottom=np.abs(band_avg2[index1])-err_avg2[index1], width=wx, align='edge', facecolor=color_Fisher, edgecolor='None', alpha=0.5, label=None) #r'$Fisher \ errors$')
        elif para.external_covariance and para.fisher_errors:
            hy = 2.*err_avg1[index1]
            ax1.bar(bands_min[index1], hy, bottom=np.abs(band_avg1[index1])-err_avg1[index1], width=wx, align='edge', facecolor=color_Fisher, edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
            hy = 2.*err_avg2[index1]
            ax1.bar(bands_min[index1], hy, bottom=np.abs(band_avg2[index1])-err_avg2[index1], width=wx, align='edge', color='None', label=r'$Fisher \ errors$', linewidth=2, ls='dashed')
        # run-to-run boxes (but run-to-run now with respect to external run on e.g. clones):
        #'''
        # if nfields > 1:
        # don't show this anymore:
        '''
        if para.external_covariance:
            wx = bands_max-bands_min
            hy = 2.*err_avg2
            #label_text = r'$Run-to-Run \ errors \ from \ $'+'$'+str(para.nruns_ext)+'$'+'$ \ clones$'
            label_text = r'$Run-to-Run \ errors$'
            #label_text = r'$Run-to-Run \ errors$'
            ax1.bar(bands_min, hy, bottom=band_avg2-err_avg2, width=wx, align='edge', color='None', linewidth=2, ls='dashed', label=label_text)
        '''
    else:
        #ax1.errorbar(l[index1], band_avg2[index1], yerr=err_avg2[index1], color='red', label=r'$Run-to-Run \ errors$', elinewidth=8)#, ls='')
        ax1.errorbar(l[index1], band_avg1[index1], yerr=err_avg1[index1], color='blue', label=None, elinewidth=2)#, ls='')r'$Fisher \ errors$'
        ax1.axhline(0., ls='--', color='black')
    #ax1.plot(l, band_mean, color='black')
    #ax1.plot(l, initial_band_powers, color='black', ls='--', label=r'$Initial \ guess$')
    # usually that is the inverse variance weighted band power using the inverse Fisher matrix
    if para.external_covariance and not para.fisher_errors:
        mask = np.sign(band_avg1[index1]) < 0.
        ax1.scatter(l[index1], band_avg1[index1], s=para.sp, color='black', marker='o')
        ax1.scatter(l[mask], np.abs(band_avg1[mask]), s=para.sp, edgecolor='black', facecolors='None', marker='o')
    elif not para.external_covariance and para.fisher_errors:
        mask = np.sign(band_avg2[index1]) < 0.
        ax1.scatter(l[index1], band_avg2[index1], s=para.sp, color='black', marker='o')
        ax1.scatter(l[mask], np.abs(band_avg2[mask]), s=para.sp, edgecolor='black', facecolors='None', marker='o')
    elif para.external_covariance and para.fisher_errors:
        mask = np.sign(band_avg1[index1]) < 0.
        ax1.scatter(l[index1], band_avg1[index1], s=para.sp, color='black', marker='o')
        ax1.scatter(l[mask], np.abs(band_avg1[mask]), s=para.sp, edgecolor='black', facecolors='None', marker='o')
        mask = np.sign(band_avg2[index1]) < 0.
        ax1.scatter(l[index1], band_avg2[index1], s=para.sp, color='black', marker='^')
        ax1.scatter(l[mask], np.abs(band_avg2[mask]), s=para.sp, edgecolor='black', facecolors='None', marker='^')
    
    '''
    # plain extracted band powers per patch:
    for i in range(len(band_powers)):
        mask = np.sign(band_powers[i,:]) < 0.
        ax1.scatter(l, band_powers[i,:], s=para.sp, color='orange', marker='h')
        ax1.scatter(l[mask], np.abs(band_powers[i,mask]), s=para.sp, edgecolor='orange', facecolors='None', marker='h')
    '''
    # shot noise:
    ell_noise, p_noise = get_noise_power_spectrum(shot_noise)
    # only for labelling of shot noise:
    #label = r'$shot \ noise \ (measured, \ \sigma_{int}=$'+'{:.2f}'.format(para.sigma_int)+r'$)$'
    #label = None
    #ax1.plot(l_noise, p_noise, color='black', label=label, ls='--')
    
    ell_noise, p_noise = get_noise_power_spectrum(shot_noise_nominal)
    #ax1.plot(ell_noise, p_noise, color='red', label=r'$shot \ noise \ (Heymans \ et \ al. \ 2013)$')
    
    # for comparison with theoretical prediction:    
    if band_type == 'EE':
        if para.plot_theory:
            for index_theo, path in enumerate(para.path_to_theory):
                # TODO This is a bit inefficient...
                ell_theo, D_ell_lcdm = get_theoretical_power_spectrum(path, index_corr)
                ax1.plot(ell_theo, D_ell_lcdm, color=para.color_theory[index_theo], ls=para.ls_theory[index_theo], lw=para.lw_theory[index_theo], label=para.label_theory[index_theo])
        if para.plot_convolved_theory:            
            mask = np.sign(conv_theo_avg[index1]) < 0.
            ax1.scatter(l, conv_theo_avg[index1], color='red', marker='o', s=30, lw=2, label=r'$\mathcal{B}_\alpha^{theo} = \sum_\ell W_{\alpha \ell} D_\ell^{theo}$')
            ax1.scatter(l[mask], np.abs(conv_theo_avg[mask]), edgecolor='red', facecolors='None', marker='o', s=50, lw=2)
    
    ax2.set_xlabel(r'$\vartheta \ (arcmin)$', fontsize=para.fontsize_label)    
    ax1.set_xlabel(r'$\ell$', fontsize=para.fontsize_label)
    if correlation is 'z1z1':
        ax1.set_ylabel(r'$\ell(\ell+1)C_{\mathrm{EE}}(\ell)/2\mathrm{\pi}$', fontsize=para.fontsize_label)
    ax1.legend(loc='upper left', fontsize=para.fontsize_legend, frameon=False)
    ax1.set_yscale('log', nonposy='clip')
    ax1.set_xscale('log', nonposx='clip')
    ax1.set_ylim([para.y_low, para.y_high])
    ell_low = para.ell_low 
    ell_high = para.ell_high
    ax1.set_xlim([ell_low, ell_high])
    theta_high = ell_to_theta(ell_high, unit='arcmin')
    theta_low = ell_to_theta(ell_low, unit='arcmin')
    #print theta_low, theta_high
    ax2.set_xlim([theta_low, theta_high])
    ax2.set_xscale('log')
    ax1.tick_params(axis='both', labelsize=para.fontsize_ticks)
    ax2.tick_params(axis='both', labelsize=para.fontsize_ticks)
    #'''
    if correlation is not 'z1z1':
        ax1.yaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
    #'''
    #ax1.annotate(text_annot, xy=pos, ha='left', va='top', fontsize=34)
    fname = 'extracted_band_powers_'+correlation+'_'+band_type+'_{:}fields.'.format(len(para.patches))
    for file_type in para.file_types:
        if file_type == 'png':
            transparent = False #True
        else:
            transparent = False
        fig1.savefig(path_out+'plots/'+fname+file_type, transparent=transparent, dpi=500)
        print 'Plot saved to:', path_out+'plots/'+fname+file_type
    
    return
    
def get_interpolated_theoretical_power_spectrum(path, ells, index_corr):
        
    ells_sample, D_ell_sample = get_theoretical_power_spectrum(path, index_corr)
        
    D_ell = interpolate.interp1d(np.log(ells_sample), np.log(D_ell_sample), kind='linear')
    
    return np.exp(D_ell(np.log(ells)))

# TODO: Make this consistent again with new BWM dimensionality!!!
def get_convolved_theory(path_to_bwm, path_to_theory, correlation, patch, band_offset, index_corr):
    # use \ell's of convolution matrix as nodes for interpolation! 
    
    # 1) load l-range on which we want to interpolate BWM and theoretical PS:
    ells_intp = np.loadtxt(path_to_bwm + 'multipole_nodes_for_band_window_functions_nell{:}.dat'.format(para.ell_nodes))
    
    # 2) determine \ell-range for taking the sum:
    nell = ells_intp[-1] - ells_intp[0] + 1
    ells_sum = np.linspace(ells_intp[0], ells_intp[-1], nell).astype(int)
    
    # 3) get interpolation of theoretical PS over this range:
    D_ell = get_interpolated_theoretical_power_spectrum(path_to_theory, ells_sum, index_corr)
    
    # 4) load BWM:
    convolution_matrix = np.load(path_to_bwm + 'band_window_matrix_sigma{:.2f}_'.format(para.sigma_int) + patch + '_nell{:}.npy'.format(para.ell_nodes))
    
    # 5) slice matrix according to z-correlation:
    # NEW ORDERING DUE TO REVISED FISHER MATRICES!!!
    slicing_points = [index_corr * band_offset, (index_corr + 1) * band_offset]
    
    # a little more complicated due to tomography...
    # I'll have to slice depending on which correlation we want...
    nbands = convolution_matrix.shape[0]
    #print nbands
    #print slicing_points
    
    nbands = range(slicing_points[0], slicing_points[1])
    #print nbands
    
    D_avg = np.zeros(len(nbands))
    #print D_avg
    
    for i, alpha in enumerate(nbands):
        # due to new dimensions of BWM including also cross-terms, we have to slice in ell-direction:
        # but the EE --> EE contribution is always the first ell-block!            
        # this should also still work with old BWMs...  
        index_ell_low = int(index_corr * len(ells_intp))
        index_ell_high = int((index_corr + 1) * len(ells_intp))
        
        w_alpha_ell = interpolate.interp1d(ells_intp, convolution_matrix[alpha, index_ell_low:index_ell_high], kind='linear')
        norm_val = np.sum(w_alpha_ell(ells_sum))
        print 'Norm of W_al = {:.2e}'.format(norm_val)        
        norm = 1. #norm_val        
        D_avg[i] = np.sum(w_alpha_ell(ells_sum) * D_ell) / norm
    
    return D_avg


# TODO: nband_types is hard-coded now...
def plot_band_window_functions(path_out, correlations, slicing_points_band_types, band_offset, ells_intp, band_window_matrix, patch, index_corr, band_type, nzcorrs, nband_types=2):
     
    # 2) determine l-range for interpolation
    nell = ells_intp[-1] - ells_intp[0]
    #print l_intp[-1], l_intp[0], nell
    ells_sum = np.linspace(ells_intp[0], ells_intp[-1] -1, nell).astype(int)
    #print len(l_sum)
    #print slicing_points_band_types
    #exit()
    # slice between EE and BB (only in bands along first axis):
    convolution_matrix = band_window_matrix[slicing_points_band_types[0]:slicing_points_band_types[-1], :]
    
    # index_corr comes from "outside" and slices along "band_axis"
    slicing_points_bands = [index_corr * band_offset, (index_corr + 1) * band_offset]        
    # slice now along the set of bands:    
    convolution_matrix = convolution_matrix[slicing_points_bands[0]:slicing_points_bands[-1], :]
    #print convolution_matrix.shape
    
    fig_bwm, axes = plt.subplots(band_offset + 1, nband_types * nzcorrs, sharex=True, figsize=(cm2inch(29.7), cm2inch(21.))) 
    fig_bwm_sum, axes2 = plt.subplots(1, nband_types * nzcorrs, sharex=True, figsize=(cm2inch(20.), 1./3.*cm2inch(20.))) 
    ax = axes    
    ax2 = axes2
    for index_band in xrange(band_offset):
        index_lin = 0
        for index_type in xrange(nband_types):
            for index_corr_ell in xrange(nzcorrs):
                if index_type > 0:
                    offset_ell = nzcorrs * len(ells_intp)
                    btype = 'BB'
                else:
                    offset_ell = 0
                    btype = 'EE'
                title = btype + ', ' + correlations[index_corr_ell]
                slicing_points_ells = [offset_ell + index_corr_ell * len(ells_intp), offset_ell + (index_corr_ell + 1) * len(ells_intp)]            
                index_ell_low = slicing_points_ells[0]
                index_ell_high = slicing_points_ells[1]
                #print index_ell_low, index_ell_high                
                w_alpha_ell = interpolate.interp1d(ells_intp, convolution_matrix[index_band, index_ell_low:index_ell_high], kind='linear')
                norm_val = np.sum(w_alpha_ell(ells_sum))
                norm = 1. #norm_val
                # Don't plot the first and the last two bands in the last summary panel, they're usually completely off-scale!        
                if index_band < band_offset - 2 and index_band > 0 and band_type == 'EE':        
                    ax[-1, index_lin].plot(ells_sum, w_alpha_ell(ells_sum) / norm)
                    ax2[index_lin].plot(ells_sum, w_alpha_ell(ells_sum) / norm)
                # We don't need to skip the first band for B-modes!
                if index_band < band_offset - 2 and band_type == 'BB':        
                    ax[-1, index_lin].plot(ells_sum, w_alpha_ell(ells_sum) / norm)
                    ax2[index_lin].plot(ells_sum, w_alpha_ell(ells_sum) / norm)
                
                # TODO: More flexible coding here...
                if index_band < band_offset - 2:
                    ax[index_band, index_lin].set_ylim([-0.05, 0.05])
                elif index_band == band_offset - 2:
                    ax[index_band, index_lin].set_ylim([-0.5, 0.5])
                elif index_band == band_offset - 1:
                    ax[index_band, index_lin].set_ylim([-5., 5.])
                    
                ax[index_band, index_lin].plot(ells_sum, w_alpha_ell(ells_sum) / norm, label=r'$N=$' + r'$' + '{:.2e}'.format(norm_val) + r'$')
                ax[index_band, index_lin].axhline(0., color='black', ls='-')
                ax2[index_lin].axhline(0., color='black', ls='-')
                ax[index_band, index_lin].scatter(ells_intp, convolution_matrix[index_band, index_ell_low:index_ell_high] / norm, s=10, marker='s')
                ax[index_band, index_lin].set_xscale('log')
                #ax[index_band, index_lin].legend(loc='upper right', fontsize=8)
                ax[index_band, index_lin].set_xticks([])
                ax[index_band, index_lin].set_xticklabels([])
                
                ax[0, index_lin].set_title(title, loc='center')                
                ax2[index_lin].set_title(title, loc='center')
                
                if index_lin != 0:
                    #ax[index_band, index_lin].set_yticks([])
                    ax[index_band, index_lin].set_yticklabels([])
                
                index_lin += 1
                
    for index_lin in xrange(nband_types * nzcorrs):
        ax[-1, index_lin].set_xlim([ells_intp[0], ells_intp[-1]])
        ax[-1, index_lin].set_xlabel(r'$\ell$')
        ax[-1, index_lin].set_xscale('log')
        ax[-1, index_lin].set_ylim([-0.05, 0.05])
        ax[-1, index_lin].axhline(0., color='black', ls='-')        
        ax2[index_lin].set_ylim([-0.05, 0.05])        
        ax2[index_lin].set_xscale('log')
        ax2[index_lin].set_xlabel(r'$\ell$')
        ax2[index_lin].set_xlim([ells_intp[0], ells_intp[-1]])
        
        if index_lin != 0:
            #ax[index_band, index_lin].set_yticks([])
            ax[-1, index_lin].set_yticklabels([])
            ax2[index_lin].set_yticklabels([])
            
    for index_band in xrange(band_offset + 1):
        #ax[index_band, 0].set_ylim([-0.05, 0.05])
        ax[index_band, 0].set_ylabel(r'$W(\ell)$')
    ax2[0].set_ylabel(r'$W(\ell)$')
    
    fig_bwm.tight_layout(w_pad=0., h_pad=0.)
    fig_bwm.subplots_adjust(wspace=0., hspace=0.)    
    
    fig_bwm_sum.tight_layout(w_pad=0., h_pad=0.)
    fig_bwm_sum.subplots_adjust(wspace=0., hspace=0.)        
    
    fname = 'band_window_functions_' + correlations[index_corr] + '_' + band_type + '_' + patch + '.'
    fname2 = 'band_window_functions_summary_' + correlations[index_corr] + '_' + band_type + '_' + patch + '.'
    for file_type in para.file_types:
        if file_type == 'png':
            transparent = True
        else:
            transparent = False
        fig_bwm.savefig(path_out + 'plots/' + fname + file_type, transparent=transparent, dpi=450)
        print 'Plot saved to:', path_out + 'plots/' + fname + file_type        
        fig_bwm_sum.savefig(path_out + 'plots/' + fname2 + file_type, transparent=transparent, dpi=450)
        print 'Plot saved to:', path_out + 'plots/' + fname2 + file_type        
        
    return


# new version with linearly plotted B-modes that don't contain any longer the \ell*(\ell+1)-scaling...
def plot_band_powers_for_paper(path_out, ells, band_avg1, band_avg2, band_mean, err_avg1, err_avg2, err_run_to_run, conv_theo_avg, bands_min, bands_max, correlations, correlation_labels, nzbins, band_type='EE'):    
    """
    Warning: bad coding, this function is tailored to two redshift bins only
    """
    
    colors = ['blue', 'orange', 'red']
    
    if band_type == 'BB':
        bands_to_plot = para.bands_to_plot[1:]
    else:
        bands_to_plot = para.bands_to_plot
    
    # set masked values in data and theory vector (over all z-correlations) to 0:
    index1 = np.where(np.asarray(bands_to_plot) == 1)[0]
    
    # "nice", ax does not support indexing, if plt.subplots creates only 1 subplot... 
    # therefore the "squeeze keyword must be set!   
    if len(correlations) > 1:
        fig, axes = plt.subplots(1, len(correlations), sharey=True, figsize=(cm2inch(20.), 1./3.*cm2inch(20.)), squeeze=False)
    else:
        fig, axes = plt.subplots(1, len(correlations), sharey=True, figsize=(cm2inch(8.4), 3./4.*cm2inch(8.4)), squeeze=False)
    # this is necessary because of "squeeze=False" which introduces one additional array-dimension...       
    ax = axes[0]    
    
    bmax = bands_max[0]
    bmin = bands_min[0]
    
    exclude_high = max(bmax[index1])
    exclude_low = min(bmin[index1])
    
    print 'Exclude: ', exclude_low, exclude_high
    
    # find auto-correlation indices:
    auto_correlation_indices = []
    correlation_indices = []
    index_lin = 0
    for index_z1 in xrange(nzbins):
        for index_z2 in xrange(index_z1 + 1):
            if index_z1 == index_z2:
                auto_correlation_indices += [index_lin]
            index_lin += 1
            correlation_indices += [index_zbin1]
    for index_corr, correlation in enumerate(correlations):
        print index_corr
        print band_avg1[index_corr]
        
        #fake point:
        #ax[index_corr].scatter(100., 1e-5, marker='None', s=0., label=correlation_label[index_corr])
        
        # hide bands that are not used in grey area, but generally show them!
        if (band_type == 'EE') or (band_type == 'BB' and not para.plot_B_modes_on_linear_scale): # or band_type == 'BB':
            y1 = [para.y_high, para.y_high]
            y2 = 0.
            norm = 1.
            ax[index_corr].fill_between([para.ell_low, exclude_low], y1, y2, interpolate=True, color='grey', alpha=0.3)
            ax[index_corr].fill_between([exclude_high, para.ell_high], y1, y2, interpolate=True, color='grey', alpha=0.3)
        elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:            
            # empirical value:
            scale = 1e+9 
            # here, we take out the \ell*(\ell+1)/2pi normalization of the band powers:
            norm = scale * 2. * np.pi / (ells[index_corr] * (ells[index_corr] + 1.))
            
            # for comparison with Edo:
            #scale = 1e+7
            #norm = scale * ells[index_corr] * 2. * np.pi / (ells[index_corr] * (ells[index_corr] + 1.))            
            
            upper_limit = para.y_high
            lower_limit = para.y_low            
            
            y1_ax0 = np.array([upper_limit, upper_limit])#*scale
            y2_ax0 = lower_limit#*scale
            
            ax[index_corr].fill_between([para.ell_low, exclude_low], y1_ax0, y2_ax0, interpolate=True, color='grey', alpha=0.3)
            ax[index_corr].fill_between([exclude_high, para.ell_high], y1_ax0, y2_ax0, interpolate=True, color='grey', alpha=0.3)
            
        wx = bands_max[index_corr]-bands_min[index_corr]
        if para.external_covariance and not para.fisher_errors:
            hy = 2. * err_avg1[index_corr]
            if band_type == 'EE': # or band_type == 'BB':
                bottom_value = np.abs(band_avg1[index_corr]) - err_avg1[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value * norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
            elif band_type == 'BB':
                #bottom_value = -err_avg1[index_corr] #np.zeros_like(band_avg1[index_corr])#band_avg1[index_corr]-err_avg1[index_corr]
                bottom_value = band_avg1[index_corr] - err_avg1[index_corr]                
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value * norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
        elif not para.external_covariance and para.fisher_errors:
            hy = 2. * err_avg2[index_corr]
            if band_type == 'EE': # or band_type == 'BB':
                bottom_value = np.abs(band_avg2[index_corr]) - err_avg2[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value * norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None) #r'$Fisher \ errors$')
            elif band_type == 'BB':
                #bottom_value = -err_avg2[index_corr]#band_avg2[index_corr]-err_avg2[index_corr]
                bottom_value = band_avg2[index_corr] - err_avg2[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value * norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None) #r'$Fisher \ errors$')
        elif para.external_covariance and para.fisher_errors:
            if band_type == 'EE': # or band_type == 'BB':
                bottom_value1 = np.abs(band_avg1[index_corr]) - err_avg1[index_corr]
                bottom_value2 = np.abs(band_avg2[index_corr]) - err_avg2[index_corr]
                hy = 2. * err_avg1[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value1 * norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
                hy = 2. * err_avg2[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value2 * norm, width=wx, align='edge', color='None', label=r'$Fisher \ errors$', linewidth=2, ls='dashed')
            elif band_type == 'BB':
                #bottom_value1 = -err_avg1[index_corr]#band_avg1[index_corr]-err_avg1[index_corr]
                #bottom_value2 = -err_avg2[index_corr]#band_avg2[index_corr]-err_avg2[index_corr]
                bottom_value1 = band_avg1[index_corr] - err_avg1[index_corr]
                bottom_value2 = band_avg2[index_corr] - err_avg2[index_corr]
                hy = 2. * err_avg1[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value1 * norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
                hy = 2. * err_avg2[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value2 * norm, width=wx, align='edge', color='None', label=r'$Fisher \ errors$', linewidth=2, ls='dashed')
        # always plotted as dashed line!
        if para.run_to_run_errors:
            hy = 2. * err_run_to_run[index_corr]
            if band_type == 'EE': # or band_type == 'BB':
                bottom_value = np.abs(band_mean[index_corr]) - err_run_to_run[index_corr]
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value * norm, width=wx, align='edge', color='None', label=r'$\mathrm{run-to-run \ errors}$', linewidth=2, ls='dashed')#label_ext_cov[para.type_ext_cov], alpha=0.5,)
            elif band_type == 'BB':
                #bottom_value = -err_avg1[index_corr] #np.zeros_like(band_avg1[index_corr])#band_avg1[index_corr]-err_avg1[index_corr]
                bottom_value = band_mean[index_corr] - err_run_to_run[index_corr]                
                ax[index_corr].bar(bands_min[index_corr], hy * norm, bottom=bottom_value * norm, width=wx, align='edge', color='None', label=r'$\mathrm{run-to-run \ errors}$', linewidth=2, ls='dashed')#label_ext_cov[para.type_ext_cov], alpha=0.5,)
        
        if para.external_covariance and not para.fisher_errors:
            if band_type == 'EE': # or band_type == 'BB':
                ax[index_corr].scatter(ells[index_corr], band_avg1[index_corr] * norm, s=para.sp, color='black', marker='o')
                mask = np.sign(band_avg1[index_corr]) < 0.
                ax[index_corr].scatter(ells[index_corr][mask], np.abs(band_avg1[index_corr][mask]) * norm, s=para.sp, edgecolor='black', facecolors='None', marker='o')
            else:
                ax[index_corr].scatter(ells[index_corr], band_avg1[index_corr] * norm, s=para.sp, color='black', marker='o')
        elif not para.external_covariance and para.fisher_errors:
            if band_type == 'EE': # or band_type == 'BB':
                ax[index_corr].scatter(ells[index_corr], band_avg2[index_corr] * norm, s=para.sp, color='black', marker='o')
                mask = np.sign(band_avg2[index_corr]) < 0.
                ax[index_corr].scatter(ells[index_corr][mask], np.abs(band_avg2[index_corr][mask]) * norm, s=para.sp, edgecolor='black', facecolors='None', marker='o')
            else:
                ax[index_corr].scatter(ells[index_corr], band_avg2[index_corr] * norm, s=para.sp, color='black', marker='o')
        elif para.external_covariance and para.fisher_errors:
            if band_type == 'EE': # or band_type == 'BB':
                ax[index_corr].scatter(ells[index_corr], band_avg1[index_corr] * norm, s=para.sp, color='black', marker='o')
                ax[index_corr].scatter(ells[index_corr], band_avg2[index_corr] * norm, s=para.sp, color='black', marker='^')
                mask = np.sign(band_avg1[index_corr]) < 0.
                ax[index_corr].scatter(ells[index_corr][mask], np.abs(band_avg1[index_corr][mask]) * norm, s=para.sp, edgecolor='black', facecolors='None', marker='o')
                mask = np.sign(band_avg2[index_corr]) < 0.
                ax[index_corr].scatter(ells[index_corr][mask], np.abs(band_avg2[index_corr][mask]) * norm, s=para.sp, edgecolor='black', facecolors='None', marker='^')
            else:
                ax[index_corr].scatter(ells[index_corr], band_avg1[index_corr] * norm, s=para.sp, color='black', marker='o')
                ax[index_corr].scatter(ells[index_corr], band_avg2[index_corr] * norm, s=para.sp, color='black', marker='^')
        
        if para.run_to_run_errors:
            if band_type == 'EE': # or band_type == 'BB':
                ax[index_corr].scatter(ells[index_corr], band_mean[index_corr] * norm, s=para.sp, color='black', marker='*')
                mask = np.sign(band_avg1[index_corr]) < 0.
                ax[index_corr].scatter(ells[index_corr][mask], np.abs(band_mean[index_corr][mask]) * norm, s=para.sp, edgecolor='black', facecolors='None', marker='*')
            else:
                ax[index_corr].scatter(ells[index_corr], band_mean[index_corr] * norm, s=para.sp, color='black', marker='*')
        
        # comparison with theoretical predictions:
        if (band_type == 'EE') or (band_type == 'BB' and not para.plot_B_modes_on_linear_scale):
            if para.plot_theory:
                for index_theo, path in enumerate(para.path_to_theory):
                    ell_theo, D_ell_lcdm = get_theoretical_power_spectrum(path, index_corr)
                    if index_theo == 0:
                        ax[index_corr].plot(ell_theo, D_ell_lcdm, color=para.color_theory[index_theo], ls=para.ls_theory[index_theo], lw=para.lw_theory[index_theo], label=para.label_theory[index_theo])
                    else:
                        ax[index_corr].plot(ell_theo, D_ell_lcdm, color=para.color_theory[index_theo], ls=para.ls_theory[index_theo], lw=para.lw_theory[index_theo])
            if para.plot_convolved_theory: 
                #print conv_theo_avg[index_corr], len(conv_theo_avg)
                #print ells, len(ells)
                mask = np.sign(conv_theo_avg[index_corr]) < 0.
                ax[index_corr].scatter(ells[index_corr], conv_theo_avg[index_corr], color='red', marker='o', s=30, lw=2, label=r'$\mathcal{B}_\alpha^{theo} = \sum_\ell W_{\alpha \ell} D_\ell^{theo}$')
                ax[index_corr].scatter(ells[index_corr][mask], np.abs(conv_theo_avg[index_corr][mask]), edgecolor='red', facecolors='None', marker='o', s=50, lw=2)
        elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:
            if para.plot_theory:
                for index_theo, path in enumerate(para.path_to_theory):
                    ell_theo, D_ell_lcdm = get_theoretical_power_spectrum(path, index_corr)
                    #print ell_theo
                    #print D_ell_lcdm
                    inv_norm = 2. * np.pi / (ell_theo * (ell_theo + 1))                
                    #print inv_norm
                    #print D_ell_lcdm*inv_norm
                    #exit()
                    ax[index_corr].plot(ell_theo, D_ell_lcdm * inv_norm * scale, color=para.color_theory[index_theo], ls=para.ls_theory[index_theo], lw=para.lw_theory[index_theo], label=para.label_theory[index_theo])
        
        if para.plot_noise_power and index_corr in auto_correlation_indices:
            index_lin = correlation_indices[index_corr]
            ells_noise, p_noise, shot_noise, n_eff = get_shot_noise(para.ell_low, para.ell_high, nzbins=nzbins) #get_shot_noise(min(ells[index_corr]), max(ells[index_corr]))
            print ells_noise.shape, p_noise.shape, shot_noise.shape
            #print index_corr, index_lin
            print shot_noise
            if nzbins > 1:
                print 'Shot noise = {:.3e}'.format(shot_noise[index_lin])
                if para.plot_B_modes_on_linear_scale and band_type == 'BB':
                    inv_norm = 2. * np.pi / (ells_noise * (ells_noise + 1.))
                    p_noise[index_lin] *= inv_norm * scale
                ax[index_corr].plot(ells_noise, p_noise[index_lin], ls='--', dashes=(2, 2), color='grey', label=r'$C^\mathrm{noise}$')
            else:
                print 'Shot noise = {:.3e}'.format(float(shot_noise))
                if para.plot_B_modes_on_linear_scale and band_type == 'BB':
                    inv_norm = 2. * np.pi / (ells_noise * (ells_noise + 1.))
                    p_noise *= inv_norm * scale
                ax[index_corr].plot(ells_noise, p_noise, ls='--', dashes=(2, 2), color='grey', label=r'$C^\mathrm{noise}$')
        
        if (band_type == 'EE') or (band_type == 'BB' and not para.plot_B_modes_on_linear_scale): # or band_type == 'BB':
            ax[index_corr].set_yscale('log', nonposy='clip')
            ax[index_corr].set_ylim([para.y_low, para.y_high])
            
            ax[index_corr].set_xlabel(r'$\ell$', fontsize=para.fontsize_label)
            ax[index_corr].set_xscale('log', nonposx='clip')
            ax[index_corr].set_xlim([para.ell_low, para.ell_high])
            #ax[index_corr].set_title(correlation_label[index_corr], loc='left')
            ax[index_corr].text(.5, .6, correlation_labels[index_corr], horizontalalignment='center', transform=ax[index_corr].transAxes)
            ax[index_corr].tick_params(axis='both', labelsize=para.fontsize_ticks)
        
        elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:
            ax[index_corr].text(.54,.1, correlation_labels[index_corr], horizontalalignment='center', transform=ax[index_corr].transAxes)
            ax[index_corr].set_xlabel(r'$\ell$', fontsize=para.fontsize_label)
            ax[index_corr].axhline(0., ls='--', color='black')
            ax[index_corr].set_ylim([y2_ax0, y1_ax0[0]])
            ax[index_corr].set_xlim([para.ell_low, para.ell_high])
            ax[index_corr].tick_params(axis='both', labelsize=para.fontsize_ticks)
            ax[index_corr].set_xscale('log')#, nonposx='clip')            
    
    if band_type == 'EE': # or band_type == 'BB':
        ax[0].set_ylabel(r'$\ell(\ell+1)C^{\mathrm{EE}}(\ell)/2\mathrm{\pi}$', fontsize=para.fontsize_label)
        ax[0].legend(loc='upper center', fontsize=para.fontsize_legend, frameon=False)#, title=correlation_label[index_corr])
    elif band_type == 'BB' and not para.plot_B_modes_on_linear_scale:
        ax[0].set_ylabel(r'$\ell(\ell+1)C^{\mathrm{BB}}(\ell)/2\mathrm{\pi}$', fontsize=para.fontsize_label)
        ax[0].legend(loc='upper center', fontsize=para.fontsize_legend, frameon=False)
    elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:
        # TODO: Make sure that number in brackets corresponds to value of variable "scale"
        ax[0].set_ylabel(r'$C^{\mathrm{BB}}(\ell)$' + r'$ \ (\times 10^{:})$'.format(int(np.log10(scale))), fontsize=para.fontsize_label)
        ax[0].legend(loc='best', fontsize=para.fontsize_legend, frameon=False)
    fig.tight_layout(w_pad=0., h_pad=0.)
    fig.subplots_adjust(wspace=0., hspace=0.)
    #fig.tight_layout(w_pad=0., h_pad=0.)
    
    for filetype in para.file_types:
        if len(para.patches) > 1:
            fname = path_out+'/plots/all_signals_'+band_type+'.'+filetype
        else:
            fname = path_out+'/plots/all_signals_'+band_type+'_'+para.patches[0]+'.'+filetype
        if filetype == 'png' and para.transparent_background:
            fig.savefig(fname, dpi=450, transparent=True)
        else:
            fig.savefig(fname, dpi=450)
        print 'Plot saved to: \n', fname
    
    if para.show_plots == True and para.plot_band_window_functions == False:
        plt.show()
    
    return
 

# new version with linearly plotted B-modes that don't contain any longer the \ell*(\ell+1)-scaling...
def plot_band_powers_for_paper_triangle(path_out, ells, band_avg1, band_avg2, band_mean, err_avg1, err_avg2, err_run_to_run, conv_theo_avg, bands_min, bands_max, nzbins, correlation_labels, band_type='EE'):    
    """
    Warning: bad coding, this function is tailored to two redshift bins only
    """
    
    colors = []
    #for index_corr in xrange(len(correlation_labels)): 
    #    colors.append('blue')
    
    for index_zbin1 in xrange(nzbins):
        for index_zbin2 in xrange(index_zbin1 + 1):
            if index_zbin1 == 0 and index_zbin2 == 0:
                colors.append('blue')
            elif index_zbin1 == 1 and index_zbin2 == 1:
                colors.append('orange')
            elif index_zbin1 == 2 and index_zbin2 == 2:
                colors.append('red')
            else:
                colors.append('grey')
    
    if band_type == 'BB':
        bands_to_plot = para.bands_to_plot[1:]
    else:
        bands_to_plot = para.bands_to_plot
    
    # set masked values in data and theory vector (over all z-correlations) to 0:
    index1 = np.where(np.asarray(bands_to_plot) == 1)[0]
     
    # From corner.py!
    # Some magic numbers for pretty axis layout.
    K = nzbins
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim    
    
    fig, ax = plt.subplots(nzbins, nzbins, figsize=(dim, dim)) #, sharey=True) # sharey=True, figsize=(cm2inch(20.), 1./3.*cm2inch(20.))) #, squeeze=False)    
    # this is necessary because of "squeeze=False" which introduces one additional array-dimension...    
    #ax = axes[0]    
    
    # From corner.py!
    # Format the figure.
    '''
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace)    
    '''
    
    bmax = bands_max[0]
    bmin = bands_min[0]
    
    exclude_high = max(bmax[index1])
    exclude_low = min(bmin[index1])
    
    print 'Exclude: ', exclude_low, exclude_high
    
    index_corr = 0
    for index_zbin1 in xrange(nzbins):
        for index_zbin2 in xrange(index_zbin1 + 1):
            
            print index_corr
            print band_avg1[index_corr]
        
            # hide bands that are not used in grey area, but generally show them!
            if (band_type == 'EE') or (band_type == 'BB' and not para.plot_B_modes_on_linear_scale): # or band_type == 'BB':
                y1 = [para.y_high, para.y_high]
                y2 = 0.
                norm = 1.
                ax[index_zbin1, index_zbin2].fill_between([para.ell_low, exclude_low], y1, y2, interpolate=True, color='grey', alpha=0.3)
                ax[index_zbin1, index_zbin2].fill_between([exclude_high, para.ell_high], y1, y2, interpolate=True, color='grey', alpha=0.3)
            elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:
                # empirical value:
                scale = 1e+9 
                # here, we take out the \ell*(\ell+1)/2pi normalization of the band powers:
                norm = scale * 2. * np.pi / (ells[index_corr] * (ells[index_corr] + 1.))
            
                upper_limit = para.y_high
                lower_limit = para.y_low            
            
                y1_ax0 = np.array([upper_limit, upper_limit])#*scale
                y2_ax0 = lower_limit#*scale
            
                ax[index_zbin1, index_zbin2].fill_between([para.ell_low, exclude_low], y1_ax0, y2_ax0, interpolate=True, color='grey', alpha=0.3)
                ax[index_zbin1, index_zbin2].fill_between([exclude_high, para.ell_high], y1_ax0, y2_ax0, interpolate=True, color='grey', alpha=0.3)
        
            wx = bands_max[index_corr]-bands_min[index_corr]
            if para.external_covariance and not para.fisher_errors:
                hy = 2. * err_avg1[index_corr]
                if band_type == 'EE': # or band_type == 'BB':
                    bottom_value = np.abs(band_avg1[index_corr])-err_avg1[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value*norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
                elif band_type == 'BB':
                    #bottom_value = -err_avg1[index_corr] #np.zeros_like(band_avg1[index_corr])#band_avg1[index_corr]-err_avg1[index_corr]
                    bottom_value = band_avg1[index_corr]-err_avg1[index_corr]                
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value*norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
            elif not para.external_covariance and para.fisher_errors:
                hy = 2. * err_avg2[index_corr]
                if band_type == 'EE': # or band_type == 'BB':
                    bottom_value = np.abs(band_avg2[index_corr])-err_avg2[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value*norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None) #r'$Fisher \ errors$')
                elif band_type == 'BB':
                    #bottom_value = -err_avg2[index_corr]#band_avg2[index_corr]-err_avg2[index_corr]
                    bottom_value = band_avg2[index_corr]-err_avg2[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value*norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None) #r'$Fisher \ errors$')
            elif para.external_covariance and para.fisher_errors:
                if band_type == 'EE': # or band_type == 'BB':
                    bottom_value1 = np.abs(band_avg1[index_corr])-err_avg1[index_corr]
                    bottom_value2 = np.abs(band_avg2[index_corr])-err_avg2[index_corr]
                    hy = 2. * err_avg1[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value1*norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
                    hy = 2. * err_avg2[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value2*norm, width=wx, align='edge', color='None', label=r'$Fisher \ errors$', linewidth=2, ls='dashed')
                elif band_type == 'BB':
                    #bottom_value1 = -err_avg1[index_corr]#band_avg1[index_corr]-err_avg1[index_corr]
                    #bottom_value2 = -err_avg2[index_corr]#band_avg2[index_corr]-err_avg2[index_corr]
                    bottom_value1 = band_avg1[index_corr]-err_avg1[index_corr]
                    bottom_value2 = band_avg2[index_corr]-err_avg2[index_corr]
                    hy = 2. * err_avg1[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value1*norm, width=wx, align='edge', facecolor=colors[index_corr], edgecolor='None', alpha=0.5, label=None)#label_ext_cov[para.type_ext_cov], alpha=0.5,)
                    hy = 2. * err_avg2[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value2*norm, width=wx, align='edge', color='None', label=r'$Fisher \ errors$', linewidth=2, ls='dashed')
            # always plotted with dashed line:
            if para.run_to_run_errors:
                hy = 2. * err_run_to_run[index_corr]
                if band_type == 'EE': # or band_type == 'BB':
                    bottom_value = np.abs(band_mean[index_corr])-err_run_to_run[index_corr]
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value*norm, width=wx, align='edge', color='None', label=r'$\mathrm{run-to-run \ errors}$', linewidth=2, ls='dashed')#label_ext_cov[para.type_ext_cov], alpha=0.5,)
                elif band_type == 'BB':
                    #bottom_value = -err_avg1[index_corr] #np.zeros_like(band_avg1[index_corr])#band_avg1[index_corr]-err_avg1[index_corr]
                    bottom_value = band_mean[index_corr]-err_run_to_run[index_corr]                
                    ax[index_zbin1, index_zbin2].bar(bands_min[index_corr], hy*norm, bottom=bottom_value*norm, width=wx, align='edge', color='None', label=r'$\mathrm{run-to-run \ errors}$', linewidth=2, ls='dashed')#label_ext_cov[para.type_ext_cov], alpha=0.5,)
                
            if para.external_covariance and not para.fisher_errors:
                if band_type == 'EE': # or band_type == 'BB':
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg1[index_corr]*norm, s=para.sp, color='black', marker='o')
                    mask = np.sign(band_avg1[index_corr]) < 0.
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr][mask], np.abs(band_avg1[index_corr][mask])*norm, s=para.sp, edgecolor='black', facecolors='None', marker='o')
                else:
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg1[index_corr]*norm, s=para.sp, color='black', marker='o')
            elif not para.external_covariance and para.fisher_errors:
                if band_type == 'EE': # or band_type == 'BB':
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg2[index_corr]*norm, s=para.sp, color='black', marker='o')
                    mask = np.sign(band_avg2[index_corr]) < 0.
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr][mask], np.abs(band_avg2[index_corr][mask])*norm, s=para.sp, edgecolor='black', facecolors='None', marker='o')
                else:
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg2[index_corr]*norm, s=para.sp, color='black', marker='o')
            elif para.external_covariance and para.fisher_errors:
                if band_type == 'EE': # or band_type == 'BB':
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg1[index_corr]*norm, s=para.sp, color='black', marker='o')
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg2[index_corr]*norm, s=para.sp, color='black', marker='^')
                    mask = np.sign(band_avg1[index_corr]) < 0.
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr][mask], np.abs(band_avg1[index_corr][mask])*norm, s=para.sp, edgecolor='black', facecolors='None', marker='o')
                    mask = np.sign(band_avg2[index_corr]) < 0.
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr][mask], np.abs(band_avg2[index_corr][mask])*norm, s=para.sp, edgecolor='black', facecolors='None', marker='^')
                else:
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg1[index_corr]*norm, s=para.sp, color='black', marker='o')
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_avg2[index_corr]*norm, s=para.sp, color='black', marker='^')
            
            if para.run_to_run_errors:
                if band_type == 'EE': # or band_type == 'BB':
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_mean[index_corr]*norm, s=para.sp, color='black', marker='*')
                    mask = np.sign(band_avg1[index_corr]) < 0.
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr][mask], np.abs(band_mean[index_corr][mask])*norm, s=para.sp, edgecolor='black', facecolors='None', marker='*')
                else:
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], band_mean[index_corr]*norm, s=para.sp, color='black', marker='*')
            
            # comparison with theoretical predictions:
            if (band_type == 'EE') or (band_type == 'BB' and not para.plot_B_modes_on_linear_scale):
                if para.plot_theory:
                    for index_theo, path in enumerate(para.path_to_theory):
                        ell_theo, D_ell_lcdm = get_theoretical_power_spectrum(path, index_corr)
                        ax[index_zbin1, index_zbin2].plot(ell_theo, D_ell_lcdm, color=para.color_theory[index_theo], ls=para.ls_theory[index_theo], lw=para.lw_theory[index_theo], label=para.label_theory[index_theo])
                if para.plot_convolved_theory:            
                    mask = np.sign(conv_theo_avg[index_corr]) < 0.
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr], conv_theo_avg[index_corr], color='red', marker='o', s=30, lw=2, label=r'$\mathcal{B}_\alpha^{theo} = \sum_\ell W_{\alpha \ell} D_\ell^{theo}$')
                    ax[index_zbin1, index_zbin2].scatter(ells[index_corr][mask], np.abs(conv_theo_avg[index_corr][mask]), edgecolor='red', facecolors='None', marker='o', s=50, lw=2)
            elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:
                if para.plot_theory:
                    for index_theo, path in enumerate(para.path_to_theory):
                        ell_theo, D_ell_lcdm = get_theoretical_power_spectrum(path, index_corr)
                        inv_norm = 2. * np.pi / (ell_theo * (ell_theo + 1))                
                        ax[index_zbin1, index_zbin2].plot(ell_theo, D_ell_lcdm * inv_norm * scale, color=para.color_theory[index_theo], ls=para.ls_theory[index_theo], lw=para.lw_theory[index_theo], label=para.label_theory[index_theo])
        
            if (band_type == 'EE') or (band_type == 'BB' and not para.plot_B_modes_on_linear_scale): # or band_type == 'BB':
                ax[index_zbin1, index_zbin2].set_yscale('log', nonposy='clip')
                ax[index_zbin1, index_zbin2].set_ylim([para.y_low, para.y_high])
            
                ax[index_zbin1, index_zbin2].set_xlabel(r'$\ell$', fontsize=para.fontsize_label)
                ax[index_zbin1, index_zbin2].set_xscale('log', nonposx='clip')
                ax[index_zbin1, index_zbin2].set_xlim([para.ell_low, para.ell_high])
                #ax[index_corr].set_title(correlation_label[index_corr], loc='left')
                ax[index_zbin1, index_zbin2].text(.54, .7, correlation_labels[index_corr], horizontalalignment='center', transform=ax[index_zbin1, index_zbin2].transAxes)
                ax[index_zbin1, index_zbin2].tick_params(axis='both', labelsize=para.fontsize_ticks)
        
            elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:
                ax[index_zbin1, index_zbin2].text(.54,.1, correlation_labels[index_corr], horizontalalignment='center', transform=ax[index_zbin1, index_zbin2].transAxes)
                ax[index_zbin1, index_zbin2].set_xlabel(r'$\ell$', fontsize=para.fontsize_label)
                ax[index_zbin1, index_zbin2].axhline(0., ls='--', color='black')
                ax[index_zbin1, index_zbin2].set_ylim([y2_ax0, y1_ax0[0]])
                ax[index_zbin1, index_zbin2].set_xlim([para.ell_low, para.ell_high])
                ax[index_zbin1, index_zbin2].tick_params(axis='both', labelsize=para.fontsize_ticks)
                ax[index_zbin1, index_zbin2].set_xscale('log')#, nonposx='clip')
            
            if para.plot_noise_power and index_zbin1 == index_zbin2:
                ells_noise, p_noise, shot_noise, n_eff = get_shot_noise(para.ell_low, para.ell_high, nzbins=nzbins)
                if para.plot_B_modes_on_linear_scale and band_type == 'BB':
                    inv_norm = 2. * np.pi / (ells_noise * (ells_noise + 1.))
                    p_noise *= inv_norm * scale
                print ells_noise.shape, p_noise.shape, shot_noise.shape, n_eff.shape
                print index_corr
                print 'Shot noise = {:.3e}'.format(shot_noise[index_zbin1])
                ax[index_zbin1, index_zbin2].plot(ells_noise, p_noise[index_zbin1], color='grey', ls='--', dashes=(2, 2), label=r'$C^\mathrm{noise}$')
                
            # don't plot the upper, redundant triangle!
            if index_zbin2 != index_zbin1:
                ax[index_zbin2, index_zbin1].set_frame_on(False)
                ax[index_zbin2, index_zbin1].set_xticks([])
                ax[index_zbin2, index_zbin1].set_yticks([])
            
            # only first column should have y-tick labels, so set others to "empty" 
            if index_zbin2 != 0:
                ax[index_zbin1, index_zbin2].set_yticklabels([])
            
            # only lowest row should have xticklabels:
            if index_zbin1 != nzbins - 1:
                ax[index_zbin1, index_zbin2].set_xticklabels([])
            
            index_corr += 1
    
    if band_type == 'EE': # or band_type == 'BB':
        for index_zbin in xrange(nzbins):        
            ax[index_zbin, 0].set_ylabel(r'$\ell(\ell+1)C^\mathrm{EE}(\ell)/2\mathrm{\pi}$', fontsize=para.fontsize_label)
        ax[1, 1].legend(loc=(1, 1), fontsize=para.fontsize_legend, frameon=False)
    elif band_type == 'BB' and not para.plot_B_modes_on_linear_scale:
        for index_zbin in xrange(nzbins):        
            ax[index_zbin, 0].set_ylabel(r'$\ell(\ell+1)C^\mathrm{BB}(\ell)/2\mathrm{\pi}$', fontsize=para.fontsize_label)
        ax[0, 0].legend(loc='upper center', fontsize=para.fontsize_legend, frameon=False)
    elif band_type == 'BB' and para.plot_B_modes_on_linear_scale:
        # TODO: Make sure that number in brackets corresponds to value of variable "scale"
        for index_zbin in xrange(nzbins):        
            ax[index_zbin, 0].set_ylabel(r'$C^\mathrm{BB}(\ell)$' + r'$ \ (\times 10^{:})$'.format(int(np.log10(scale))), fontsize=para.fontsize_label)
        ax[0, 0].legend(loc='best', fontsize=para.fontsize_legend, frameon=False)
    fig.tight_layout(w_pad=0., h_pad=0.)
    fig.subplots_adjust(wspace=0., hspace=0.)
    #fig.tight_layout(w_pad=0., h_pad=0.)

    #TODO: Here's the place to mess around to get rid of overlapping ytick labels...
    # this prevents overlapping of xtick and ytick labels  
    # index is manual guess-work...
    try:    
        for index_zbin in xrange(nzbins):
            print ax[index_zbin, 0].get_yticklabels()[para.prune_index_y]
            ax[index_zbin, 0].get_yticklabels()[para.prune_index_y].set_visible(False)
    except:
        pass
    
    try:
        for index_zbin in xrange(nzbins):
            print ax[index_zbin, 0].get_yticklabels()[para.prune_index_x]
            ax[index_zbin, 0].get_yticklabels()[para.prune_index_x].set_visible(False)
    except:
        pass
    
    for filetype in para.file_types:
        if len(para.patches) > 1:
            fname = path_out + '/plots/all_signals_' + band_type + '.' + filetype
        else:
            fname = path_out + '/plots/all_signals_' + band_type + '_' + para.patches[0] + '.' + filetype
        if filetype == 'png' and para.transparent_background:
            fig.savefig(fname, dpi=450, transparent=True)
        else:
            fig.savefig(fname, dpi=450)
        print 'Plot saved to: \n', fname
    
    if para.show_plots == True and para.plot_band_window_functions == False:
        plt.show()
    
    return
       
       
if __name__ == '__main__':
    
    plt.style.use('classic')
    
    # read in parameter file:
    filename_params = sys.argv[1]
    
    get_input_parameters(filename_params)    
      
    z_min = min(para.z_min)
    z_max = max(para.z_max)    
    
    redshift_bins = []
    for index_zbin in xrange(len(para.z_min)):
        redshift_bin = '{:.2f}z{:.2f}'.format(para.z_min[index_zbin], para.z_max[index_zbin])
        redshift_bins.append(redshift_bin)
    # number of z-bins
    nzbins = len(redshift_bins)
    # number of *unique* correlations between z-bins
    nzcorrs = nzbins * (nzbins + 1) / 2      
    
    if para.apply_multiplicative_bias_calibration:
        if 'EE' in para.band_types:
            nbands = len(para.bands_to_plot)
        elif 'BB' in para.band_types:
            nbands = len(para.bands_to_plot) - 1
        elif 'EE' in para.band_types and 'BB' in para.band_types:
            print 'Script must be run for either EE or BB!'
            exit()
        else:
            print 'Script must be run for either EE or BB!'
            exit()
        fname = para.file_for_multiplicative_bias_correction
        if nzbins == 1:
            m_corr_per_zbin = np.asarray([np.loadtxt(fname, usecols=[1])])
        else:
            m_corr_per_zbin = np.loadtxt(fname, usecols=[1])
        
        #index_corr = 0
        m_corr = []
        for index_zbin1 in xrange(nzbins):
            for index_zbin2 in xrange(index_zbin1 + 1):
                val_m_corr = (1. + m_corr_per_zbin[index_zbin1]) * (1. + m_corr_per_zbin[index_zbin2]) * np.ones(nbands)
                m_corr.append(val_m_corr)
                '''
                if index_corr == 0:
                    m_corr = val_m_corr
                else:
                    m_corr = np.concatenate((m_corr, val_m_corr))
                        
                index_corr += 1
                '''    
                       
    # create a map of indices, mapping from unique correlation index to redshift bin indices
    index_map = []
    correlations = []
    correlation_labels = []    
    for index_zbin1 in xrange(nzbins):
        for index_zbin2 in xrange(index_zbin1+1):
            index_map.append((index_zbin1, index_zbin2))    
            correlations.append('z{:}xz{:}'.format(index_zbin1 + 1, index_zbin2 + 1))    
            correlation_labels.append(r'$z_{:} \times \, z_{:} $'.format(index_zbin1 + 1, index_zbin2 + 1))
    #print correlation_labels            
    #print index_map    
    
    '''    
    if nzbins > 1:
        root_path = para.root_in+'/sigma_int{:.2f}/tomographic/{:.2f}z{:.2f}/'.format(para.sigma_int, z_min, z_max)
    else:
        root_path = para.root_in+'/sigma_int{:.2f}/{:.2f}z{:.2f}/'.format(para.sigma_int, z_min, z_max)
    '''
    
    # new, more logical folder structure:
    root_path = para.root_in + '/sigma_int{:.2f}/{:.2f}z{:.2f}/{:}zbins/'.format(para.sigma_int, z_min, z_max, nzbins)
    
    if len(para.patches) == 1:
        path_out = root_path + para.patches[0] + '/'
    # ordinary case for combination of W1W2W3W4
    elif len(para.patches) == 4:
        foldername = ''
        for patch in para.patches: 
            foldername += patch + '_'
        path_out = root_path + foldername[:-1] + '/'
    # this is for combination of more than 4 fields (e.g. clones or GRFs)
    else:
        foldername = para.patches[0] + '_to_' + para.patches[-1]
        path_out = root_path+foldername + '/'
    
    if not os.path.isdir(path_out):
        os.makedirs(path_out)
        os.makedirs(path_out + 'plots/')
        #os.makedirs(path_out+'control_outputs/')
   
    print 'Outputs will be saved in: ', path_out
    
    collect_all_bp_avg1_EE = []
    collect_all_bp_avg2_EE = []
    collect_all_bp_mean_EE = []
    collect_all_err_avg1_EE = []
    collect_all_err_avg2_EE = []
    collect_all_err_run_to_run_EE = []
    collect_conv_theo_avg = []
    collect_all_bands_min_EE = []
    collect_all_bands_max_EE = []
    collect_all_ells_EE = []
    
    collect_all_bp_avg1_BB = []
    collect_all_bp_avg2_BB = []
    collect_all_bp_mean_BB = []    
    collect_all_err_avg1_BB = []
    collect_all_err_avg2_BB = []
    collect_all_err_run_to_run_BB = []    
    collect_all_bands_min_BB = []
    collect_all_bands_max_BB = []
    collect_all_ells_BB = []
                        
    for index_corr, correlation in enumerate(correlations):
        for band_type in para.band_types:
            # this functions makes sure that we are using the correct redshift bin etc...
            # DEPRECATED:            
            #set_zbins(correlation)
            # these three functions collect and analyse all outputs from quadratic_estimator.py
            # and get_outputs will also plot the correlation matrices...
            slicing_points, band_offset, ells, bands_min, bands_max = get_slicing_points_and_offset(root_path, band_type, nzcorrs)
            #print slicing_points
            #exit()
            band_powers, inv_variances, variances_ext, conv_theo, ell_intp_bwm, collect_bwm, weights_ext = collect_outputs(root_path, path_out, band_type, correlation, slicing_points, band_offset, index_corr, nzcorrs)
            band_avg1, band_avg2, band_mean, err_avg1, err_avg2, err_run_to_run, conv_theo_avg = get_averages_and_errors(band_powers, inv_variances, variances_ext, conv_theo, band_type, weights_ext)
            
            if para.apply_multiplicative_bias_calibration:
                band_mean = band_mean / m_corr[index_corr]
                band_avg1 = band_avg1 / m_corr[index_corr]
                band_avg2 = band_avg2 / m_corr[index_corr]
                # Shouldn't the errors also be scaled with m_corr?
                err_avg1 = err_avg1 / m_corr[index_corr]
                err_avg2 = err_avg2 / m_corr[index_corr]
                err_run_to_run = err_run_to_run / m_corr[index_corr]
                
            # correct spectra for excess noise:
            if para.correct_for_excess_noise:
                fname = para.path_to_excess_noise + '{:}zbins/all_estimates_band_powers_excess_noise_'.format(nzbins) + band_type + '_' + correlation + '.dat' 
                excess_noise = np.loadtxt(fname)
                fname = para.path_to_excess_noise + '{:}zbins/errors_excess_noise_'.format(nzbins) + band_type + '_' + correlation + '.dat' 
                err_excess_noise = np.loadtxt(fname)
                # signs for excess noise are derived for THEORY spectrum, so signs need to be flipped (i.e. subtracted)!
                band_avg1 -= excess_noise
                band_avg2 -= excess_noise
                #print band_mean, excess_noise
                band_mean -= excess_noise
                #print band_mean
                if para.propagate_error_excess_noise:
                    err_avg1 = np.sqrt(err_avg1**2 + err_excess_noise**2)
                    err_avg2 = np.sqrt(err_avg2**2 + err_excess_noise**2)
                    err_run_to_run = np.sqrt(err_run_to_run**2 + err_excess_noise**2)
                
            # collect some stuff here for paper-plot:
            if band_type == 'EE':
                collect_all_bp_avg1_EE.append(band_avg1)
                collect_all_bp_avg2_EE.append(band_avg2)
                collect_all_bp_mean_EE.append(band_mean)
                collect_all_err_avg1_EE.append(err_avg1)
                collect_all_err_avg2_EE.append(err_avg2)
                collect_all_err_run_to_run_EE.append(err_run_to_run)
                collect_all_bands_min_EE.append(bands_min)
                collect_all_bands_max_EE.append(bands_max)
                collect_conv_theo_avg.append(conv_theo_avg)
                collect_all_ells_EE.append(ells)
                #print collect_all_bp_avg1_EE.append(band_avg1)
            elif band_type == 'BB':   
                if para.subtract_fiducial_B_modes:
                    #TODO: filename here is hardcoded!
                    fname = para.path_to_fiducial_B_modes + '{:}zbins/all_estimates_band_powers_fiducial_BB_'.format(nzbins) + correlation + '.dat' 
                    fiducial_B_modes = np.loadtxt(fname)
                    fname = para.path_to_fiducial_B_modes + '{:}zbins/errors_fiducial_BB_'.format(nzbins) + correlation + '.dat' 
                    err_fiducial_B_modes = np.loadtxt(fname)
                    band_avg1 -= fiducial_B_modes
                    band_avg2 -= fiducial_B_modes
                    #print band_mean, fiducial_B_modes
                    band_mean -= fiducial_B_modes
                    #print band_mean
                    if para.propagate_error_resetting_bias:
                        err_avg1 = np.sqrt(err_avg1**2 + err_fiducial_B_modes**2)
                        err_avg2 = np.sqrt(err_avg2**2 + err_fiducial_B_modes**2)
                        err_run_to_run = np.sqrt(err_run_to_run**2 + err_fiducial_B_modes**2)
                    
                collect_all_bp_avg1_BB.append(band_avg1)
                collect_all_bp_avg2_BB.append(band_avg2)
                collect_all_bp_mean_BB.append(band_mean)
                collect_all_err_avg1_BB.append(err_avg1)
                collect_all_err_avg2_BB.append(err_avg2)
                collect_all_err_run_to_run_BB.append(err_run_to_run)                
                collect_all_bands_min_BB.append(bands_min)
                collect_all_bands_max_BB.append(bands_max)
                # only needed for consistency...
                collect_conv_theo_avg.append(conv_theo_avg)
                collect_all_ells_BB.append(ells)
                
            # save some plotting data:
            # I should also save bands_min and bands_max for box plots... from future import...
            fname = path_out + 'data_for_plots_sigma{:.2f}_'.format(para.sigma_int) + '{:}fields_'.format(len(para.patches)) + band_type + '_' + correlation + '.dat'
            header = 'ells, ells_min, ells_max, band_avg1, band_avg2, band_mean, err_avg1, err_avg2, std / sqrt_N'
            savedata = np.column_stack((ells, bands_min, bands_max, band_avg1, band_avg2, band_mean, err_avg1, err_avg2, err_run_to_run))
            np.savetxt(fname, savedata, header=header)
    
            # now we produce the main plot(s):
            # collect everything here and make option to plot everything into one figure...
            if para.make_single_plots:
                plot_band_powers_single(root_path, path_out, band_type, ells, band_avg1, band_avg2, band_mean, bands_min, bands_max, err_avg1, err_avg2, err_run_to_run, conv_theo_avg, correlation, index_corr)
            if para.plot_band_window_functions:
                for i, patch in enumerate(para.patches):
                    plot_band_window_functions(path_out, correlations, slicing_points, band_offset, ell_intp_bwm, collect_bwm[i], patch, index_corr, band_type, nzcorrs, nband_types=2)
    
    if para.make_plots_for_paper:
        if 'EE' in para.band_types:
            if nzcorrs <= 3:
                plot_band_powers_for_paper(path_out, collect_all_ells_EE, collect_all_bp_avg1_EE, collect_all_bp_avg2_EE, collect_all_bp_mean_EE, collect_all_err_avg1_EE, collect_all_err_avg2_EE, collect_all_err_run_to_run_EE, collect_conv_theo_avg, collect_all_bands_min_EE, collect_all_bands_max_EE, correlations, correlation_labels, nzbins, band_type='EE')
            else:
                plot_band_powers_for_paper_triangle(path_out, collect_all_ells_EE, collect_all_bp_avg1_EE, collect_all_bp_avg2_EE, collect_all_bp_mean_EE, collect_all_err_avg1_EE, collect_all_err_avg2_EE, collect_all_err_run_to_run_EE, collect_conv_theo_avg, collect_all_bands_min_EE, collect_all_bands_max_EE, nzbins, correlation_labels, band_type='EE')
        if 'BB' in para.band_types:
            if nzcorrs <= 3:
                plot_band_powers_for_paper(path_out, collect_all_ells_BB, collect_all_bp_avg1_BB, collect_all_bp_avg2_BB, collect_all_bp_mean_BB, collect_all_err_avg1_BB, collect_all_err_avg2_BB, collect_all_err_run_to_run_BB, collect_conv_theo_avg, collect_all_bands_min_BB, collect_all_bands_max_BB, correlations, correlation_labels, nzbins, band_type='BB')
            else:
                plot_band_powers_for_paper_triangle(path_out, collect_all_ells_BB, collect_all_bp_avg1_BB, collect_all_bp_avg2_BB, collect_all_bp_mean_BB, collect_all_err_avg1_BB, collect_all_err_avg2_BB, collect_all_err_run_to_run_BB, collect_conv_theo_avg, collect_all_bands_min_BB, collect_all_bands_max_BB, nzbins, correlation_labels, band_type='BB')
    print 'Done.'
