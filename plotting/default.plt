#!/usr/bin/env python
# encoding: UTF8

# Always use Python notation for lists, strings etc.!!!
# Create your own .plt file with adjusted parameters following the notation and variable names as given here.

# output parameters:

# root path to data (i.e. code is expecting to find subfolders containing all the results from one patch, e.g. 'W1':
root_in = '/your/path/to/the/output/'

# pass always as a list!
# all:
patches = ['W1', 'W2', 'W3', 'W4']
# single:
#patches = ['W1']
# you can also supply a loop (e.g. very useful for mocks!):
#patches = ['Mock{:}_suffix'.format(i + 1) for i in range(100)]

# either 'EE', 'BB' or 'EB'; pass always as a list!
band_types = ['EE']
#band_types = ['BB']
#band_types = ['EB']

# if BB/EB-modes should be plotted on linear scale:
plot_on_linear_scale = True
# if plotting on linear scale, you might want to supply a sclae to be divided out:
scale = 1e+9
 
# if fiducial B-modes should be subtracted off the signal:
subtract_fiducial_B_modes = False
# set to "True" if errors should be propagated into plot:
propagate_error_resetting_bias = False
# supply path to fiducial B-mode file:
# you need to look up file-format and name in code...
path_to_fiducial_B_modes = '/your/path/to/those/files/'

# set to "True" if multiplicative calibration should be applied to signal in plot:
# this is needed if your catalogue does not contain any multiplicative bias correction values per component!
apply_multiplicative_bias_calibration = False
# if set to "True" supply here file (including path):
# you need to look up file-format in code...
file_for_multiplicative_bias_correction = '/your/file/to/m_correction_avg.txt'

# if the signals should be corrected for excess-noise power:
correct_for_excess_noise = False
# set to "True" if errors should be propagated into plot:
propagate_error_excess_noise = False
# supply path to excess-noise file:
# you need to look up file-format and name in code...
path_to_excess_noise = '/your/path/to/those/files/'

# bands to plot per z-correlation and band type (for BB first element is automatically removed!):
# 1 --> plot band power; 0 --> don't plot band power!
# tomography:
bands_to_plot = [0, 1, 1, 1, 1, 0, 0]

# smallest and highest multipole \ell for plotting (adjust for beautification!):
ell_low = 1e+1  
ell_high = 6e+3

# specify the range for the y-axis (different for EE or BB!!!)
# this is for EE:
y_low = 1e-7
y_high = 1e-2
# this is for linear B-modes:
#y_low = -0.5 
#y_high = 1. 

# if y-axis or x-axis tick labels overlap, choose a label index which will be pruned (guess work and only relevant for more than two z-bins!)
# E-modes:
prune_index_y = -3
# B-modes:
#prune_index_y = -1

# supply lower and upper limits for redshift bins:
# e.g. 2 z-bins:
z_min = [0.50, 0.85]
z_max = [0.85, 1.30]
# e.g. 1 z-bin (i.e. 2D):
#z_min = [0.50]
#z_max = [1.30]

# Just needed for subfolder names...
sigma_int = 0.290

# if there's an external (e.g. from runs on clones) covariance available set boolean to True:
external_covariance = False
# if you want to compare to Fisher errors at the same time, set to True:
fisher_errors = True
# run-to-run errors (i.e. std / sqrt(N)):
run_to_run_errors = False
# if the data shouldn't be weighted by external covariance:
external_weights = True
# and supply path to external weights:
# you need to look up file-format and name in code...
path_to_weights = '/your/path/to/such/a/file/'

# and supply the path pointing to the external covariance:
# you need to look up file-format and name in code...
root_in_ext_cov = '/your/path/to/such/a/file/'
# type of external covariance: 'clones', 'grfs', 'stitched', 'kids450'
# you need to look up function in code...
type_ext_cov = 'kids450'

# and supply the number of runs used in the estimation:
# only applicable if external covariance is based on run-to-run covariance to which no Hartlap-factor was applied yet 
nruns_ext = 184
 
### PLOTTING PARAMETERS ###    

# various fontsizes:
fontsize_label = 10 
fontsize_ticks = 9 
fontsize_legend = 9
fontsize_title = 9

# linewidth used in main plot:
lw = 2

# size of points for mean of band power:
sp = 40

# type of output file(s), e.g. 'pdf', 'eps', 'png'
# pass as a list (even for single extension)!
file_types = ['pdf']
# for transparent background set to 'True' (only works for 'png'):
transparent_background = False

# 1 plot per correlation and band type:
make_single_plots = False

# plot all correlations of one band type in one figure with subplots:
make_plots_for_paper = True

# type of error presentation in plot (errorboxes or errorbars):
box_plot = True

# if you want to plot a correlation matrix:
plot_band_correlations = False

# if theoretical prediction of power spectrum should be plotted:
plot_theory = True

# if noise power spectrum should be plotted:
plot_noise_power = True
# supply path down to file containing noise parameters:
# you need to look up file-format and name in code...
path_to_noise_parameters = '/your/path/to/such/a/file/your_file.dat'

# if you want to plot the convolution of the theoretical curve with the band window functions:
plot_convolved_theory = False

# supply path including file name from where to load theory (required if plot_theory = True or plot_convolved_theory = True):
# expected format of file:
# ell, D_ell_11, D_ell_21, D_ell_22, D_ell_31, D_ell_32...
# and D_ell = C_ell*ell*(ell+1)/2pi
path_to_theory = ['/your/path/to/such/a/file/your_theory1.dat', '/your/path/to/such/a/file/your_theory2.dat', ...]

# also supply label for theory:
label_theory = [r'$\mathrm{theory}_1$', r'$\mathrm{theory}_2$', ...]

# color for theory lines:      
# try to avoid green and red at the same time!!!               
color_theory = ['black', 'blue', ...]
# linestyles for theory lines:
ls_theory = ['-', ':', ...]     
# linewidth for theory plots:
lw_theory = [2, 1, ...]               
                     
# if band window functions should be plotted:
plot_band_window_functions = False
# show plots immediately (only works if plot_band_windows = False):
show_plots = True
# number of \ell-nodes used for this calculation must be specified:
ell_nodes = 100
