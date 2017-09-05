This repository contains code for a quadratic estimator algorithm in order to estimate the weak gravitational lensing shear power spectrum in multipole band powers and in tomographic bins (i.e. redshift slices along the line-of-sight). The algorithm was proposed in [Hu & White 2001 (ApJ, 554, 67)](http://adsabs.harvard.edu/abs/2001ApJ...554...67H), further described in [Lin et al. 2012 (ApJ, 761, 15)](http://adsabs.harvard.edu/abs/2012ApJ...761...15L) and it was expanded in [Koehlinger et al. 2016 (MNRAS, 456, 1508)](http://adsabs.harvard.edu/abs/2016MNRAS.456.1508K) to allow for a tomographic analysis, including its application to shear catalogues from CFHTLenS. Most recently it was used in [Koehlinger et al. 2017 (MNRAS, 471, 4412)](http://adsabs.harvard.edu/abs/2017MNRAS.471.4412K) on shear catalogues from KiDS-450. 

The last two papers also include a self-consistent description of the algorithm. Please cite them if results based on this code are published! 

The repository contains the following folders:

* `quadratic_estimator` -- containing the main modules for the algorithm
* `plotting` -- containing optional (and yet not very user-friendly) plotting routines
* `input` -- containing an example parameter-file for the main algorithm

The code is written in Python (2.7.X) and requires the following packages in addition to other standard packages:

* numpy
* scipy
* multiprocessing
* astropy
* h5py

The main module of the code is `quadratic_estimator.py` (contained in the same folder) and it can be used with the command:
```
python /your/path/to/quadratic_estimator.py /your/path/to/input/your_algorithm_parameters.ini > terminal_output.out
```
All settings for the algorithm (e.g. input catalogues, output folders, etc.) have to be adjusted in a file `your_algorithm_parameters.ini`. In the folder `input` we provide the example file `default.ini` and we hope that all options are self-consistently and sufficiently explained in there.

The major output of the code will be stored within the specified output folder(s):

* `band_powers_<EE, BB, EB>_<z_i>x<z_j>.dat` -- a simple text-file containing the measured E-mode, B-mode, and/or EB-mode band powers for each (unique) redshift bin correlation <z\_i\> x <z\_j\> from lowest multipole band to highest multipole band (as given in the file below)
* `multipoles_<EE, BB, EB>.dat` -- a simple text-file (with header!) containing the naive multipole bin centers and the multipole bin ranges for each band type as specified in 'your_algorithm_parameters.ini'-file (assumed to be the same for each redshift bin correlation!)
* `band_window_matrix_nell<number>.dat` -- if requested, the matrix containing the band window functions of each specified band power of type <EE, BB, EB\> for each unique redshift bin correlation
* `last_Fisher_matrix.dat` -- if nothing better is available, the INVERSE of this matrix can serve as a very idealised approximation to the covariance of the band powers and it comes 'for free'
* `*.log`-file -- copies all set parameters from the `*.ini`-file and lists other useful quantities related to the performance of the algorithm
* all other files are saved out of legacy reasons or for further diagnostics only and their purpose should become more obvious when digging into the code...
     
The folder `plotting` contains an experimental script for plotting the results (e.g. for ALL subfields of a survey) in publication quality. Major settings can be adjusted and must be supplied with a parameter file, an example of all options are listed in `default.plt`. Unfortunately, the script is in its current form not very user-friendly and still very tailored to the requirements of our previous analyses, so please consider it rather as a blueprint for your own (better) plotting scripts.

If you want to contribute to this project (e.g. a more efficient implementation using more advanced linear algebra tricks or the bruteforce way of porting it to C and/or GPUs...), please use a dedicated branch per feature/bugfix.

For questions/comments/issues/requests please use the issue-tracking system!

Last but not least you might want to play around with some real data, hence here are links to a personal choice of public shear catalogues (follow their acknowledgement policy if you use them!):

* CFHTLenS: http://www.cfhtlens.org/astronomers/data-store
* KiDS-450: http://kids.strw.leidenuniv.nl/sciencedata.php

