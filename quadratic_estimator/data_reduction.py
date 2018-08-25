#!/usr/bin/env python
# encoding: UTF8
# Code for estimating maximum likelihood band powers of the lensing power spectrum
# following the "quadratic estimator" method by Hu & White 2001 (ApJ, 554, 67)
# implementation by Fabian Koehlinger

import os
import numpy as np
#import scipy.integrate as integrate
#import scipy.interpolate as interpolate
#import scipy.special as special
import astropy.io.fits as fits
#import astropy.coordinates as coord
#import astropy.units as units
# this is better for timing than "time"
#import time
from timeit import default_timer as timer

def weighted_variance(values, weights):
    '''Return the weighted average and standard deviation.

       values, weights -- Numpy ndarrays with the same shape.
    '''

    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)

    return variance

def get_distances_and_angles(x_rad, y_rad):
    ''' convenience function

        Args:

        x: tangent-plane position in radians
        y: tangent-plane position in radians

        Return:

        r: array of all distances between all points (x_i, y_i) and (x_j, y_j)
        phi: array of all angles between x-axis and distance vector between all points (x_i, y_i) and (x_j, y_j)

    '''

    r = []
    phi = []

    for i in xrange(x_rad.size):
        r.append(get_distances(x_rad[i], y_rad[i], x_rad, y_rad))
        phi.append(get_angles(x_rad[i], y_rad[i], x_rad, y_rad))

    return np.asarray(r), np.asarray(phi)

def get_distances(x1, y1, x2, y2):
    '''
        fine to use in current approach, since I deproject the area in the sky
    '''

    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_angular_distances(x1, y1, x2, y2, deg=False):
    ''' Calculates distance between points (x1, y1) and (x2, y2) on a (unit) sphere

        Args:
            x1, y1, x2, y2: float or array of floats; assumed to be in unit decimal degree
            deg:            boolean, if True, output is in decimal degree, else in radian

        Returns:
            theta:          Spherical distance between (x1, y1) and (x2, y2) in units set with keyword 'deg'
            t:              runtime

    '''

    t0 = timer()
    a1 = np.deg2rad(x1)
    a2 = np.deg2rad(x2)
    d1 = np.deg2rad(y1)
    d2 = np.deg2rad(y2)

    denominator = np.sqrt(np.cos(d2)**2*np.sin(a2-a1)**2+(np.cos(d1)*np.sin(d2)-np.sin(d1)*np.cos(d2)*np.cos(a2-a1))**2)
    nominator = np.sin(d1)*np.sin(d2)+np.cos(d1)*np.cos(d2)*np.cos(a2-a1)
    theta = np.arctan2(denominator, nominator)

    if deg:
        theta *= 180. / np.pi

    dt = timer() - t0

    return theta, dt

def get_angles(x1, y1, x2, y2):
    #x1, y1 = vector1
    #x2, y2 = vector2

    # I modified this now for testing BB to:
    # np.arctan2(y2-y1, x2-x1); didn't change anything
    # I think it should be the other way around, i.e.: y2-y1 & x2-x1
    dy = y2 - y1
    dx = x2 - x1

    # this is arctan(dy/dx)
    return np.arctan2(dy, dx)

def get_tangent_plane_coords(RA, Dec, tangent_point=None, mode='data'):
    r'''
    This is actually the same as "get_gnomonic_coords", just with minus signs in
    front of x and y, because we use sky coordinates and not longitude and latitude
    on an Earth globe!

    Define
    scale:  number of pixels per degree in the map
    alpha, delta:  Equatorial coordinates of a given position
    alpha0, delta0:  Equatorial coordinates of the map center

    Reference:
    http://lambda.gsfc.nasa.gov/product/iras/coordproj.cfm

    '''

    # According to the reference this scale means "pixel/degree" and is converted to "pixel/radian" in the calculation of F
    # This is absolutely irrelevant for my analysis, since it will only rescale the sizes in the projected tangent plane,
    # but I'm only interested in distances between points of this tangent plane (and I use 64Bit floats).
    scale = np.pi / 180.
    # for CFHTLenS: 0.187 arcsec/pixel
    # Hence:
    #scale = 1./(0.187/3600.)
    # but still not sure if that is relevant... Currently the effective scale is set to "1"
    # quick test shows: using the CFHTLenS scale yields super-large numbers for the tangent-plane coordinates; this is not correct

    if mode == 'data':
        sign = -1.
    else:
        sign = 1.

    alpha = np.deg2rad(RA)
    delta = np.deg2rad(Dec)

    # Define arcs through edges of fields in order to determine tangent point (= intersection of two great circles as defined by arcs)
    point1 = (RA.min(), Dec.max())
    point2 = (RA.max(), Dec.min())

    arc1 = (point1, point2)

    point1 = (RA.max(), Dec.max())
    point2 = (RA.min(), Dec.min())

    arc2 = (point1, point2)

    if tangent_point:
        RA0, Dec0 = tangent_point[0], tangent_point[1]
    else:
        RA0, Dec0 = get_tangent_point(arc1, arc2)

    alpha0 = np.deg2rad(RA0)
    delta0 = np.deg2rad(Dec0)

    print 'Tangent point: RA = {:.5f} deg, Dec = {:.5f} deg'.format(RA0, Dec0)

    A = np.cos(delta) * np.cos(alpha - alpha0)
    # ATTENTION: F carries a sign, too, depending on the signs of d0 and d!
    F = scale * (180. / np.pi) / (np.sin(delta0) * np.sin(delta) + A * np.cos(delta0))

    #then the pixel coordinates in the image are
    # here I have to be careful, the sign flip should only occur when I'm dealing with a 'real', i.e., measured R.A. coordinate, otherwise there shouldn't occur any sign flip (e.g. for mocks like GRFs or clones)
    x_rad = sign * F * np.cos(delta) * np.sin(alpha - alpha0)
    # I think the minus sign here was wrong (as given in the reference)... We still want y-axis to be aligned with increasing Dec...
    # After a lengthy discussion with Massimo, the minus sign in the coordinate transformation should be fine; e2 should get a sign-flip though AFTER the c2-correction, which is not necessary for GRFs or clones though!
    # After having reconsidered, I think again that there shouldn't be a minus sign (in contrast to the formulas as given in the reference)
    # TODO: BWMs are funny for W1 and W2 (negative norm already in E5) which are the only fields with negative Dec...
    # Latest conclusion (01.06.15):
    # if x and y are both defined positive (like in Wolfram reference) or negative (like in NASA reference) doesn't matter in general.
    # in my particular case it does matter though in the sense of that I have to make sure that increasing x is aligned with decreasing RA and that y is always pointing to the North pole!
    # I have changed the definition of arctan2(dy/dx) now from dy=y1-y2 and dx=x1-x2 to dy=y2-y1 and dx=x2-x1... See if that'll make things consistent now...
    y_rad = F * (np.cos(delta0) * np.sin(delta) - A * np.sin(delta0))

    return x_rad, y_rad, (RA0, Dec0)

def get_tangent_point(arc1=((2., -2.), (-2., 2.)), arc2=((2., 2.), (-2., -2.))):
    r"""
      Calculate intersection point of two great circles on sphere which are defined by 2 points per great circle.
      Points are expected in (RA, Dec) measured in degrees.

    """

    def get_polar_coords(point):

        alpha = np.deg2rad(point[0])
        delta = np.deg2rad(point[1])

        # Note: since RA, Dec is inside sphere vs longitude, latitude which is outside sphere "x" is defined with cosine(alpha) for RA, Dec!
        x = np.cos(delta) * np.cos(alpha)
        y = np.cos(delta) * np.sin(alpha)
        z = np.sin(delta)

        vec = np.asarray([x, y, z])

        return vec

    # 1. get 3-vectors in polar coordinates for each (RA, Dec)-point
    vec1_arc1 = get_polar_coords(arc1[0])
    vec2_arc1 = get_polar_coords(arc1[1])

    vec1_arc2 = get_polar_coords(arc2[0])
    vec2_arc2 = get_polar_coords(arc2[1])


    # 2. Get normal to planes containing great circles.
    #    It's the cross product of vector to each point from the origin.
    normal1 = np.cross(vec1_arc1, vec2_arc1)
    normal2 = np.cross(vec1_arc2, vec2_arc2)

    # 3. Find line of intersection between two planes.
    #    It is normal to the poles of each plane.
    intersect = np.cross(normal1, normal2)

    # 4. Find intersection points.
    intersection_point1 = intersect / np.sqrt(intersect[0]**2 + intersect[1]**2 + intersect[2]**2)
    intersection_point2 = -intersection_point1

    RA1 = np.rad2deg(np.arctan2(intersection_point1[1], intersection_point1[0]))
    Dec1 = np.rad2deg(np.arcsin(intersection_point1[2]))

    RA2 = np.rad2deg(np.arctan2(intersection_point2[1], intersection_point2[0]))
    Dec2 = np.rad2deg(np.arcsin(intersection_point2[2]))

    #print 'First intersection point:'
    #print RA1, Dec1
    #print 'Second intersection point:'
    #print RA2, Dec2

    # Take care of negative RA!
    if RA1 < 0.:
        RA1 += 360.

    if RA2 < 0.:
        RA2 += 360.

    #print 'First intersection point:'
    #print RA1, Dec1
    #print 'Second intersection point:'
    #print RA2, Dec2

    RA_min = min(arc1[0][0], arc1[1][0], arc2[0][0], arc2[1][0])
    RA_max = max(arc1[0][0], arc1[1][0], arc2[0][0], arc2[1][0])

    Dec_min = min(arc1[0][1], arc1[1][1], arc2[0][1], arc2[1][1])
    Dec_max = max(arc1[0][1], arc1[1][1], arc2[0][1], arc2[1][1])

    '''
    print RA_min, RA_max
    print Dec_min, Dec_max
    '''
    # check that RA, Dec is contained in arcs:
    if RA1 >= RA_min and RA1 <= RA_max and Dec1 >= Dec_min and Dec1 <= Dec_max:

        #print 'Tangent point will be first intersection point.'

        return RA1, Dec1

    else:

        #print 'Tangent point will be second intersection point.'

        return RA2, Dec2

# generalized data reduction top-level function for 'nzbins':
def get_data(paths_to_data, filename, names_zbins=['0.50z0.85', '0.85z1.30'], identifier='W2', sigma_int=[0.25], pixel_scale=0.15, nzbins=1, mode='data', min_num_elements_pix=1., column_names={}):

    fname_control = 'control_outputs/' + identifier + '_binned_'

    x_coords = []
    y_coords = []
    xmins = np.zeros(nzbins)
    ymins = np.zeros(nzbins)
    xmaxs = np.zeros(nzbins)
    ymaxs = np.zeros(nzbins)

    tangent_points = []

    # first loop is necessary for finding minimal mask coordinates:
    for index_zbin in xrange(nzbins):

        # load catalog:
        table = fits.open(paths_to_data[index_zbin] + filename, memmap=True)
        tbdata = table[1].data

        # try to read a GLOBAL (i.e. independent of zbin-cut!) tangent point
        try:
            header = table[0].header
            tangent_point = (header['TP_ALPHA_J2000'], header['TP_DELTA_J2000'])
            print 'Read tangent point from FITS-header which is expected to be global (i.e. independent of any catalog cuts).'
        except:
            # we calculate the tangent point for lowest tomographic bin and pass it on to all other bins!
            tangent_point = None
            if mode == 'data':
                print 'Could not read a global tangent point from FITS-header. Calculating it now for first redshift bin and pass it on.'

        #RA = tbdata['ALPHA_J2000']
        #Dec = tbdata['DELTA_J2000']
        RA = tbdata[column_names['RA']]
        Dec = tbdata[column_names['Dec']]

        # project coordinates on tangent-plane:
        # after that I can just use Euclidean geometry (e.g. binning in square pixels and distances between pixel centres etc...)
        # if we are dealing with GRFs or Clones, we don't need a tangent plane projection:
        #if identifier == 'G' or identifier == 'M':
        if mode != 'data':
            # I think I still have to flip the x-coordinate though...  --> nope!
            x_rad, y_rad = np.deg2rad(RA), np.deg2rad(Dec)
            # mock data is always assumed to be centred around (0., 0.)
            tangent_point = (0., 0.)
        else:
            x_rad, y_rad, tangent_point = get_tangent_plane_coords(RA, Dec, tangent_point=tangent_point, mode=mode)

        tangent_points.append(tangent_point)

        xmins[index_zbin] = x_rad.min()
        ymins[index_zbin] = y_rad.min()
        xmaxs[index_zbin] = x_rad.max()
        ymaxs[index_zbin] = y_rad.max()

        x_coords.append(x_rad)
        y_coords.append(y_rad)

    # check all catalog borders and always choose the smaller value (in order to avoid introducing artificial lower number densities in the larger catalog):
    # minimal overlap between zbins (and xmin, ymin < 0!)!!!
    xmin = xmins.max()
    ymin = ymins.max()

    #print xmins, xmin
    #print ymins, ymin

    # minimal overlap between zbins!!!
    xmax = xmaxs.min()
    ymax = ymaxs.min()

    # second loop is really for data reduction:
    g1_all_zbins = []
    g2_all_zbins = []
    e1_all_zbins = []
    e2_all_zbins = []
    weights_all_zbins = []
    N_eff_all_zbins = []
    #shot_noise_all_zbins = []
    x_all_zbins = []
    y_all_zbins = []
    field_props_all_zbins = []
    for index_zbin in xrange(nzbins):

        # some naming:
        fname_binned = fname_control + names_zbins[index_zbin] + '.cat'

        # load catalog:
        # loading it again is probably more efficient than creating a 'list of catalogs' in the first loop...
        table = fits.open(paths_to_data[index_zbin] + filename, memmap=True)
        tbdata = table[1].data
        cols = table[1].columns
        # in order to do things correctly, I should return data unmasked and combine the masks here...
        g1, g2, e1, e2, weights, N_eff, sigma, field_properties, x, y, mask = reduce_data(tbdata, cols, x_coords[index_zbin], y_coords[index_zbin], xmin, xmax, ymin, ymax, identifier=identifier, pixel_scale=pixel_scale, fname_control=fname_binned, min_num_elements_pix=min_num_elements_pix, index_zbin=index_zbin, column_names=column_names)

        # pass tangent point to dictionary
        field_properties['Tangent_point'] = tangent_points[index_zbin]

        # collect them all:
        g1_all_zbins.append(g1)
        g2_all_zbins.append(g2)
        e1_all_zbins.append(e1)
        e2_all_zbins.append(e2)
        #shot_noise_all_zbins.append(shot_noise)
        weights_all_zbins.append(weights)
        N_eff_all_zbins.append(N_eff)
        x_all_zbins.append(x)
        y_all_zbins.append(y)
        field_props_all_zbins.append(field_properties)

        if index_zbin == 0:
            maximal_mask = mask
        else:
            maximal_mask = maximal_mask & mask

    # third loop for proper masking with maximal mask between redshift bins:
    data_masked = []
    x_masked = []
    y_masked = []
    shot_noise_masked = []
    #sigma_all_zbins = []
    for index_zbin in xrange(nzbins):
        # shear = (g11, ..., g1n, g21, ..., g2n)
        data_masked.append(np.concatenate((g1_all_zbins[index_zbin][maximal_mask.flatten()], g2_all_zbins[index_zbin][maximal_mask.flatten()])))
        x_masked.append(x_all_zbins[index_zbin][maximal_mask.flatten()])
        y_masked.append(y_all_zbins[index_zbin][maximal_mask.flatten()])
        e1_var = weighted_variance(e1_all_zbins[index_zbin][maximal_mask.flatten()], weights_all_zbins[index_zbin][maximal_mask.flatten()])
        e2_var = weighted_variance(e2_all_zbins[index_zbin][maximal_mask.flatten()], weights_all_zbins[index_zbin][maximal_mask.flatten()])
        sigma_int_est_sqr = (e1_var + e2_var) / 2.
        if sigma_int[index_zbin] == -1.:
            # TODO: double check shot noise estimate --> yap: sigma^2 / N_eff is what we save in "shot_noise""
            shot_noise_masked.append(sigma_int_est_sqr / N_eff_all_zbins[index_zbin][maximal_mask.flatten()])
            # overwrite:
            sigma_int[index_zbin] = np.sqrt(sigma_int_est_sqr)
        else:
            shot_noise_masked.append(sigma_int[index_zbin]**2 / N_eff_all_zbins[index_zbin][maximal_mask.flatten()])

    return np.asarray(data_masked), shot_noise_masked, field_props_all_zbins, np.asarray(x_masked), np.asarray(y_masked)

def reduce_data(tbdata, cols, x_rad, y_rad, xmin, xmax, ymin, ymax, identifier='W2', pixel_scale=0.15, fname_control='your_name_here', min_num_elements_pix=1., index_zbin=1, column_names={}):
    '''This function is a copy of "get_data", except for the loading of the catalogs and the projection into the tangent-plane.

       @arguments:

       tbdata

       x_rad, y_rad

       xmin, xmax, ymin, ymax

       identifier

       sigma_int

       pixel_scale

       sub_id

       chop

       fname_control

       @return:


    '''

    #RA = tbdata['ALPHA_J2000']
    #Dec = tbdata['DELTA_J2000']

    RA = tbdata[column_names['RA']]
    Dec = tbdata[column_names['Dec']]

    # apply c-correction on the fly!
    '''
    if 'c1' in cols.names:
        c1 = tbdata['c1']
    else:
        c1 = np.zeros_like(RA)
    '''
    if column_names['c1'] != 'dummy':
        c1 = tbdata[column_names['c1']]
    else:
        c1 = np.zeros_like(RA)
    #e1 = tbdata['e1'] - c1
    e1 = tbdata[column_names['e1']] - c1

    '''
    if 'c2' in cols.names:
        c2 = tbdata['c2']
    else:
        c2 = np.zeros_like(RA)
    '''
    if column_names['c2'] != 'dummy':
        c2 = tbdata[column_names['c2']]
    else:
        c2 = np.zeros_like(RA)

    #e2 = tbdata['e2'] - c2
    e2 = tbdata[column_names['e2']] - c2

    #weight = tbdata['weight']
    if column_names['weight'] != 'dummy':
        weight = tbdata[column_names['weight']]
    else:
        weight = np.ones_like(RA)

    '''
    if 'm1' in cols.names and 'm2' in cols.names:
        m1 = tbdata['m1']
        m2 = tbdata['m2']
    elif 'm' in cols.names:
        m = tbdata['m']
        m1 = m
        m2 = m
    else:
        m1 = np.zeros_like(RA)
        m2 = m1
    '''
    if column_names['m1'] != 'dummy' and column_names['m2'] != 'dummy':
        m1 = tbdata[column_names['m1']]
        m2 = tbdata[column_names['m2']]
    else:
        m1 = np.zeros_like(RA)
        m2 = m1

    #photo_z_pdf = tbdata['PZ_full']
    #photo_z = tbdata['Z_B']

    #g1 = weight * e1
    # introduce a sign flip (RA = -x...)
    #g2 = weight * e2

    m1_bias = weight * m1
    m2_bias = weight * m2

    # this control file is rather large and for KiDS a copy of the input-catalogs (+tangent plane coords)
    fname_check_out = 'control_outputs/' + identifier + '_check_coord_zbin{:}.cat'.format(index_zbin + 1)
    if not os.path.isfile(fname_check_out):
        tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='ALPHA_J2000', format='D', array=RA),
                                               fits.Column(name='DELTA_J2000', format='D', array=Dec),
                                               fits.Column(name='x_rad', format='D', array=x_rad),
                                               fits.Column(name='y_rad', format='D', array=y_rad),
                                               fits.Column(name='e1', format='D', array=e1),
                                               fits.Column(name='e2', format='D', array=e2),
                                               fits.Column(name='weight', format='D', array=weight),
                                               fits.Column(name='c1', format='D', array=c1),
                                               fits.Column(name='c2', format='D', array=c2),
                                               fits.Column(name='m1', format='D', array=m1),
                                               fits.Column(name='m2', format='D', array=m2)])
                                               #fits.Column(name='PZ_full', format='D', array=photo_z_pdf),
                                               #fits.Column(name='Z_B', format='D', array=photo_z)])

        prihdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(fname_check_out)

    #checkdata = [np.column_stack((alpha_deg, delta_deg, x_rad, y_rad, weight, e1, e2, c2, m)), header]

    # convert pixel scale from deg to rad:
    scale = np.deg2rad(pixel_scale)

    # dictionary for properties of field:
    field_properties = {}

    RA_min = RA.min()
    RA_max = RA.max()
    Dec_min = Dec.min()
    Dec_max = Dec.max()

    dist_flat = get_distances(xmin, ymin, xmax, ymax)
    dist_sph = get_angular_distances(RA_min, Dec_min, RA_max, Dec_max, deg=False)[0]

    print 'Max distance between tangent-plane coordinates (theta_x, theta_y): {:.4f} rad'.format(dist_flat)
    print 'Max distance between polar coordinates (alpha, delta): {:.4f} rad'.format(dist_sph)
    print 'Fractional difference between flat and spherical distance: {:.2f}%'.format(np.abs(dist_flat - dist_sph) / dist_sph * 100.)

    # tangent-plane coordinates are centred around 0!
    length_x = np.abs(xmin) + np.abs(xmax)
    length_y = np.abs(ymin) + np.abs(ymax)

    #print xmin, xmax, ymin, ymax

    # alternative to np.arange:
    nbins_x = length_x / scale
    nbins_y = length_y / scale

    #print nbins_x, nbins_y
    bins_x, steps = np.linspace(xmin, xmax, int(nbins_x), retstep=True)
    #print np.rad2deg(steps)
    bins_y, steps = np.linspace(ymin, ymax, int(nbins_y), retstep=True)
    #print np.rad2deg(steps)

    # np.arange is described to be unreliable for non-integer steps...
    bins_x_alt = np.arange(xmin, xmax, scale)
    bins_y_alt = np.arange(ymin, ymax, scale)

    #header = 'bins_x[linspace], bins_y[linspace]'
    #np.savetxt('control_outputs/'+identifier+'_bins_linspace.dat', np.column_stack((bins_x, bins_y)), header=header)

    #header = 'bins_x[arange], bins_y[arange]'
    #np.savetxt('control_outputs/'+identifier+'_bins_arange.dat', np.column_stack((bins_x, bins_y)), header=header)

    '''
    print 'xmin, xmax:', xmin, xmax
    print 'linspace x: \n', bins_x
    print 'arange x: \n', bins_x_alt
    print 'n linspace, n arange: \n', bins_x.size, bins_x_alt.size
    print 'step size linspace: \n', np.rad2deg(np.diff(bins_x))
    print 'step size arange: \n', np.rad2deg(np.diff(bins_x_alt))


    print 'ymin, ymax:', ymin, ymax
    print 'linspace y: \n', bins_y
    print 'arange y: \n', bins_y_alt
    print 'n linspace, n arange: \n', bins_y.size, bins_y_alt.size
    print 'step size linspace: \n', np.rad2deg(np.diff(bins_y))
    print 'step size arange: \n', np.rad2deg(np.diff(bins_y_alt))
    '''

    #exit()
    #print 'x_bins', bins_x.shape
    #print 'y_bins', bins_y.shape

    bins_x = bins_x_alt
    bins_y = bins_y_alt

    #nbins = (bins_x.shape[0]-1) * (bins_y.shape[0]-1)

    scale_x = np.diff(bins_x)[0]
    scale_y = np.diff(bins_y)[0]

    print 'Length of field (x) = {:.5} deg'.format(np.rad2deg(length_x))
    print 'Length of field (y) = {:.5} deg'.format(np.rad2deg(length_y))
    print '1pix (length in x) = {:.2} deg'.format(np.rad2deg(scale_x))
    print '1pix (length in y) = {:.2} deg'.format(np.rad2deg(scale_y))

    field_area_arcmin = np.rad2deg(length_x) * np.rad2deg(length_y) * 3600. # in arcmin^2
    field_area_deg = np.rad2deg(length_x) * np.rad2deg(length_y) # in deg^2
    cell_area_arcmin = np.rad2deg(scale_x) * np.rad2deg(scale_y) * 3600. # in arcmin^2
    cell_area_deg = np.rad2deg(scale_x) * np.rad2deg(scale_y) # in deg^2
    print 'Field area = {:.6} arcmin^2'.format(field_area_arcmin)
    print 'Field area = {:.6} deg^2'.format(field_area_deg)
    print 'Cell area = {:.6} arcmin^2'.format(cell_area_arcmin)
    print 'Cell area = {:.6} deg^2'.format(cell_area_deg)

    field_properties['borders_x'] = (np.rad2deg(xmin), np.rad2deg(xmax))
    field_properties['borders_y'] = (np.rad2deg(ymin), np.rad2deg(ymax))
    field_properties['pixel'] = (np.rad2deg(scale_x), np.rad2deg(scale_y), cell_area_deg)
    field_properties['field'] = (np.rad2deg(length_x), np.rad2deg(length_y), field_area_deg)

    sum_weight, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=weight)
    sum_sqr_weight, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=weight*weight)

    N_obj_per_pixel, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y])

    field_properties['npixel'] = sum_weight.size
    #print bins_x.shape, bins_y.shape

    sum_weight = sum_weight.T
    sum_sqr_weight = sum_sqr_weight.T
    N_obj_per_pixel = N_obj_per_pixel.T

    N_eff = sum_weight * sum_weight / sum_sqr_weight
    n_eff = N_eff / cell_area_arcmin

    m1_bias_avg, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=m1_bias)
    m1_bias_avg = m1_bias_avg.T / sum_weight

    m2_bias_avg, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=m2_bias)
    m2_bias_avg = m2_bias_avg.T / sum_weight

    e1_avg, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=weight * e1)
    g1_avg = e1_avg.T / sum_weight / (1. + m1_bias_avg)

    e2_avg, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=weight * e2)
    g2_avg = e2_avg.T / sum_weight / (1. + m2_bias_avg)

    weights_avg = sum_sqr_weight.T / sum_weight.T

    #print 'g1_avg, g2_avg', g1_avg.shape, g2_avg.shape

    # TODO: Instead of regular grid of cell-midpoints use averaged x, y?!
    x_avg, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=weight * x_rad)
    x_avg = x_avg.T / sum_weight

    y_avg, bins_x, bins_y = np.histogram2d(x_rad, y_rad, bins=[bins_x, bins_y], weights=weight * y_rad)
    y_avg = y_avg.T / sum_weight

    #x_mid = bins_x[:-1] + np.diff(bins_x) / 2.
    #y_mid = bins_y[:-1] + np.diff(bins_y) / 2.
    #xx, yy = np.meshgrid(x_mid, y_mid)

    print 'Shape of patch in bins:', xx.shape
    field_properties['field_shape_in_bins'] = xx.shape

    # two options:
    # 1) we calculate a weighted number of objects per pixel:
    #shot_noise = sigma_int**2. / N_eff
    # 2) we take the variance of the weighted mean:
    # ATTENTION: NOT true for weights defined in LensFit!!!
    # and we take the square root!!!
    # Don't use this... --> lensfit weights are NOT inverse variance weights!
    #shot_noise2 = 1. / np.sqrt(sum_weight)

    # TODO: update this...
    # make it a Fits-file...
    #header = 'x, y, g1, g2, N_eff, n_eff[arcmin^-2], weighted_counts, shot_noise, shot_noise2'
    #savedata = np.column_stack((xx.flatten(), yy.flatten(), g1_avg.flatten(), g2_avg.flatten(), N_eff.flatten(), n_eff.flatten(), sum_weight.flatten(), shot_noise.flatten(), shot_noise2.flatten()))
    #np.savetxt(fname_control, savedata, header=header)
    if not os.path.isfile(fname_control):
        tbhdu = fits.BinTableHDU.from_columns([fits.Column(name='x_rad', format='D', array=xx.flatten()),
                                               fits.Column(name='y_rad', format='D', array=yy.flatten()),
                                               fits.Column(name='g1', format='D', array=g1_avg.flatten()),
                                               fits.Column(name='g2', format='D', array=g2_avg.flatten()),
                                               fits.Column(name='N_eff', format='D', array=N_eff.flatten()),
                                               fits.Column(name='n_eff', format='D', array=n_eff.flatten()),
                                               fits.Column(name='sum_weight', format='D', array=sum_weight.flatten()),
                                               fits.Column(name='sum_sqr_weight', format='D', array=sum_sqr_weight.flatten())])
                                               #fits.Column(name='shot_noise', format='D', array=shot_noise.flatten())])

        prihdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([prihdu, tbhdu])
        thdulist.writeto(fname_control)

    # define a mask, because of survey geometry (e.g. W4)
    # we want to keep only pixels that contain elements with at least min_num_elements_pix
    #mask = sum_weight >= min_num_elements_pix #0.
    # Use N_eff here instead?!
    mask = N_eff >= min_num_elements_pix
    # no masking needed in this step, will be applied in top-level function!
    #shot_noise = shot_noise.flatten()
    #shot_noise2 = shot_noise2.flatten()

    # this should be correct now...
    # NOPE! I can't determine A_eff with a catalog that only contains unmasked objects,
    # hence, A_eff = #unmasked pixels * A_pix will always be overestimating the true A_eff
    # Therefore, n_eff is underestimated... I have to define that consistently with external data!
    # Just use it as indicators here (and maybe add over/under-estimated in logfile!)
    #'''
    nominator = np.sum(sum_weight[mask].flatten())
    denominator = np.sum(sum_sqr_weight[mask].flatten())
    N_eff_patch = nominator**2. / denominator
    field_area_masked = len(sum_weight[mask]) * cell_area_arcmin
    n_eff_patch = N_eff_patch / field_area_masked
    #'''

    field_properties['n_eff'] = n_eff_patch #[mask].mean()
    field_properties['N_eff'] = N_eff_patch
    # shear = (g11, ..., g1n, g21, ..., g2n)
    #shear = np.concatenate((g1_avg.flatten(), g2_avg.flatten()))

    return g1_avg.flatten(), g2_avg.flatten(), e1_avg.T.flatten(), e2_avg.T.flatten(), weights_avg.flatten(), N_eff.flatten(), field_properties, xx.flatten(), yy.flatten(), mask
