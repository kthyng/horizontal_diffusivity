'''
Calculate FSLE of drifters starting from the same locations.
'''

import numpy as np
import pdb
from matplotlib.mlab import find
import netCDF4 as netCDF
from scipy import ndimage
import time
from glob import glob
import tracpy
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['font.sans-serif'] = 'Arev Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Helvetica, Avant Garde, sans-serif'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.cal'] = 'cursive'
mpl.rcParams['mathtext.rm'] = 'sans'
mpl.rcParams['mathtext.tt'] = 'monospace'
mpl.rcParams['mathtext.it'] = 'sans:italic'
mpl.rcParams['mathtext.bf'] = 'sans:bold'
mpl.rcParams['mathtext.sf'] = 'sans'
mpl.rcParams['mathtext.fallback_to_cm'] = 'True'

def get_dist(lon1, lons, lat1, lats): 
    '''
    Function to compute great circle distance between point lat1 and lon1 
    and arrays of points given by lons, lats or both same length arrays.
    Uses Haversine formula. Distance is in km.
    '''

    lon1 = lon1*np.pi/180.
    lons = lons*np.pi/180.
    lat1 = lat1*np.pi/180.
    lats = lats*np.pi/180.

    earth_radius = 6373.
    distance = earth_radius*2.0*np.arcsin(np.sqrt(np.sin(0.50*(lat1-lats))**2 \
                                       + np.cos(lat1)*np.cos(lats) \
                                       * np.sin(0.50*(lon1-lons))**2))
    return distance

def calc_fsle(lonp, latp, tp, alpha=np.sqrt(2)):
    '''
    Calculate the relative dispersion of tracks lonp, latp as directly compared with
    the tracks described by lonpc, latpc. The two sets of tracks must start in the same 
    locations since this is assumed for making "pairs" of drifters for comparison (and 
    therefore pairs do not need to be found). The relative dispersion in this case is a
    measure of the difference between the two simulations, and is aimed at being used
    for examining differences in tracks due to changes in the numerical simulation.
    The tracks should also be coincident in time, but the script will find a way to match
    them up for the overlap periods.


    Inputs:
        lonpc, latpc    Longitude/latitude of the control drifter tracks [ndrifter,ntime]
        lonp, latp      Longitude/latitude of the drifter tracks [ndrifter,ntime]
        squared         Whether to present the results as separation distance squared or 
                        not squared. Squared by default.

    Outputs:
        D2              Relative dispersion (squared or not) averaged over drifter 
                        pairs [ntime]. In km (squared or not).
        nnans           Number of non-nan time steps in calculations for averaging properly.
                        Otherwise drifters that have exited the domain could affect calculations.

    To combine with other calculations of relative dispersion, first multiply by nnans, then
    combine with other relative dispersion calculations, then divide by the total number
    of nnans.

    Example call:
    dc = netCDF.Dataset('tracks/tseas_use300_nsteps1.nc') (5 min, control output)
    d = netCDF.Dataset('tracks/tseas_use1200_nsteps1.nc') (20 min, comparison output)
    tracpy.calcs.rel_dispersion_comp(dc.variables['lonp'][:], dc.variables['latp'][:], dc.variables['tp'][:],
                                     d.variables['lonp'][:], d.variables['latp'][:], d.variables['tp'][:],
                                     squared=True)
    '''
 
    #pdb.set_trace()

    ndrifters = lonp.shape[0]
    ntime = lonp.shape[1]

    dist = np.empty((np.cumsum(xrange(ndrifters))[-1],ntime))
    driftercount = 0 # holds index in dist for drifters

    # Construct drifters into [distance x time] in dist
    for idrifter in xrange(ndrifters):

        # lonpc = lonp[idrifter,:]
        # latpc = latp[idrifter,:]

        ndriftless = ndrifters-idrifter-1

# TRYING TO PUT ALL PAIRS OF DRIFTERS IN ORDER IN 1ST DIMENSION
# WITH TIME IN SECOND IN ORDER TO GO THROUGH THE DRIFTERS MORE QUICKLY

        dist[driftercount:driftercount+ndriftless,:] = get_dist(lonp[idrifter,:], lonp[idrifter+1:,:], 
                                                            latp[idrifter,:], latp[idrifter+1:,:]) # in km

        driftercount += ndriftless

    dist = dist[np.newaxis,:]

    # distances increasing with factor alpha
    Rs = np.asarray([np.array([0.7])*alpha**i for i in np.arange(20)]) # in km

    # times at the relevant distances
    #tSave = tp[idist] # in datetime representation
    from datetime import datetime
    units = 'seconds since 1970-01-01'
    t0 = netCDF.date2num(datetime(2009,10,1,0,0), units)
    tshift = (tp-t0)

    #pdb.set_trace()

    # # The distances right after passing Rs values
    # dist[0,(dist>=Rs).argmax(axis=1)]
    # Indices of where in dist the first entries are that are
    # just bigger than the Rs
    idist = (dist.repeat(Rs.size, axis=0).T>=Rs.T).argmax(axis=0)
    # idist = (dist>=Rs).argmax(axis=1)

    #pdb.set_trace()

    # ## STILL DEAL WITH THIS
    # indtemp = ind!=0
    # ind = ind[indtemp] # don't want to skip first zero if it is there
    # ## STILL DEAL WITH THIS

    # distances at the relevant distances (including ones we don't want at the end)
    # dSave = dist[0][idist]
    tSave = tshift[idist] # in seconds
    tSave[:,1:] = abs(np.diff(tSave, axis=1))
    tSave[:,0] = 0

    # Indices of entries that don't count
    ind = tSave==0
    # ind = find(idist==0)

    # Eliminate bad entries, but skip first since that 0 value should be there
    # dSave[ind] = np.nan
    tSave[ind] = np.nan
    # tSave[:,1:] = np.diff(tSave)
    # tSave[:,0] = 0
    # return dSave, tSave/(3600.*24) # tSave in days
    return tSave/(3600.*24) # tSave in days


def plot():

    alpha = np.sqrt(2)
    Rs = np.asarray([np.array([0.7])*alpha**i for i in np.arange(20)])

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111)

    # ----- all the lines ----- #

    # doturb=2, ah=5
    tSave = np.zeros(20)
    # dSave = np.zeros(20)
    nnans = np.zeros(20)
    Files = glob('tracks/doturb2_ah5/*fsle.npz')
    for File in Files:
        d = np.load(File)
        # dSave += d['dSave']
        tSave += d['tSave']
        nnans += d['nnans']
    d.close()
    l = 1/((tSave/nnans))
    ax.loglog(Rs, l, '-', lw=4, color='0.1', ms=10)

    # doturb=1, ah=20
    tSave = np.zeros(20)
    # dSave = np.zeros(20)
    nnans = np.zeros(20)
    Files = glob('tracks/doturb1_ah20/*fsle.npz')
    for File in Files:
        d = np.load(File)
        # dSave += d['dSave']
        tSave += d['tSave']
        nnans += d['nnans']
    d.close()
    l = 1/((tSave/nnans))
    ax.loglog(Rs, l, '-.', lw=4, color='0.4', ms=10)

    # doturb=0, ah=0 -- no interpolation
    tSave = np.zeros(20)
    # dSave = np.zeros(20)
    nnans = np.zeros(20)
    Files = glob('tracks/doturb0_ah0/fsle_nointerp/*fsle.npz')
    for File in Files:
        d = np.load(File)
        # dSave += d['dSave']
        tSave += d['tSave']
        nnans += d['nnans']
    d.close()
    l = 1/((tSave/nnans))
    ax.loglog(Rs, l, ':', lw=4, color='0.6', ms=10)

    ## interpolation makes no difference 
    ## doturb=0, ah=0 -- with interpolation
    #tSave = np.zeros(20)
    #dSave = np.zeros(20)
    #nnans = np.zeros(20)
    #Files = glob('tracks/doturb0_ah0/*fsle.npz')
    ## Files = glob('tracks/doturb0_ah0/fsle_interp/*fsle.npz')
    #for File in Files:
    #    d = np.load(File)
    #    dSave += d['dSave']
    #    tSave += d['tSave']
    #    nnans += d['nnans']
    #d.close()
    #l = 1/((tSave/nnans))
    #ax.loglog(Rs, l, 'o', color='0.8', ms=10)

    # theory
    ax.loglog(Rs[-11:-4], 1*(Rs[-11:-4])**(-2/3.), 'r', lw=3, alpha=0.6)
    ax.loglog(Rs[-11:-4], 2*(Rs[-11:-4])**(-2/7.), 'r', lw=3, alpha=0.6)

    # data
    lacasce = np.loadtxt('LaCasce2008_fsle.txt')
    ax.loglog(lacasce[:,0], lacasce[:,1], 'r*', ms=14)

    # legend and labels
    ax.text(0.5, 0.3, r'$D^{-2/3}$', color='r', transform=ax.transAxes)
    ax.text(0.5, 0.7, r'$D^{-2/7}$', color='r', transform=ax.transAxes)
    ax.set_xlabel('Separation Distance [km]')
    ax.set_ylabel(r'$\lambda$')
    ax.text(0.05, 0.3, 'Data', color='r', transform=ax.transAxes)
    ax.text(0.05, 0.25, 'Functions', color='r', transform=ax.transAxes, alpha=0.6)
    ax.text(0.05, 0.2, r'doturb=2, $A_H=5$ (-)', color='0.1', transform=ax.transAxes)
    ax.text(0.05, 0.15, r'doturb=1, $A_H=20$ (-.)', color='0.4', transform=ax.transAxes)
    ax.text(0.05, 0.1, r'doturb=0, $A_H=0$ (:)', color='0.6', transform=ax.transAxes)
    fig.savefig('figures/fsle_comp.pdf', bbox_inches='tight')



def run():
    '''
    Run FSLE calculation for shelf transport drifter simulations.
    '''

    loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_nesting6.nc'
    grid = tracpy.inout.readgrid(loc, usebasemap=False)

    Files = ['tracks/doturb0_ah0/2009-10-08T00.nc',
               'tracks/doturb0_ah0/2009-10-15T00.nc']
    # Files = glob('tracks/doturb0_ah0/*.nc')
    # Files = glob('tracks/doturb1_ah20/*.nc')
    # Files = glob('tracks/doturb2_ah5/*.nc')

    for File in Files:

        fname = File[:-3] + 'fsle.npz'

        # if os.path.exists(fname): # don't redo if already done
        #     continue

        d = netCDF.Dataset(File)
        lonp = d.variables['lonp'][:]
        latp = d.variables['latp'][:]
        tp = d.variables['tp'][:]
        d.close()


        # Loop over pairs of drifters from this area/time period and sum the FSLE, 
        # then average at the end

        ndrifters = lonp.shape[0]
        # dSave = np.zeros(20)
        tSave = np.zeros((1,20))
        nnans = np.zeros((1,20)) # to collect number of non-nans over all drifters for a time
        ddrifter = 500 # how many drifter indices to include at once
        driftercount = 0

        # logic for looping through more than 1 drifter at once
        while driftercount < ndrifters:
            print 'drifter ' + str(driftercount) + ' of ' + str(ndrifters)
            tSavetemp = calc_fsle(lonp[driftercount:driftercount+ddrifter,:], 
                                latp[driftercount:driftercount+ddrifter,:], tp)
           #pdb.set_trace()
            ind = ~np.isnan(tSavetemp)
            tSave += np.nansum(tSavetemp, axis=0)
            nnans += ind.sum(axis=0)
            driftercount += ddrifter

        # Save fsle for each file/area combination, NOT averaged
        np.savez(fname, tSave=tSave, nnans=nnans)
        print 'saved file', fname



if __name__ == "__main__":
    run()    
