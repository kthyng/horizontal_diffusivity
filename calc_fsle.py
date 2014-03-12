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

def calc_fsle(lonpc, latpc, lonp, latp, tp, alpha=np.sqrt(2)):
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
 
    dist = get_dist(lonpc, lonp, latpc, latp) # in km
    dist = dist[np.newaxis,:]

    # distances increasing with factor alpha
    Rs = np.asarray([0.5*alpha**i for i in np.arange(20)]) # in km
    # Rs = np.asarray([0.1*alpha**i for i in np.arange(28)]) # in km

    pdb.set_trace()

    r0 = dist[:,0]
    Rs = np.asarray([r0*alpha**i for i in np.arange(20)]) # in km
    # # The distances right after passing Rs values
    # dist[0,(dist>=Rs).argmax(axis=1)]
    # Indices of where in dist the first entries are that are
    # just bigger than the Rs
    idist = (dist>=Rs).argmax(axis=1)
    # Indices of entries that don't count
    ind = find(idist==0)
    # distances at the relevant distances (including ones we don't want at the end)
    dSave = dist[0,idist]
    # times at the relevant distances
    tSave = (tp-tp[0])[idist] # in seconds
    # Eliminate bad entries, but skip first since that 0 value should be there
    dSave[ind[1:]] = np.nan
    tSave[ind[1:]] = np.nan

    return dSave, tSave

    # ntrac = dist.shape[0]
    # nt = dist.shape[1]

    # # Find first time dist>delta and dist>delta*alpha for each delta to
    # # then linearly interpolate to find the corresponding time
    # tau = np.zeros(Rs.size)
    # nnans = np.zeros(Rs.size) # not nans

    # for i, R in enumerate(Rs[:-1]):

    #     # indices of the first time the info changes from lower than R to higher
    #     ind1 = np.diff((dist<=R).astype(int), axis=1).argmin(axis=1)

    #     # These contain the indices in dist and tp of the elements below and above R
    #     distUse = np.vstack((dist[np.arange(0,ntrac),ind1], dist[np.arange(0,ntrac),ind1+1])).T
    #     tp2d = tp[np.newaxis,:].repeat(ntrac, axis=0)
    #     tpUse = np.vstack((tp2d[np.arange(0,ntrac),ind1], tp2d[np.arange(0,ntrac),ind1+1])).T


    #     # Replace incorrect cases (when zero was chosen by default) with nan's
    #     # bad cases: had defaulted to zero, or picked out last index
    #     # or: picked out last index before nans
    #     # or if dist that is greater than R accidentally got picked out
    #     nanind = (distUse[:,1]<distUse[:,0]) \
    #                 + (ind1==nt-1) \
    #                 + (np.isnan(dist[np.arange(0,ntrac),ind1+1])) \
    #                 + (dist[np.arange(0,ntrac),ind1]>R)
    #     distUse[nanind,:] = np.nan
    #     tpUse[nanind,:] = np.nan

    #     # Do linear interpolation by hand because interp won't take in arrays
    #     rp = (R-distUse[:,0])/(distUse[:,1] - distUse[:,0]) # weighting for higher side
    #     rm = 1 - rp # weighting for lower side

    #     # now find the interpolation time for each drifter
    #     time1 = rm*tpUse[:,0] + rp*tpUse[:,1]

    #     ## for delta*alpha ##

    #     # indices of the first time the info changes from lower than R to higher
    #     indtemp = np.diff((dist<=Rs[i+1]).astype(int), axis=1)
    #     ibad = np.sum(indtemp==-1, axis=1)==0 # when indtemp doesnt change sign, values never below Rs[i+1]
    #     ind2 = indtemp.argmin(axis=1)
    #     ind2[ibad] = -1 # need to use an integer since int array
    #     # pdb.set_trace()
    #     while np.sum(ind2[~ibad]<ind1[~ibad])>0: # don't count the -1's
    #         iskip = ind2<ind1
    #         # indtemp = np.diff((dist<=Rs[i+1]).astype(int), axis=1)
    #         indtemp[iskip,ind2[iskip]] = 0
    #         ibad = np.sum(indtemp==-1, axis=1)==0 # when indtemp doesnt change sign, values never below Rs[i+1]
    #         ind2 = indtemp.argmin(axis=1)
    #         ind2[ibad] = -1 # need to use an integer since int array

    #     # These contain the indices in dist and tp of the elements below and above R
    #     distUse = np.vstack((dist[np.arange(0,ntrac),ind2], dist[np.arange(0,ntrac),ind2+1])).T
    #     tpUse = np.vstack((tp2d[np.arange(0,ntrac),ind2], tp2d[np.arange(0,ntrac),ind2+1])).T


    #     # Replace incorrect cases (when zero was chosen by default) with nan's
    #     # bad cases: had defaulted to zero, or picked out last index
    #     # also eliminating when I have already identified drifters that do not go below Rs[i+1]
    #     nanind = (distUse[:,1]<distUse[:,0]) + (ind2==-1)
    #     distUse[nanind,:] = np.nan
    #     tpUse[nanind,:] = np.nan

    #     # Do linear interpolation by hand because interp won't take in arrays
    #     rp = (Rs[i+1]-distUse[:,0])/(distUse[:,1] - distUse[:,0]) # weighting for higher side
    #     rm = 1 - rp # weighting for lower side

    #     # now find the interpolation time for each drifter
    #     time2 = rm*tpUse[:,0] + rp*tpUse[:,1]

    #     dt = time2 - time1 # in seconds
    #     dt /= 3600.*24 # in days

    #     nanind = np.isnan(dt)
    #     tau[i] = dt[~nanind].sum()
    #     nnans[i] = (~nanind).sum()

    # return tau, nnans, Rs

def plot():

    alpha = np.sqrt(2)

    # ----- all the lines ----- #

    # iWTX - 2007 - 02
    Files = glob('tracks/doturb2_ah5/*fsle.npz')
    fsle = np.zeros(20); nnans = np.zeros(20)
    for File in Files:
        # sum all values for this combination
        d = np.load(File)
        fsle += d['fsle']
        nnans += d['nnans']
        Rs = d['Rs']
        d.close()
    # average values for this combination
    # pdb.set_trace()
    l = np.log(alpha)/(fsle/nnans)

    ## ----- Make the plot ----- ##
    lcx = np.array([0.706, 1.009,1.447,1.996,2.822,4.013,5.668,7.964,11.373,16.228,22.713,31.856,45.782,63.837,91.029,127.415,180.191,256.303,362.240,509.652,727.706,])
    lcy = np.array([1.020,1.025,0.766,0.883,0.614,0.526,0.517,0.484,0.415,0.330,0.277,0.225,0.175,0.136,0.105,0.084,0.067,0.057,0.044,0.034,0.030])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(Rs, l, '*', color='darkcyan')
    ax.loglog(Rs[13:], 10*Rs[13:]**(-2/3), 'r')
    ax.loglog(lcx, lcy, 'rs')
    # ax1 = fig.add_subplot(2,1,1)
    # ax1.loglog(Rs, l200702W, '*', color='darkcyan', alpha=0.5)
    # ax1.loglog(Rs, l200706W, 'o', color='darkorange', alpha=0.5)
    # ax1.loglog(Rs, l200802W, 's', color='yellow', alpha=0.5)
    # ax1.loglog(Rs, l200806W, '^', color='purple', alpha=0.5)
    # ax2 = fig.add_subplot(2,1,2)
    # ax2.loglog(Rs, l200702E, '*', color='darkcyan', alpha=0.5)
    # ax2.loglog(Rs, l200706E, 'o', color='darkorange', alpha=0.5)
    # ax2.loglog(Rs, l200802E, 's', color='yellow', alpha=0.5)
    # ax2.loglog(Rs, l200806E, '^', color='purple', alpha=0.5)


def run():
    '''
    Run FSLE calculation for shelf transport drifter simulations.
    '''

    loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_nesting6.nc'
    grid = tracpy.inout.readgrid(loc)

    Files = glob('tracks/doturb2_ah5/*.nc')
    # fname = 'calcs/2007-02_WTXfsle.npz'

    for File in Files:

        fname = File[:-3] + 'fsle.npz'

        # if os.path.exists(fname): # don't redo if already done
        #     continue

        d = netCDF.Dataset(File)
        lonp = d.variables['lonp'][:]
        latp = d.variables['latp'][:]
        tp = d.variables['tp'][:]
        d.close()

        # let the index in axis 0 be the drifter id
        ID = np.arange(lonp.shape[0])

        # save pairs to save time since they are always the same
        if not os.path.exists('tracks/pairs.npz'):

            dist = np.zeros((lonp.shape[0],lonp.shape[0]))
            for idrifter in xrange(lonp.shape[0]):
                # dist contains all of the distances from other drifters for each drifter
                dist[idrifter,:] = get_dist(lonp[idrifter,0], lonp[:,0], latp[idrifter,0], latp[:,0])
            pairs = []
            for idrifter in xrange(lonp.shape[0]):
                ind = find(dist[idrifter,:]<=1)
                for i in ind:
                    if ID[idrifter] != ID[i]:
                        pairs.append([min(ID[idrifter], ID[i]), 
                                        max(ID[idrifter], ID[i])])

            pairs_set = set(map(tuple,pairs))
            pairs = map(list,pairs_set)# now pairs has only unique pairs of drifters
            pairs.sort() #unnecessary but handy for checking work
            np.savez('tracks/pairs.npz', pairs=pairs)
        else:
            pairs = np.load('tracks/pairs.npz')['pairs']

        # pdb.set_trace()


        # Loop over pairs of drifters from this area/time period and sum the FSLE, 
        # then average at the end

        dSave = np.zeros(20)
        nnans = np.zeros(20) # to collect number of non-nans over all drifters for a time
        for ipair in xrange(len(pairs)):

            dSavetemp, tSave = calc_fsle(lonp[pairs[ipair][0],:], latp[pairs[ipair][0],:], 
                                        lonp[pairs[ipair][1],:], latp[pairs[ipair][1],:], tp)
            ind = ~np.isnan(dSavetemp)
            dSave[ind] += dSavetemp[ind]
            nnans[~ind] += 1
            # fsle += fsletemp
            # nnans += nnanstemp

        # Save fsle for each file/area combination, NOT averaged
        np.savez(fname, fsle=fsle, nnans=nnans, Rs=Rs)
        print 'saved file', fname



if __name__ == "__main__":
    run()    
