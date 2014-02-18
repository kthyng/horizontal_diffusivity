'''
Script to run drifters backward from Galveston Bay to examine the Bay's 
connectivity with the shelf region.
'''

import matplotlib as mpl
mpl.use("Agg") # set matplotlib to use the backend that does not require a windowing system
import numpy as np
import os
import netCDF4 as netCDF
import pdb
import matplotlib.pyplot as plt
import tracpy
import init
from datetime import datetime, timedelta
import glob
from matplotlib.mlab import find


# function to compute great circle distance between point lat1 and lon1 
# and arrays of points given by lons, lats or both same length arrays
# Haversine formula
def get_dist(lon1,lons,lat1,lats): 
    lon1 = lon1*np.pi/180.
    lons = lons*np.pi/180.
    lat1 = lat1*np.pi/180.
    lats = lats*np.pi/180.

    earth_radius = 6373.
    distance = earth_radius*2.0*np.arcsin(np.sqrt(np.sin(0.50*(lat1-lats))**2 \
                                       + np.cos(lat1)*np.cos(lats) \
                                       * np.sin(0.50*(lon1-lons))**2))
    return distance

def calc_dispersion(name, grid=None, which='relative', r=1):
    '''
    Default is to look at relative dispersion (which='relative'), but
    also can do lagrangian dispersion, comparing with the mean (which='lagrangian').
    r is the radius for initial separation distance in km
    '''

    if grid is None:
        loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_nesting6.nc'
        grid = tracpy.inout.readgrid(loc)
    else:
        grid = grid

    # Read in tracks
    d = netCDF.Dataset(name)
    # d = netCDF.Dataset('tracks/2006-02-01T00C_doturb2_ah20.nc')
    lonp = d.variables['lonp'][:]
    latp = d.variables['latp'][:]
    t = d.variables['tp'][:]
    d.close()

    dist = np.zeros((lonp.shape[0],lonp.shape[0]))
    for idrifter in xrange(lonp.shape[0]):
        # dist contains all of the distances from other drifters for each drifter
        dist[idrifter,:] = get_dist(lonp[idrifter,0], lonp[:,0], latp[idrifter,0], latp[:,0])

    # let the index in axis 0 be the drifter id
    ID = np.arange(lonp.shape[0])

    # save pairs to save time since they are always the same
    if not os.path.exists('tracks/pairs.npz'):
        pairs = []
        for idrifter in xrange(lonp.shape[0]):
            ind = find(dist[idrifter,:]<=r)
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

    D2 = np.ones(lonp.shape[1])*np.nan
    nnans = np.zeros(lonp.shape[1]) # to collect number of non-nans over all drifters for a time
    for ipair in xrange(len(pairs)):
        if which == 'relative':
            dist = get_dist(lonp[pairs[ipair][0],:], lonp[pairs[ipair][1],:], 
                        latp[pairs[ipair][0],:], latp[pairs[ipair][1],:])
        elif which == 'lagrangian':
            dist = get_dist(lonp[pairs[ipair][0],:], lonp[pairs[ipair][1],:], 
                        latp[pairs[ipair][0],:], latp[pairs[ipair][1],:])
        D2 = np.nansum(np.vstack([D2, dist**2]), axis=0)
        nnans = nnans + ~np.isnan(dist)
    # D2 = D2.squeeze()/nnans #len(pairs) # average over all pairs

    # Distances squared, separately; times; number of non-nans for this set
    np.savez(name[:-3] + 'D2.npz', D2=D2, t=t, nnans=nnans)
    return D2, t, nnans
    # return D2, t

def run_dispersion():
    '''
    Run code to save dispersion calculations.
    '''

    # # Make sure necessary directories exist
    # if not os.path.exists('calcs'):
    #     os.makedirs('calcs')

    # Location of TXLA model output
    loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_nesting6.nc'

    grid = tracpy.inout.readgrid(loc)

    tests = glob.glob('tracks/doturb2_ah50_nsteps50') # types of simulations
    # tests = glob.glob('tracks/do*') # types of simulations

    for test in tests: # loop through types of simulations
        runs = glob.glob(test + '/*.nc')
        Dnameoverall = os.path.join(test, 'D2overall.npz')
        D2 = []; nnans = [];
        for run in runs: # loop through all the runs of that type
            D2name = run[:-3] + 'D2.npz'
            if not os.path.exists(D2name):
                D2_temp, t_temp, nnans_temp = calc_dispersion(run, grid)
                # have already summed but not averaged
            else:
                d = np.load(D2name)
                D2_temp = d['D2']; t = d['t']; nnans_temp = d['nnans'];
                d.close()
                # pdb.set_trace()
            D2.append(D2_temp)
            nnans.append(nnans_temp)

        # After I have run through all the times for this type of run, do average and save
        D2 = np.nansum(np.asarray(D2), axis=0) # sum the individual sums (already squared)
        nnans = np.nansum(np.asarray(nnans), axis=0) # sum non-nans for averages

        D2 = D2.squeeze()/nnans
        # save a sample time
        np.savez(Dnameoverall, D2=D2, t=t)

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

    # We know that drifters from the two sets have a one to one correspondence
    dist = get_dist(lonpc, lonp, latpc, latp)
    #pdb.set_trace()
    Rs = np.asarray([0.1*alpha**i for i in np.arange(28)])

    # Find first time dist>delta and dist>delta*alpha for each delta to
    # then linearly interpolate to find the corresponding time
    # FOR ONE DRIFTER TO START AND ONE DELTA
    tau = np.zeros(Rs.size)
    nnans = np.zeros(Rs.size) # not nans
    for idrifter in xrange(dist.shape[0]):
    # idrifter = 0
        # delta = Rs[10]
        for i, R in enumerate(Rs[:-1]):
            # print idrifter, i
    	    # pdb.set_trace()
            if R>=np.nanmin(dist[idrifter,:]) \
                and Rs[i+1]<=np.nanmax(dist[idrifter,:]):# \
                # and R>=dist[idrifter,:].any():

                ##  for delta ##

                # indices where separation is less than R
                iwhereless = find(dist[idrifter,:]<=R)

                # If there is more than 1 element
                if iwhereless.size>1:
                    # ind gives difference in instances of elements that are less than R
                    # to find if consecutive or not
                    ind =  np.diff(iwhereless)

                    # if there is more than one set of consecutive drifters
                    if (ind>1).sum():
                        iUse = iwhereless[find(ind>1)][0]
                    # otherwise there is just one set
                    else:
                        iUse = iwhereless[-1]
                else:
                    iUse = iwhereless


                # indA = find(dist[idrifter,:]>=R)[0]
                # indB = find(dist[idrifter,:]<=R)[-1]

                # Can't do this if iUse is the last index
                if not iUse==dist[idrifter,:].size-1:
                    # if i==0:
                    #     pdb.set_trace()
                    time1 = np.interp(R, dist[idrifter, iUse:iUse+2], tp[iUse:iUse+2])
                else:
                    time1 = np.nan
                # print R, dist[idrifter,ind-1:ind+1]


                ## for delta*alpha ##

                # indices where separation is less than R
                iwhereless = find(dist[idrifter,:]<=Rs[i+1])
                 # has to end up with greater time than for R
                iwhereless = iwhereless[find(iwhereless>=iUse)]

                # If there is more than 1 element
                if iwhereless.size>1:
                    # ind gives difference in instances of elements that are less than R
                    # to find if consecutive or not
                    ind =  np.diff(iwhereless)

                    # if there is more than one set of consecutive drifters
                    if (ind>1).sum():
                        iUse = iwhereless[find(ind>1)][0]
                    # otherwise there is just one set
                    else:
                        iUse = iwhereless[-1]
                else:
                    iUse = iwhereless


                # ind = find(dist[idrifter,:]>=Rs[i+1])[0]
                # Can't do this if iUse is the last index
                if not iUse==dist[idrifter,:].size-1:
                    time2 = np.interp(Rs[i+1], dist[idrifter, iUse:iUse+2], tp[iUse:iUse+2])
                # print Rs[i+1], dist[idrifter,ind-1:ind+1]
                else:
                    time2 = np.nan

                dt = time2-time1
                if dt<0:
                    pdb.set_trace()

            else:
                dt = np.nan

            if not np.isnan(dt):
                # print R, dt
                tau[i] += dt
                nnans[i] += 1 # counting not-nan entries for averaging later
    # pdb.set_trace()
    return tau, nnans, Rs

def run_fsle():
    '''
    Calculate FSLE for all possible pairs of drifters (just need to be coincident
    in time). 
    '''

    Files = glob.glob('tracks/doturb2_ah5/*.nc')

    for File in Files:

        fname = 'tracks/' + File[:-5].split('/')[-1] + 'fsle.npz'

        if os.path.exists(fname): # don't redo if already done
            continue

        d = netCDF.Dataset(File)
        lonp = d.variables['lonp'][:]
        latp = d.variables['latp'][:]
        tp = d.variables['tp'][:]
        d.close()

        # Save altogether for a single simulation
        fsle = np.zeros(28)
        nnans = np.zeros(28)
        ntrac = lonp.shape[0] # num drifters

        for i in xrange(ntrac-1): # loop over drifters
            # pdb.set_trace()
            fsletemp, nnanstemp, Rs = calc_fsle(lonp[i,:], latp[i,:], 
                                        lonp[i+1:,:], latp[i+1:,:], tp)
    	    # pdb.set_trace()
            fsle += fsletemp
            nnans += nnanstemp

            # NOT Now average all pairs starting at this unique location
            # fsle = fsle/nnans
        # pdb.set_trace()
        # save: fsle in time, averaged over all combinations of drifters starting at
        # a unique river input point for a unique starting time
        np.savez(fname, fsle=fsle, nnans=nnans, Rs=Rs)


def run():

    # Location of TXLA model output
    loc = 'http://barataria.tamu.edu:8080/thredds/dodsC/NcML/txla_nesting6.nc'

    # Make sure necessary directories exist
    if not os.path.exists('tracks'):
        os.makedirs('tracks')
    if not os.path.exists('figures'):
        os.makedirs('figures')

    grid = tracpy.inout.readgrid(loc)

    # Weekly Oct, Nov, Dec; biweekly Jan, Feb, Mar; monthly Apr, May, Jun, Jul
    # startdates = np.array([datetime(2010, 2, 1, 0, 1), datetime(2010, 2, 15, 0, 1),
    #                         datetime(2010, 3, 1, 0, 1), datetime(2010, 3, 15, 0, 1),
    #                         datetime(2010, 4, 1, 0, 1), datetime(2010, 5, 1, 0, 1),
    #                         datetime(2010, 6, 1, 0, 1), datetime(2010, 7, 1, 0, 1)])
    startdates = np.array([datetime(2009, 10, 1, 0, 1), datetime(2009, 10, 8, 0, 1),
                            datetime(2009, 10, 15, 0, 1), datetime(2009, 10, 22, 0, 1),
                            datetime(2009, 11, 1, 0, 1), datetime(2009, 11, 8, 0, 1),
                            datetime(2009, 11, 15, 0, 1), datetime(2009, 11, 22, 0, 1),
                            datetime(2009, 12, 1, 0, 1), datetime(2009, 12, 8, 0, 1),
                            datetime(2009, 12, 15, 0, 1), datetime(2009, 12, 22, 0, 1),
                            datetime(2010, 1, 1, 0, 1), datetime(2010, 1, 15, 0, 1),
                            datetime(2010, 2, 1, 0, 1), datetime(2010, 2, 15, 0, 1),
                            datetime(2010, 3, 1, 0, 1), datetime(2010, 3, 15, 0, 1),
                            datetime(2010, 4, 1, 0, 1), datetime(2010, 5, 1, 0, 1),
                            datetime(2010, 6, 1, 0, 1), datetime(2010, 7, 1, 0, 1)])

    # loop through state dates
    for startdate in startdates:

        date = startdate

        # Read in simulation initialization
        nstep, ndays, ff, tseas, ah, av, lon0, lat0, z0, zpar, do3d, doturb, \
                grid, dostream = init.disp(date, loc, grid=grid)
        # nstep, ndays, ff, tseas, ah, av, lon0, lat0, z0, zpar, do3d, doturb, \
        #         grid, dostream, N, T0, U, V = init.disp(date, loc, grid=grid)

        # for dt test:
        mod = 'doturb' + str(doturb) + '_ah' + str(int(ah)) + '_nsteps50/'
        # original test:
        # mod = 'doturb' + str(doturb) + '_ah' + str(int(ah)) + '/'

        if not os.path.exists('tracks/' + mod):
            os.makedirs('tracks/' + mod)
        if not os.path.exists('figures/' + mod):
            os.makedirs('figures/' + mod)

        name =  mod + date.isoformat()[0:13] 


        # If the particle trajectories have not been run, run them
        if not os.path.exists('tracks/' + name + '.nc'):

            # Run tracpy
            lonp, latp, zp, t, grid \
                = tracpy.run.run(loc, nstep, ndays, ff, date, tseas, ah, av, \
                                    lon0, lat0, z0, zpar, do3d, doturb, name, \
                                    grid=grid, dostream=dostream)
            # lonp, latp, zp, t, grid, T0, U, V \
            #     = tracpy.run.run(loc, nstep, ndays, ff, date, tseas, ah, av, \
            #                         lon0, lat0, z0, zpar, do3d, doturb, name, \
            #                         grid=grid, dostream=dostream, T0=T0, U=U, V=V)

        # # If basic figures don't exist, make them
        # if not os.path.exists('figures/' + name + '*.png'):

            # Read in and plot tracks
            d = netCDF.Dataset('tracks/' + name + '.nc')
            lonp = d.variables['lonp'][:]
            latp = d.variables['latp'][:]
            tracpy.plotting.tracks(lonp, latp, name, grid=grid)
            # tracpy.plotting.hist(lonp, latp, name, grid=grid, which='hexbin')
            d.close()
            # # Do transport plot
            # tracpy.plotting.transport(name='', fmod=date.isoformat()[0:13], 
            #     extraname=date.isoformat()[0:13], 
            #     Title='Transport on Shelf, for a week from ' + date.isoformat()[0:13], dmax=1.0)

   
        # # Do transport plot
        # tracpy.plotting.transport(name='', fmod=startdate.isoformat()[0:7] + '*', 
        #     extraname=startdate.isoformat()[0:7], Title='Transport on Shelf', dmax=1.0)


if __name__ == "__main__":
    run()    
