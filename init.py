'''
Functions to initialize various numerical experiments.

Make a new init_* for your application.

loc     Path to directory of grid and output files
nsteps  Number of steps to do between model outputs (iter in tracmass)
ndays   number of days to track the particles from start date
ff      ff=1 to go forward in time and ff=-1 for backward in time
date    Start date in datetime object
tseas   Time between outputs in seconds
ah      Horizontal diffusion in m^2/s. 
        See project values of 350, 100, 0, 2000. For -turb,-diffusion
av      Vertical diffusion in m^2/s.
do3d    for 3d flag, do3d=0 makes the run 2d and do3d=1 makes the run 3d
doturb  turbulence/diffusion flag. 
        doturb=0 means no turb/diffusion,
        doturb=1 means adding parameterized turbulence
        doturb=2 means adding diffusion on a circle
        doturb=3 means adding diffusion on an ellipse (anisodiffusion)
lon0    Drifter starting locations in x/zonal direction.
lat0    Drifter starting locations in y/meridional direction.
z0/zpar Then z0 should be an array of initial drifter depths. 
        The array should be the same size as lon0 and be negative
        for under water. Currently drifter depths need to be above 
        the seabed for every x,y particle location for the script to run.
        To do 3D but start at surface, use z0=zeros(ia.shape) and have
         either zpar='fromMSL'
        choose fromMSL to have z0 starting depths be for that depth below the base 
        time-independent sea level (or mean sea level).
        choose 'fromZeta' to have z0 starting depths be for that depth below the
        time-dependent sea surface. Haven't quite finished the 'fromZeta' case.
        Then: 
        set z0 to 's' for 2D along a terrain-following slice
         and zpar to be the index of s level you want to use (0 to km-1)
        set z0 to 'rho' for 2D along a density surface
         and zpar to be the density value you want to use
         Can do the same thing with salinity ('salt') or temperature ('temp')
         The model output doesn't currently have density though.
        set z0 to 'z' for 2D along a depth slice
         and zpar to be the constant (negative) depth value you want to use
        To simulate drifters at the surface, set z0 to 's' 
         and zpar = grid['km']-1 to put them in the upper s level
         z0='s' is currently not working correctly!!!
         In the meantime, do surface using the 3d set up option but with 2d flag set
xp      x-locations in x,y coordinates for drifters
yp      y-locations in x,y coordinates for drifters
zp      z-locations (depths from mean sea level) for drifters
t       time for drifter tracks
name    Name of simulation to be used for netcdf file containing final tracks

'''

import numpy as np
import os
import netCDF4 as netCDF
import pdb
import glob
from datetime import datetime, timedelta
from matplotlib.mlab import *
import tracpy

units = 'seconds since 1970-01-01'

def disp(date, loc, grid=None):
    '''
    Initialization for seeding drifters at all shelf model grid points to be run
    forward.

    Optional inputs for making tests easy to run:
        date    Input date for name in datetime format
                e.g., datetime(2009, 11, 20, 0). If date not input,
                name will be 'temp' 
        grid    If input, will not redo this step. 
                Default is to load in grid.
    '''

    # Initialize parameters
    nsteps = 5 # 5 time interpolation steps
    ndays = 50
    ff = 1 # This is a forward-moving simulation

    # Time between outputs
    tseas = 4*3600 # 4 hours between outputs, in seconds, time between model outputs 
    ah = 0.
    av = 0. # m^2/s

    if grid is None:
        # if loc is the aggregated thredds server, the grid info is
        # included in the same file
        grid = tracpy.inout.readgrid(loc)
    else:
        grid = grid

    # Initial lon/lat locations for drifters
    # Start uniform array of drifters across domain using x,y coords
    # llcrnrlon = -92.25; urcrnrlon = -91.75; llcrnrlat = 29; urcrnrlat = 29.3; #C
    # llcrnrlon = -97; urcrnrlon = -96.5; llcrnrlat = 27; urcrnrlat = 27.5; #B
    llcrnrlon = -93.8; urcrnrlon = -92.2; llcrnrlat = 28; urcrnrlat = 29.2; # New
    xcrnrs, ycrnrs = grid['basemap']([llcrnrlon, urcrnrlon], [llcrnrlat, urcrnrlat])
    X, Y = np.meshgrid(np.arange(xcrnrs[0], xcrnrs[1], 700), 
                        np.arange(ycrnrs[0], ycrnrs[1], 700))
    # X, Y = np.meshgrid(np.arange(grid['xr'].min(),grid['xr'].max(),700), 
    #                     np.arange(grid['yr'].min(),grid['yr'].max(),700))
    lon0, lat0 = grid['basemap'](X, Y, inverse=True)

    # Eliminate points that are outside domain or in masked areas
    lon0, lat0 = tracpy.tools.check_points(lon0, lat0, grid)
    # pdb.set_trace()

    # Interpolate to get starting positions in grid space
    xstart0, ystart0, _ = tracpy.tools.interpolate2d(lon0, lat0, grid, 'd_ll2ij')

    # Initialize seed locations 
    ia = np.ceil(xstart0).astype(int) #[253]#,525]
    ja = np.ceil(ystart0).astype(int) #[57]#,40]

    np.savez('starting_locations.npz', lon0=lon0, lat0=lat0, xstart0=xstart0, ystart0=ystart0, ia=ia, ja=ja)
    # d = np.load('starting_locations.npz')
    # lon0 = d['lon0']; lat0=d['lat0']; xstart0=d['xstart0']; ystart0=d['ystart0'];
    # ia=d['ia']; ja=d['ja'];

    # lon0, lat0 already at cell centers
    # # Change to get positions at the center of the given cell
    # lon0, lat0, _ = tracpy.tools.interpolate2d(ia - 0.5, ja - 0.5, grid, 'm_ij2ll')
    N = 1 #lon0.size since there is only one drifter per box in this setup

    # surface drifters
    z0 = 's'  
    zpar = 29 

    # for 3d flag, do3d=0 makes the run 2d and do3d=1 makes the run 3d
    do3d = 0
    doturb = 0

    # Flag for streamlines. All the extra steps right after this are for streamlines.
    dostream = 0
    # convert date to number
    datenum = netCDF.date2num(date, units)
    # Number of model outputs to use
    tout = np.int((ndays*(24*3600))/tseas)
    # Figure out what files will be used for this tracking - to get tinds for
    # the following calculation
    nc, tinds = tracpy.inout.setupROMSfiles(loc, datenum, ff, tout)
    # Get fluxes at first time step in order to find initial drifter volume transport
    uf, vf, dzt, zrt, zwt  = tracpy.inout.readfields(tinds[0],grid,nc,z0,zpar)
    nc.close()
    # Initial total volume transport as a scalar quantity to be conserved, I think
    T0 = (abs(uf[ia, ja, 0]) + abs(vf[ia, ja, 0]))/N
    # # Initialize arrays of lon0, lat0 and U, V for full number of drifters
    # lon0 = np.ones(N,order='F')*lon0
    # lat0 = np.ones(N,order='F')*lat0
    # T0 = np.ones(N,order='F')*T0

    # Initialize the arrays to save the transports on the grid in the loop.
    # These arrays aggregate volume transport when a drifter enters or exits a grid cell
    # These should start at zero since we don't know which way things will travel yet
    U = np.ma.zeros(grid['xu'].shape,order='F')
    V = np.ma.zeros(grid['xv'].shape,order='F')

    # name = date.isoformat()[0:13] + '_doturb' + str(doturb) + '_ah' + str(int(ah)) 

    return nsteps, ndays, ff, tseas, ah, av, lon0, lat0, \
            z0, zpar, do3d, doturb, grid, dostream, N, T0.data, U, V
