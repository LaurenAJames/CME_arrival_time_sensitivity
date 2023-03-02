"""
This documents hold a number the definition of functions that have been used to identify ghost-front features in ensemble CME modelling using HUXt. These have been developed based on HIEnsembleHindcast work by Luke Barnard (2020).
"""

# Import libaries
import HUXt as H
import tables
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import pandas as pd
import os 
from astropy.time import Time, TimeDelta
from astropy.visualization import time_support
import scipy.stats as st
from scipy.interpolate import interp1d
import glob
import math
import sunpy.coordinates.sun as sn
import h5py

# Calculate elongation angles


### RUN AN ENSEMBLE ###

def run_huxt_ensemble(v_boundary,cr_num,cr_lon, n_ensemble=100, eventtag = " "):
    """
    Produce a determinisitc and ensemble of HUXt runs for a specified solar stormwatch event. For the deterministic run,
    both the full model solution, and the CME profile are saved in data>HUXt. For the ensemble, only the CME profiles
    are saved in data>HUXt, as this reduces the storage requirements significantly.
    Parameters
    ----------
    v_boundary = An array of solar wind speeds at the inner boundary.
    n_ensemble: Number of ensemble members to include, defaults to 200.
    tag = For uniquely identifying the files
    Returns
    -------
    A set of files in data>HUXt for the specified event.
    """
    # Get the carrington rotation number, and Earth's coordinates, at initial observation time.
#     cr_num = np.fix(sn.carrington_rotation_number(earth.time[0]))
#     ert = H.Observer('EARTH', earth.time[0])

#     print("Carrington rotation: {}".format(cr_num))
#     print("Earth Carrington Lon at init: {:3.2f}".format(ert.lon_c.to(u.deg)))
#     print("Earth HEEQ Lat at init: {:3.2f}".format(ert.lat.to(u.deg)))
    
    # Set up HUXt for a 5 day simulation
    vr_in = v_boundary
    model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=cr_lon *u.deg, simtime=5*u.day, dt_scale=4) 

    # Check units are correct
    
    # Deterministic run first:

    r_ib = model.r.min()
#    dt = ((r_ib - swpc_cme['r_obs']) / cme.v).to('s')
    thickness = 5 * u.solRad
    # Setup a ConeCME with these parameters
#     conecme = H.ConeCME(t_launch=dt, longitude=swpc_cme['lon'], latitude=swpc_cme['lat'],
#                         width=swpc_cme['width'], v=swpc_cme['v'], thickness=thickness)
    conecme = H.ConeCME(t_launch=0.0*u.day, longitude=10.0*u.deg, width=46*u.deg, v=498*(u.km/u.s), thickness=5*u.solRad)

    # Run HUXt with this ConeCME
    tag = "{}_{}".format(eventtag, 'deterministic')
    model.solve([conecme], save=True, tag=eventtag)
    
    # Now produce ensemble of HUXt runs with perturbed ConeCME parameters
    np.random.seed(1987)

    lon_spread = 10 * u.deg
    lat_spread = 10 * u.deg
    width_spread = 10 * u.deg
    v_spread = 50 * model.kms
    thickness_spread = 2 * u.solRad
    r_init_spread = 3 * u.solRad

    for i in range(n_ensemble):

        lon = 10.0*u.deg + np.random.uniform(-1, 1, 1)[0] * lon_spread
        lat = 0.0 *u.deg + np.random.uniform(-1, 1, 1)[0] * lat_spread
        width = width=46*u.deg + np.random.uniform(-1, 1, 1)[0] * width_spread
        v = 498*(u.km/u.s) + np.random.uniform(-1, 1, 1)[0] * v_spread
        thickness = 5.0*u.solRad + np.random.uniform(-1, 1, 1)[0] * thickness_spread

        # Workout time of CME at inner boundary, assuming fixed speed.
#         r_init = 0*u.solRad + np.random.uniform(-1, 1, 1)[0] * r_init_spread
#         r_ib = model.r.min()
#         dt = ((r_ib - r_init) / v).to('s')
        dt=0*u.s
        
        # Setup the ConeCME and run the model.
        conecme = H.ConeCME(t_launch=dt, longitude=lon, latitude=lat, width=width, v=v, thickness=thickness)
        tag = "{}_ensemble_{:02d}".format(eventtag, i)
        model.solve([conecme],save=True, tag=tag)
    
    return

# load cme file

# track cme features

# read in csv files (can this be done for all my csv files?)
# HI observation, ace, ensemble angles

# plot elongation-time profile (can this be for ensemble and deterministic run?)

# Compute Earth Arrival


