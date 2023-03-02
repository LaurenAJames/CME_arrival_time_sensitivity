# Import libraries
import HUXt as H
#import HIEnsembleHindcast as heh
#import HI_analysis as hip
import tables
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import pandas as pd
import os 
from astropy.time import Time, TimeDelta
import scipy.stats as st
import glob

# CME Event

# Solar Wind


        
def huxt_t_e_profile_fast(cme):
    """
    A faster, but riskier, version of computing the CME flank. Calculates the elongation to find the flank, and then uses
    spice to compute the position angle. Saves using spice to loop around the cme front, which is slow. This might fail for
    some geometries where the elongation is technically larger along PA angles not visible to either STA or STB. However, this agrees with
    huxt_t_e_profile for the deterministic runs, so I think is safe for the events in this study.
    """
    
#    times = Time([coord['time'] for i, coord in cme.coords.items()])
#    sta = H.Observer('STA', times)
#    stb = H.Observer('STB', times)

#    sta_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon', 'pa'])
#    stb_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon', 'pa'])

#    sta_profile['time'] = times.jd
#    stb_profile['time'] = times.jd

    for i, coord in cme.coords.items():

        if len(coord['r'])==0:
            continue

        r_sta = sta.r[i]
        x_sta = sta.r[i] * np.cos(sta.lat[i]) * np.cos(sta.lon[i])
        y_sta = sta.r[i] * np.cos(sta.lat[i]) * np.sin(sta.lon[i])
        z_sta = sta.r[i] * np.sin(sta.lat[i])

        r_stb = stb.r[i]
        x_stb = stb.r[i] * np.cos(stb.lat[i]) * np.cos(stb.lon[i])
        y_stb = stb.r[i] * np.cos(stb.lat[i]) * np.sin(stb.lon[i])
        z_stb = stb.r[i] * np.sin(stb.lat[i])


        lon_cme = coord['lon']
        lat_cme = coord['lat']
        r_cme = coord['r']

        x_cme = r_cme * np.cos(lat_cme) * np.cos(lon_cme)
        y_cme = r_cme * np.cos(lat_cme) * np.sin(lon_cme)
        z_cme = r_cme * np.sin(lat_cme)
        #############
        # Compute the observer CME distance, S, and elongation

        x_cme_s = x_cme - x_sta
        y_cme_s = y_cme - y_sta
        z_cme_s = z_cme - z_sta
        s = np.sqrt(x_cme_s**2 + y_cme_s**2 + z_cme_s**2)

        numer = (r_sta**2 + s**2 -r_cme**2).value
        denom = (2.0 * r_sta * s).value
        e_sta  = np.arccos(numer / denom )

        x_cme_s = x_cme - x_stb
        y_cme_s = y_cme - y_stb
        z_cme_s = z_cme - z_stb
        s = np.sqrt(x_cme_s**2 + y_cme_s**2 + z_cme_s**2)

        numer = (r_stb**2 + s**2 -r_cme**2).value
        denom = (2.0 * r_stb * s).value
        e_stb  = np.arccos(numer / denom )

        # Find the flank coordinate
        id_sta_flank = np.argmax(e_sta)
        id_stb_flank = np.argmax(e_stb)

        e_sta = e_sta[id_sta_flank]
        e_stb = e_stb[id_stb_flank]

        # STA PA first
       # r = r_cme[id_sta_flank].to('km').value
        #lon = lon_cme[id_sta_flank].to('rad').value
        #lat = lat_cme[id_sta_flank].to('rad').value
        #coords = np.array([r, lon, lat])
        #coord_hpr = spice.convert_lonlat(times[i], coords, 'HEEQ', 'HPR', observe_dst='STA', degrees=False)
        #e_sta_2 = coord_hpr[1]
        #pa_sta = coord_hpr[2]

        # STB PA
        #r = r_cme[id_stb_flank].to('km').value
        #lon = lon_cme[id_stb_flank].to('rad').value
        #lat = lat_cme[id_stb_flank].to('rad').value
        #coords = np.array([r, lon, lat])
        #coord_hpr = spice.convert_lonlat(times[i], coords, 'HEEQ', 'HPR', observe_dst='STB', degrees=False)
       # e_stb_2 = coord_hpr[1]
        #pa_stb = coord_hpr[2]
        
        #id_sta_fov = (pa_sta >= 0) & (pa_sta < np.pi)
        #if id_sta_fov & np.allclose(e_sta, e_sta_2):
        #    sta_profile.loc[i, 'lon'] = lon_cme[id_sta_flank].value
        #    sta_profile.loc[i, 'r'] = r_cme[id_sta_flank].value
        #    sta_profile.loc[i, 'el'] = np.rad2deg(e_sta_2)
        #    sta_profile.loc[i, 'pa'] = np.rad2deg(pa_sta)
        #else:
        #    sta_profile.loc[i, 'lon'] = np.NaN
        #    sta_profile.loc[i, 'r'] = np.NaN
        #    sta_profile.loc[i, 'el'] = np.NaN
        #    sta_profile.loc[i, 'pa'] = np.NaN

        #id_stb_fov = (pa_stb >= np.pi) & (pa_stb < 2*np.pi)
        #if id_stb_fov & np.allclose(e_stb, e_stb_2):
        #    stb_profile.loc[i, 'lon'] = lon_cme[id_stb_flank].value
        #    stb_profile.loc[i, 'r'] = r_cme[id_stb_flank].value
        #    stb_profile.loc[i, 'el'] = np.rad2deg(e_stb_2)
        #    stb_profile.loc[i, 'pa'] = np.rad2deg(pa_stb)
        #else:
        #    stb_profile.loc[i, 'lon'] = np.NaN
        #    stb_profile.loc[i, 'r'] = np.NaN
        #    stb_profile.loc[i, 'el'] = np.NaN
        #    stb_profile.loc[i, 'pa'] = np.NaN
            
    keys = ['lon', 'r', 'el', 'pa']
    sta_profile[keys] = sta_profile[keys].astype(np.float64)

    keys = ['lon', 'r', 'el', 'pa']
    stb_profile[keys] = stb_profile[keys].astype(np.float64)

    return sta_profile, stb_profile



def track_cme_flanks(ssw_event, fast=True):
    """
    Compute the CME flank elongation for each ensemble member and save to file.
    """
    
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "CME_*{}*_ensemble_*.hdf5".format(ssw_event))
    ensemble_files = glob.glob(path)
    n_ens = len(ensemble_files)

    # Produce a dictionary of keys of column headings for the dataframes 
    # storing the ensemble of time elonation profiles
    keys = []
    parameters = ['el', 'pa', 'r', 'lon']
    for param in parameters:
        for i in range(n_ens):
            keys.append("{}_{:02d}".format(param, i))

    keys = {k:0 for k in keys}

    # Loop over the ensemble files, pull out the elongation profiles and compute arrival time.
    for i, file in enumerate(ensemble_files):

        cme_list = H.load_cme_file(file)
        cme = cme_list[0] 

        # Compute the time-elongation profiles of the CME flanks from STA and STB,
        # and store into dataframes for each set of ensembles
        
        if fast:
            hxta, hxtb = huxt_t_e_profile_fast(cme)
        else:
            hxta, hxtb = huxt_t_e_profile(cme)
            
        if i == 0:    
            # Make pandas array to store all ensemble t_e_profiles.
            keys['time'] = hxta['time']
            ensemble_sta = pd.DataFrame(keys)
            ensemble_stb = pd.DataFrame(keys)

        # Update the ensemble dataframes
        for key in ['r', 'lon', 'el', 'pa']:
            e_key = "{}_{:02d}".format(key, i)
            ensemble_sta[e_key] = hxta[key]
            ensemble_stb[e_key] = hxtb[key]

    out_path = project_dirs['out_data']
    out_name = ssw_event + '_ensemble_sta.csv'
    ensemble_sta.to_csv(os.path.join(out_path, out_name))
    out_name = ssw_event + '_ensemble_stb.csv'
    ensemble_stb.to_csv(os.path.join(out_path, out_name))
    return

