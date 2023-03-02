"""
Python script used to show how ghost-fronts can be modelled in HUXt, and how this muli-edge viewing can be used in ensemble modelling.
"""

# Import libaries
# As seen in HIEnsembleHindcast/ensemble_analysis.ipynb by L.Barnard ()
import HUXt as H
import HUXt_updated as H2
#import HIEnsembleHindcast as heh
#import HI_analysis as hip
import testing as TEST
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
import ensemble as ens
import matplotlib.ticker as ticker

#==========================================
# Functions for loading observations
#==========================================
def load_HI_obs():
    
    global front1a, front2a, HItimeA, front1b, front2b, front3b, HItimeB
    
    # STEREO-A
    #-----------
    df_a = pd.read_csv(r"ghost_data_stereo_a_dec2008.csv")
    df_a['time'] = pd.to_datetime(df_a['time'])                                                       
    front1a = pd.DataFrame({'time':df_a['time'],'el':df_a['front1_el'],'el_lo_error':df_a['front1_el_lo'], 'el_hi_error':df_a['front1_el_hi']})
    front2a = pd.DataFrame({'time':df_a['time'],'el':df_a['front2_el'],'el_lo_error':df_a['front2_el_lo'], 'el_hi_error':df_a['front2_el_hi']})
    HItimeA = df_a['time']

    # STEREO-B
    #-----------
    df_b = pd.read_csv(r"HCME_B__20081212_01_pa_slice_264_degrees.csv")
    df_b = df_b.drop(columns=['Unnamed: 0'])
    df_b['time'] = pd.to_datetime(df_b['time'])
    # Add error bar columns to dataframe
    df_b["el_lo_error"] = df_b["el"] - df_b["el_lo"]
    df_b["el_hi_error"] = df_b["el_hi"] - df_b["el"]

    # Create subset of dataframes for each front
    front1b = df_b[df_b['front'] == "Draw OUTER front"]
    front2b = df_b[df_b['front'] == "Draw INNER front"]
    front3b = df_b[df_b['front'] == "Draw THIRD front"]
    HItimeB = df_b['time'][df_b['front'] == "Draw INNER front"]

    # Delete columns that contain no new information
    front1b = front1b.drop(columns=['front'])
    front2b = front2b.drop(columns=['front'])
    front3b = front3b.drop(columns=['front'])
    
    return

def load_ACE_obs():
    # Load in ACE observation
    df_ace = pd.read_csv(r"ACE_SWBulk_201108_171208.csv")
    df_ace = df_ace.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time','SW_H_SPEED_km/s':'speed' })
    df_ace['time'] = pd.to_datetime(df_ace['time'])

    # Interp. CME arrival speed based upon model arrival time.
    ace_time = Time(df_ace['time'])
    ace_arrival = np.interp(cme.earth_arrival_time.jd,ace_time.jd,df_ace['speed']) * u.km/u.s

    # Interp. CME arrival speed based upon estimated arrival time.
    obs_arrival_time = Time("2008-12-16T07:00:00")
    obs_arrival_speed = np.interp(obs_arrival_time.jd,ace_time.jd,df_ace['speed']) * u.km/u.s

    print('Using model TOA, arrival speed is:', ace_arrival)
    print('Using observed TOA, arrival speed is:', obs_arrival_speed)

    # the impact of ACE's distance from Earth.
    aedis = 1500000 * u.km
    aspeed = ace_arrival
    #ACE - Earth distance / Speed of CME at ACE
    aetime = aedis/aspeed
    print("ACE to Earth time:", "%.2f" %aetime.to_value('h'),'hours')
    
    return

#==========================================
# Functions for running an single event
#==========================================

def huxt_t_e_profile_fast(cme):
    """
    Bases of Luke's code. Added the identifcation of the nose and the secondary flank.
    
    "Compute the time elongation profile of the flank of a ConeCME in HUXt, from both the STEREO-A or STEREO-B
    perspective. A faster, but less reliable, version of computing the CME flank with huxt_t_e_profile. Rather than
    using stereo_spice for the full calculation, which is a bit slow, this function does it's own calculation of the
    flank elongation, and then uses stereo_spice to compute the flank position angle. This might fail for some
    geometries where the elongation is technically larger along PA angles not visible to either STA or STB. However,
    this agrees with huxt_t_e_profile for the deterministic runs, so I think is safe for the events in this study.
    Parameters
    ----------
    cme: A ConeCME object from a completed HUXt run (i.e the ConeCME.coords dictionary has been populated).
    Returns
    -------
    sta_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STA's perspective, including the
                time, elongation, position angle, and HEEQ radius and longitude.
    stb_profile: Pandas dataframe giving the coordinates of the ConeCME flank from STB's perspective, including the
                time, elongation, position angle, and HEEQ radius and longitude."" 
    """
    # Gather information of body positions at each timestep
#     earth = model.get_observer('earth')
#     sta = model.get_observer('sta')
#     stb = model.get_observer('stb')
    times = Time([coord['time'] for i, coord in cme.coords.items()])
    sta = H.Observer('STA', times)
    stb = H.Observer('STB', times)

    # Create dataframe for storing the elogation profiles 
    sta_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'lon', 'r', 'el', 'lon_n', 'r_n','el_n' ,'lon_sec_flank', 'r_sec_flank','el_sec_flank'])
    stb_profile = pd.DataFrame(index=np.arange(times.size), columns=['time', 'lon', 'r', 'el', 'lon_n', 'r_n','el_n' ,'lon_sec_flank', 'r_sec_flank','el_sec_flank'])
    # format time 
    sta_profile['time'] = times.jd
    stb_profile['time'] = times.jd

    # Loop through all the boundary cooridinate points to work out the elongation angle from both STEREO-A and STEREO-B point of view
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
            lat_cme = 0 * u.deg
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
            e_sta_list  = np.arccos(numer / denom )

            x_cme_s = x_cme - x_stb
            y_cme_s = y_cme - y_stb
            z_cme_s = z_cme - z_stb
            s = np.sqrt(x_cme_s**2 + y_cme_s**2 + z_cme_s**2)

            numer = (r_stb**2 + s**2 -r_cme**2).value
            denom = (2.0 * r_stb * s).value
            e_stb_list  = np.arccos(numer / denom )

            # Find the flank coordinate
            id_sta_flank = np.argmax(e_sta_list)
            id_stb_flank = np.argmax(e_stb_list)

            e_sta = e_sta_list[id_sta_flank]
            e_stb = e_stb_list[id_stb_flank]

            sta_profile.loc[i, 'lon'] = lon_cme[id_sta_flank].value
            sta_profile.loc[i, 'r'] = r_cme[id_sta_flank].value
            sta_profile.loc[i, 'el'] = np.rad2deg(e_sta)

            stb_profile.loc[i, 'lon'] = lon_cme[id_stb_flank].value
            stb_profile.loc[i, 'r'] = r_cme[id_stb_flank].value
            stb_profile.loc[i, 'el'] = np.rad2deg(e_stb)

            # Find the nose coordinate (inc. identifying the front of the CME)
            lon_error = lon_cme.value - cme.longitude.value


            # Ensuring the correct boundary point is identifed for the nose by 1)sorting the error list in ascending order then 2)by using
            # the radius, ensuring the boundary point is on the leading edge of the CME rather than the back end of the event.
            # Here we use a radius difference greater than 10% to identify the boundary point is not on the front edge.
            idx = np.argsort(abs(lon_error)) 
            if r_cme[idx[0]] < r_cme[idx[1]]*0.9:
                id_nose = idx[1]
            # in the case in which three boundary points are identified to have nose like longtudinal errors, take the point with greastest radius 
            elif r_cme[idx[1]] < r_cme[idx[2]]*0.9 and r_cme[idx[0]] < r_cme[idx[2]]*0.9:      
                id_nose = idx[2]
            else:
                id_nose = idx[0]
            
            e_sta_nose = e_sta_list[id_nose]
            e_stb_nose = e_stb_list[id_nose]

            sta_profile.loc[i, 'lon_n'] = lon_cme[id_nose].value
            sta_profile.loc[i, 'r_n'] = r_cme[id_nose].value
            sta_profile.loc[i, 'el_n'] = np.rad2deg(e_sta_nose)

            stb_profile.loc[i, 'lon_n'] = lon_cme[id_nose].value
            stb_profile.loc[i, 'r_n'] = r_cme[id_nose].value
            stb_profile.loc[i, 'el_n'] = np.rad2deg(e_stb_nose)

            # Find the secondary flank coordinate 
            e_sta = e_sta_list[id_stb_flank]
            e_stb = e_stb_list[id_sta_flank]

            sta_profile.loc[i, 'lon_sec_flank'] = lon_cme[id_stb_flank].value
            sta_profile.loc[i, 'r_sec_flank'] = r_cme[id_stb_flank].value
            sta_profile.loc[i, 'el_sec_flank'] = np.rad2deg(e_sta)

            stb_profile.loc[i, 'lon_sec_flank'] = lon_cme[id_sta_flank].value
            stb_profile.loc[i, 'r_sec_flank'] = r_cme[id_sta_flank].value
            stb_profile.loc[i, 'el_sec_flank'] = np.rad2deg(e_stb)

    keys = ['lon', 'r', 'el', 'lon_n', 'r_n','el_n' ,'lon_sec_flank', 'r_sec_flank','el_sec_flank']
    sta_profile[keys] = sta_profile[keys].astype(np.float64)

    keys = ['lon', 'r', 'el', 'lon_n', 'r_n','el_n' ,'lon_sec_flank', 'r_sec_flank','el_sec_flank']
    stb_profile[keys] = stb_profile[keys].astype(np.float64)
            
    return sta_profile, stb_profile

def elongation_plot(cme, FOV,save=False, tag=''):
    """
    Plot elongation-time profiles from STEREO POV, inc HI observations.
    Paramaters
    FOV: either 'HI1' or 'HI2'
    save: if Ture, figure is saved to files
    tag: tag for file names
    
    Return:
    A 2x1 plot of the elongation-time profile for all the features from the viewpoint of 1) STEREO-A and 2) STEREO-B
    """
    load_HI_obs()
    sta_profile, stb_profile = huxt_t_e_profile_fast(cme)
    
    # Find plotting limits
    # HI-1: maximum elongation of 25˚
    if FOV == 'HI1':
        ymax = 25.0
        for i in range(len(sta_profile)):
            if sta_profile.el_n[i] < 30:
                FOVlimit_a = i
            if stb_profile.el_n[i] < 30:
                FOVlimit_b = i

    # HI-2: maximum elongation of ?˚
    elif FOV == 'HI2':
        ymax = np.max(sta_profile.el)
        FOVlimit_a = len(sta_profile)
        FOVlimit_b = len(stb_profile)
        
    else:
        print ("Field of view is not valid. Please use HI1 or HI2.")
    # this may need a try command.    

    # Format time  -  Here I am using the output from the model.getobserver() command. Would be simplier if I used sta_profile.time
                                                                        #     time_a = sta.time.to_value('datetime')[0:FOVlimit_a]
                                                                        #     time_b = stb.time.to_value('datetime')[0:FOVlimit_b]
    time_a = Time(sta_profile.time, format = 'jd').datetime[0:FOVlimit_a]
    time_b = Time(stb_profile.time, format = 'jd').datetime[0:FOVlimit_b]


    plt.rcParams.update({'font.size': 22, 'axes.labelsize':20, 'legend.fontsize':20,'xtick.labelsize': 16.0,'ytick.labelsize': 20.0,})

    # Plot figure
    fig, ax = plt.subplots(1, 2, figsize = [20,12])
    
    for nn, axs in enumerate(ax):
        locator = mdates.AutoDateLocator(minticks=12, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        axs.xaxis.set_major_locator(locator)
        axs.xaxis.set_major_formatter(formatter)
    
    #ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M'))
    ax[0].set_title('STEREO-A')
    ax[0].set_xlabel('Time', labelpad=30)
    ax[0].set_ylabel('Elongation (˚)')
    ax[0].set_ylim(top=ymax)                                 
    #axA.set_xlim(right=xmax)                                # Change to time
    ax[0].errorbar(HItimeA,front1a['el'],yerr=(front1a['el_lo_error'], front1a['el_hi_error']),
                fmt='.',color='pink', label='HI-1 front 1')
    ax[0].errorbar(HItimeA,front2a['el'],yerr=(front2a['el_lo_error'], front2a['el_hi_error']),
                fmt='.',color='skyblue', label='HI-1 front 2')
    #axA.errorbar(HItimeA,front3a['el'],yerr=(front3a['el_lo_error'], front3a['el_hi_error']),
    #             fmt='.',color='lightgreen', label='HI-1 front 3')
    ax[0].plot(time_a, sta_profile.el[0:FOVlimit_a],'k', label='Initial Flank')
    ax[0].plot(time_a, sta_profile.el_n[0:FOVlimit_a], 'k--', label='Nose')
    # axA.plot(time_a, sta_profile.el_sec_flank[0:FOVlimit_a], 'darkgrey', label='Secondary Flank')
#     ax[0].plot(HItimeA,interp_elA[0:FOVlimit_a],'k.')
#     ax[0].plot(HItimeA,interp_el_nA[0:FOVlimit_a],'k.')
    #axA.plot(HItimeA,interp_el_sec_flankaA,'.',color='darkgrey')
    
    #ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y %H:%M'))
    ax[1].set_title('STEREO-B')
    ax[1].set_xlabel('Time', labelpad=30)
    ax[1].set_ylabel('Elongation (˚)')
    ax[1].set_ylim(top=ymax)                                 
    #ax[1].set_xlim(right=60)                                # Change to time
    ax[1].errorbar(HItimeB,front1b['el'],yerr=(front1b['el_lo_error'], front1b['el_hi_error']),
                 fmt='.',color='pink', label='HI-1 front 1 (OUTER)')
    ax[1].errorbar(HItimeB,front2b['el'],yerr=(front2b['el_lo_error'], front2b['el_hi_error']),
                 fmt='.',color='skyblue', label='HI-1 front 2 (INNER)')
    ax[1].errorbar(HItimeB,front3b['el'],yerr=(front3b['el_lo_error'], front3b['el_hi_error']),
                 fmt='.',color='lightgreen', label='HI-1 front 3')
    ax[1].plot(time_b, stb_profile.el[0:FOVlimit_b], 'k', label='Initial Flank')
    ax[1].plot(time_b, stb_profile.el_n[0:FOVlimit_b], 'k--', label='Nose')
    ax[1].plot(time_b, stb_profile.el_sec_flank[0:FOVlimit_b], 'darkgrey', label='Secondary Flank')
#     ax[1].plot(HItimeB,interp_elB,'k.')
#     ax[1].plot(HItimeB,interp_el_nB,'k.')
#     ax[1].plot(HItimeB,interp_el_sec_flankB,'.',color='darkgrey')
    plt.setp( ax[0].xaxis.get_majorticklabels(), rotation=60 ) 
    plt.setp( ax[1].xaxis.get_majorticklabels(), rotation=60 )   
    #axB.legend(loc='best', bbox_to_anchor=[1.6,0.5])
    ax[1].legend(loc=[1.1,0.35], frameon=False)
    fig.tight_layout()

    if save:
        cr_num = np.int32(model.cr_num.value)
        filename = "HUXt_CR{:03d}_{}_{}_elongation_profile.png".format(cr_num, tag, FOV)
        filepath = os.path.join(model._figure_dir_, filename)            
        fig.savefig(filepath)
        
    return

def compute_earth_arrival(self,model, print_values=False, plot=False):
    """
    Function to compute arrival time at a set longitude and radius of the CME front.
    Tracks radial distance of front along a given longitude out past specified longitude.
    Then interpolates the r-t profile to find t_arr at arr_rad. And now this does the same for speed.

    Returns the transit time, arrival time, and arrival speed of the leading edge at Earth.
    If plot, returns a timeseries of the CME propogation that indicated the point of arrivial
    """

    # Uses elongation-profile to assign the time at which Earth's location is be calculated
    times = Time([coord['time'] for i, coord in self.coords.items()])
    #times = Time(sta_profile.time, format = 'jd')                
    ert = H.Observer('EARTH', times)

    # Need to force units to be the same to make interpolations work 
    arr_lon = 0*u.rad
    arr_rad = np.mean(ert.r)

    # Check if hit or miss.
    # Put longitude between -180 - 180, centered on CME lon.
    lon_diff = arr_lon - self.longitude
    if lon_diff < -180*u.deg:
        lon_diff += 360*u.deg
    elif lon_diff > 180*u.deg:
        lon_diff -= 360*u.deg

    cme_hw = self.width/2.0
    if (lon_diff >= -cme_hw) & (lon_diff <= cme_hw):
        # HIT, so get t-r profile along lon of interest.
        t_front = []
        r_front = []
        v_front = []

        radius = 1.0 * u.au
        lon = 0.0 * u.deg
        id_r = np.argmin(np.abs(model.r - radius))
        id_lon = np.argmin(np.abs(model.lon - lon))

        for i, coord in self.coords.items():

            if len(coord['r'])==0:
                continue

            #t_front.append(coord['model_time'].to('d').value)
            t_front.append(model.time_out[i].to(u.day).value)       # different to Luke's version
            v_front.append(model.v_grid_cme[i,id_r,id_lon].value)

            # Lookup radial coord at earth lon
            r = coord['r'].value
            lon = coord['lon'].value

            # Only keep front of cme
            id_front = r > np.mean(r)
            r = r[id_front]
            lon = lon[id_front]

            r_ans = np.interp(arr_lon.value, lon, r, period=2*np.pi)
            r_front.append(r_ans)
            # Stop when max r 
            if r_ans > arr_rad.value:
                break

        t_front = np.array(t_front)
        r_front = np.array(r_front)
        v_front = np.array(v_front)
        try:
            t_transit = np.interp(arr_rad.value, r_front, t_front)
            self.earth_transit_time = t_transit * u.d
            self.earth_arrival_time = times[0] + self.earth_transit_time
            v_atarrivial = np.interp(arr_rad.value, r_front, v_front)
            self.earth_arrival_speed = v_atarrivial * u.km/u.s
        except:
            self.earth_transit_time = np.NaN*u.d
            self.earth_arrival_time = Time('0000-01-01T00:00:00')
            self.earth_arrival_speed = np.NaN *u.km/u.s
    else:
        self.earth_transit_time = np.NaN*u.d
        self.earth_arrival_time = Time('0000-01-01T00:00:00')
        self.earth_arrival_speed = np.NaN *u.km/u.s

    if print_values:
        print("Leading edge transit time:", "%.4f" %self.earth_transit_time.value *u.d)
        print("Leading edge arrival time at Earth:", self.earth_arrival_time.iso)
        print("Leading edge arrival speed at Earth:", "%.4f" %self.earth_arrival_speed.value *u.km/u.s)

    if plot:
        model.plot_timeseries(1.0*u.AU, lon=0.0*u.deg,field='both', save=True, tag='1d_cone_test_radial')
        plt.plot(self.earth_transit_time,self.earth_arrival_speed,'rx',label="Leading Edge Arrival")
        plt.legend()
    return

def calculate_error(sta_profile, stb_profile):
    """
    This will calcuate the total residual error and RMSE of the model run in comparions to the 
    HI observations. To do this, the model results are interpolated to have the same time-stamp
    as the observations. 
    Params:
    --------
    sta_profile: The elongation profile of the features as seen from STEREO-A
    stb_profile: The elongation profile of the features as seen from STEREO-B
    
    Return:
    --------
    
    """
    # Check for HI data. Ensure it's been read in.
    load_HI_obs()
    
    # create dataframes to store error information for each timestep
    ts_errorA = pd.DataFrame()
    ts_errorB = pd.DataFrame()
    
    # create dataframes to store error information for the whole model run
    errorA = pd.DataFrame()
    errorB = pd.DataFrame()
    
    # interpret model values to have the same timestep as the HI observations
    # np.interp(x: where to interp., xp: x-coord. points of data, fp: y-coord. point of data)
    timeA = Time(HItimeA)
    interp_elA = np.interp(timeA.jd, sta_profile['time'], sta_profile['el'], left=float("NaN"), right=float("NaN"))
    interp_el_nA = np.interp(timeA.jd, sta_profile['time'], sta_profile['el_n'], left=float("NaN"), right=float("NaN"))
    #interp_el_sec_flankA = np.interp(timeA.jd, sta_profile['time'], sta_profile['el_sec_flank'], left=float("NaN"), right=float("NaN"))

    #timeB = Time(HItimeB)
    
    timeB = Time(front1b.time)
#     time2b = Time(front2b.time)
#     time3b = Time(front3b.time)
    interp_elB = np.interp(timeB.jd, stb_profile['time'], stb_profile['el'], left=float("NaN"), right=float("NaN"))
    interp_el_nB = np.interp(timeB.jd, stb_profile['time'], stb_profile['el_n'], left=float("NaN"), right=float("NaN"))
    interp_el_sec_flankB = np.interp(timeB.jd, stb_profile['time'], stb_profile['el_sec_flank'], left=float("NaN"), right=float("NaN"))
    
    # Make interpolated values NaN when obs. angle is NaN. This is to ensure residual error is calculated using the right values.
    frt1a=front1a.el.tolist()
    for i in range(len(frt1a)):
        if math.isnan(frt1a[i]):
            interp_elA[i] = float('NaN')

    frt2a=front2a.el.tolist()
    for i in range(len(frt2a)):
        if math.isnan(frt2a[i]):
            interp_el_nA[i] = float('NaN')

    # frt3a=front3a.el.tolist()
    # for i in range(len(frt3a)):
    #     if math.isnan(frt3a[i]):
    #         interp_el_sec_flankA[i] = float('NaN')

    frt1b=front1b.el.tolist()
    for i in range(len(frt1b)):
        if math.isnan(frt1b[i]):
            interp_elB[i] = float('NaN')

    frt2b=front2b.el.tolist()
    for i in range(len(frt2b)):
        if math.isnan(frt2b[i]):
            interp_el_nB[i] = float('NaN')

    frt3b=front3b.el.tolist()
    for i in range(len(frt3b)):
        if math.isnan(frt3b[i]):
            interp_el_sec_flankB[i] = float('NaN')
            
            
    # Save to dataframe
    ts_errorA['time'] = timeA
    ts_errorA['obs flank'] = front1a.el
    ts_errorA['interp flank'] = interp_elA
    ts_errorA['R2 flank'] = (ts_errorA['obs flank'] - ts_errorA['interp flank'])**2
    ts_errorA['obs nose'] = front2a.el
    ts_errorA['interp nose'] = interp_el_nA
    ts_errorA['R2 nose'] = (ts_errorA['obs nose'] - ts_errorA['interp nose'])**2
    ts_errorA['R2 two features'] = ts_errorA['R2 flank'] + ts_errorA['R2 nose']
    
    
    #residualB['time'] = front1b.time
    ts_errorB['time'] = front1b.time
    ts_errorB['obs flank'] = front1b.el
    ts_errorB['interp flank'] = interp_elB
    ts_errorB['R2 flank'] = (ts_errorB['obs flank'] - ts_errorB['interp flank'])**2
    ts_errorB['obs nose'] = frt2b
    ts_errorB['interp nose'] = interp_el_nB
    ts_errorB['R2 nose'] = (ts_errorB['obs nose'] - ts_errorB['interp nose'])**2
    ts_errorB['obs sec flank'] = frt3b
    ts_errorB['interp sec flank'] = interp_el_sec_flankB
    ts_errorB['R2 sec flank'] = (ts_errorB['obs sec flank'] - ts_errorB['interp sec flank'])**2
    ts_errorB['R2 two features'] = ts_errorB['R2 flank'] + ts_errorB['R2 nose']
#     ts_errorB['3 features'] = ts_errorB['R2 flank'] + ts_errorB['R2 nose'] + ts_errorB['R2 sec flank']
    
    # Total residual error -  saved as variable 
    tot_R2_flankA = np.nansum(ts_errorA['R2 flank'] )
    tot_R2_noseA = np.nansum(ts_errorA['R2 nose'] )
    #tot_R2_sec_flankA = np.nansum(R2_sec_flankA)
    tot_R2_two_featuresA = np.nansum(ts_errorA['R2 two features'])

    tot_R2_flankB = np.nansum(ts_errorB['R2 flank'])
    tot_R2_noseB = np.nansum(ts_errorB['R2 nose'])
    tot_R2_sec_flankB = np.nansum(ts_errorB['R2 sec flank'])
    tot_R2_two_featuresB = np.nansum(ts_errorB['R2 two features'])
    
    # Average residual for the run
    avg_error_flankA = tot_R2_flankA / (ts_errorA["interp flank"].count())
    avg_error_noseA = tot_R2_noseA / (ts_errorA["interp nose"].count())
    avg_error_two_featuresA = tot_R2_two_featuresA / (ts_errorA["R2 two features"].count())    # Double check this method. Is the length correct?
    
    avg_error_flankB = tot_R2_flankB / (ts_errorB["interp flank"].count())
    avg_error_noseB = tot_R2_noseB / (ts_errorB["interp nose"].count())
    avg_error_sec_flankB = tot_R2_sec_flankB / (ts_errorB["interp sec flank"].count())
    avg_error_two_featuresB = tot_R2_two_featuresB / (ts_errorB["R2 two features"].count())    # Double check this method. Is the length correct?
    
    rmse_flankA = np.sqrt(avg_error_flankA)
    rmse_noseA = np.sqrt(avg_error_noseA)
    rmse_two_featuresA = np.sqrt(avg_error_two_featuresA)
    
    rmse_flankB = np.sqrt(avg_error_flankB)
    rmse_noseB = np.sqrt(avg_error_noseB)
    rmse_sec_flankB = np.sqrt(avg_error_sec_flankB)
    rmse_two_featuresB = np.sqrt(avg_error_two_featuresB)
    
                                                                              
    # Store error values for whole model run in dataframe
    errorA = errorA.append({'flank':tot_R2_flankA, 
                            'nose':tot_R2_noseA,
                            'N & F': tot_R2_two_featuresA,
                            'RMSE flank': rmse_flankA, 
                            'RMSE nose': rmse_noseA,
                            'RMSE N & F': rmse_two_featuresA}, ignore_index=True)
    
    errorB = errorB.append({'flank':tot_R2_flankB, 
                            'nose':tot_R2_noseB, 
                            'sec flank': tot_R2_sec_flankB,
                            'N & F': tot_R2_two_featuresB,
                            'RMSE flank': rmse_flankB, 
                            'RMSE nose': rmse_noseB, 
                            'RMSE sec flank': rmse_sec_flankB,
                            'RMSE N & F': rmse_two_featuresB}, ignore_index=True)

    return ts_errorA, ts_errorB, errorA, errorB

def deterministic_error(cme,print_RMSE=True, plot=False, save=False):
    """
    
    """
    print("CME width:", cme.width)
    # Find the elongation profiles for the model run
    sta_profile, stb_profile = huxt_t_e_profile_fast(cme)
    
    # Find the error of the model run
    ts_errorA, ts_errorB, errorA, errorB = calculate_error(sta_profile, stb_profile)
    
    # -----------------------
    # Testing an unusual case. 
    if cme.width.value == 37.4065736506236:
        project_dirs = H._setup_dirs_()
        out_path = project_dirs['HUXt_data']
        out_nameA = 'TSerrorA_37.406_det.csv'
        ts_errorA.to_csv(os.path.join(out_path, out_nameA))
        out_nameB = '_TSerrorB_37.406_det.csv'
        ts_errorB.to_csv(os.path.join(out_path, out_nameB))
        
        out_nameC = 'ST-B_profile_37.406.csv'
        stb_profile.to_csv(os.path.join(out_path, out_nameC))
    # -----------------------
    
    if print_RMSE:
        display(errorA, errorB)
            
    if plot:
        # Plot a figure of the interp. values and the HI obs.
        fig, ((axA,axB)) = plt.subplots(1, 2, figsize = [15,6])
        
        axA.errorbar(HItimeA,front1a['el'],yerr=(front1a['el_lo_error'], front1a['el_hi_error']), fmt='.',color='pink', label='HI-1 front 1')
        axA.errorbar(HItimeA,front2a['el'],yerr=(front2a['el_lo_error'], front2a['el_hi_error']), fmt='.',color='skyblue', label='HI-1 front 2')
        axA.plot(HItimeA, ts_errorA['interp flank'], 'rx', label='flank')
        axA.plot(HItimeA, ts_errorA['interp nose'], 'bx', label='nose')
        axA.set_title('STEREO-A')
        axA.set_xlabel('Time', labelpad=30)
        
        timeB = front1b.time
        axB.errorbar(timeB,front1b['el'],yerr=(front1b['el_lo_error'], front1b['el_hi_error']), fmt='.',color='pink', label='HI-1 front 1 (OUTER)')
        axB.errorbar(timeB,front2b['el'],yerr=(front2b['el_lo_error'], front2b['el_hi_error']), fmt='.',color='skyblue', label='HI-1 front 2 (INNER)')
        axB.errorbar(timeB,front3b['el'],yerr=(front3b['el_lo_error'], front3b['el_hi_error']), fmt='.',color='lightgreen', label='HI-1 front 3')
        axB.plot(timeB, ts_errorB['interp flank'], 'rx', label='flank')
        axB.plot(timeB, ts_errorB['interp nose'], 'bx', label='nose')
        axB.plot(timeB, ts_errorB['interp sec flank'], 'gx', label='Sec. flank')
        axB.set_title('STEREO-B')
        axB.set_xlabel('Time', labelpad=30)
        
        plt.setp(axA.xaxis.get_majorticklabels(), rotation=60) 
        plt.setp(axB.xaxis.get_majorticklabels(), rotation=60)   
        axB.legend(loc=[1.1,0.35], frameon=False)
        
        plt.show()
        #=================
        #Plot figure of the residual squared error with time
        
        fig, ((axA,axB)) = plt.subplots(1, 2, figsize = [15,6])
                                                                                                                                                  
        axA.set_title('STEREO-A')
        axA.set_xlabel('Time', labelpad=30)
        axA.set_ylabel('$Residual^ 2$')
        axA.plot(HItimeA, ts_errorA['R2 flank'],'r-', label = "Flank")
        axA.plot(HItimeA, ts_errorA['R2 nose'],'b-', label= "Nose")
        #axA.plot(HItimeA, residualA['2 features'],'k-', label = "Flank & Nose")
        # axA.plot(tot_R2_sec_flankA,'g.', label= "Secondary Flank")
        # axA.plot(tot_R2_flankA + tot_R2_noseA + tot_R2_sec_flankA,'y.', label = "Flank & Nose & Sec. Flank")
        axA.set_yticks(np.arange(0, 8, 1))

        axB.set_title('STEREO-B')
        axB.set_xlabel('Time', labelpad=30)
        axB.set_ylabel('$Residual^ 2$')
        axB.plot(timeB, ts_errorB['R2 flank'],'r-', label = "Flank")
        axB.plot(timeB, ts_errorB['R2 nose'],'b-', label= "Nose")
        #axB.plot(timeB, ts_errorB['2 features'],'k-', label = "Flank & Nose")
        axB.plot(timeB, ts_errorB['R2 sec flank'],'-',color='limegreen', label= "Secondary Flank")
       #axB.plot(timeB, ts_errorB['3 features'],'-',color='orange', label = "Flank & Nose & Sec. Flank")
        axB.set_yticks(np.arange(0, 8, 1))
        
        plt.setp(axA.xaxis.get_majorticklabels(), rotation=60) 
        plt.setp(axB.xaxis.get_majorticklabels(), rotation=60) 
        axB.legend(loc=[1.1,0.35], frameon=False)

        fig.tight_layout()
        
    return errorA, errorB 
    
    
    
    
    
    
#==========================================
# Functions for running an ensemble
#==========================================

def run_huxt_ensemble(v_boundary, cr_num, cr_lon_init, n_ensemble=100, variable_test = False, variable='',param_space=0, event=''):
    """
    Produce a determinisitc and ensemble of HUXt runs for a specified solar stormwatch event. For the deterministic run,
    both the full model solution, and the CME profile are saved in data>HUXt. For the ensemble, only the CME profiles
    are saved in data>HUXt, as this reduces the storage requirements significantly.
    
    Parameters
    ----------
    v_boundary: the list of solar wind speed at inner boundary
    n_ensemble: Number of ensemble memebers, defaults to 100
    variable_test: True or False. Indicates whether you're only changing one variable.
    variable: If variable_test is True, assign the variable you want to change. Only "Width", "Longitude", "Latitude", "Speed", or "Thickness" are accepted.
    param_space: If variable_test is True, assign the parameter space of the variable including the units.
    event: Tag name
    
    Returns
    -------
    A set of files in data>HUXt for the specified event.
    """

    # Get the carrington rotation number, and Earth's coordinates, at SWPCs initial observation time.
    #     cr_num = np.fix(sn.carrington_rotation_number(swpc_cme['t_obs']))
    #     ert = H.Observer('EARTH', swpc_cme['t_obs'])
#     cr_num = np.fix(sn.carrington_rotation_number(earth.time[0]))
#     ert = H.Observer('EARTH', earth.time[0])
#     # This may currently need an inital run for the variables needed the above lines. 

#     print("Carrington rotation: {}".format(cr_num))
#     print("Earth Carrington Lon at init: {:3.2f}".format(ert.lon_c.to(u.deg)))
#     print("Earth HEEQ Lat at init: {:3.2f}".format(ert.lat.to(u.deg)))
    
    # Set up HUXt for a 5 day simulation
    vr_in = v_boundary
    #model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=ert.lon_c.to(u.deg), simtime=5*u.day, dt_scale=4)       # The same model run as I had coded originally
    model = H.HUXt(v_boundary=vr_in, cr_num=cr_num, cr_lon_init=cr_lon_init, simtime=5*u.day, dt_scale=4)       # The same model run as I had coded originally


    # Deterministic run first:
   #---------------------------
    # Get time of CME at inner boundary, assuming fixed speed.
    #r_ib = model.r.min()
    #dt = ((r_ib - swpc_cme['r_obs']) / cme.v).to('s')
    #thickness = 5 * u.solRad
    
    # Setup a ConeCME with these parameters
    conecme = H2.ConeCME(t_launch=0.0*u.day, longitude=10.0*u.deg, width=46*u.deg, v=498*(u.km/u.s), thickness=5*u.solRad)

    # Run HUXt with this ConeCME
    TAG = "{}_{}".format(event , 'deterministic')
    model.solve([conecme], save=True, tag=TAG)
    
    
    # Ensemble run:
#   ---------------------------    
    # Now produce ensemble of HUXt runs with perturbed ConeCME parameters
    np.random.seed(1987)
    
                                                #     lon_spread = 20 * u.deg
                                                #     lat_spread = 20 * u.deg
                                                #     width_spread = 20 * u.deg
                                                #     v_spread = 200 * model.kms
                                                #     thickness_spread = 2 * u.solRad
                                                #     r_init_spread = 3 * u.solRad

    lon_spread = 15 * u.deg
    lat_spread = 15 * u.deg
    width_spread = 20 * u.deg
    v_spread = 100 * model.kms
    thickness_spread = 2 * u.solRad
    r_init_spread = 0 * u.solRad
   
    # Testing an ensemble with changing parameter
    if variable_test:
        lon_spread = 0 * u.deg
        lat_spread = 0 * u.deg
        width_spread = 0 * u.deg
        v_spread = 0 * model.kms
        thickness_spread = 0 * u.solRad
        r_init_spread = 0 * u.solRad
        
        if variable == "Longitude":
            lon_spread = param_space
        elif variable == 'Latitude':
            lat_spread = param_space
        elif variable == 'Width':
            width_spread = param_space
        elif variable == 'Speed':
            v_spread = param_space
        elif variable == 'Thickness':
            thickness_spread = param_space
            
        # should put in a unit check here
    
    for i in range(n_ensemble):
        
        lon = 10.0*u.deg + np.random.uniform(-1, 1, 1)[0] * lon_spread
        lat = 0.0 *u.deg + np.random.uniform(-1, 1, 1)[0] * lat_spread
        width = 46*u.deg + np.random.uniform(-1, 1, 1)[0] * width_spread
        v = 498*(u.km/u.s) + np.random.uniform(-1, 1, 1)[0] * v_spread
        thickness = 5.0*u.solRad + np.random.uniform(-1, 1, 1)[0] * thickness_spread

        # Workout time of CME at inner boundary, assuming fixed speed.
#         r_init = 0*u.solRad + np.random.uniform(-1, 1, 1)[0] * r_init_spread
#         r_ib = model.r.min()
#         dt = ((r_ib - r_init) / v).to('s')
        dt=0*u.s
        
        # Setup the ConeCME and run the model.
        conecme = H2.ConeCME(t_launch=dt, longitude=lon, latitude=lat, width=width, v=v, thickness=thickness)
        tag = "{}_ensemble_{:02d}".format(event, i)
        model.solve([conecme],save=True, tag=tag)
    
    return

def run_huxt_ensemble_new(v_boundary,cr_num, cr_lon_init, longitude, latitude, width, v, thickness, t_launch=0.0*u.day, n_ensemble=100, map_inwards=False, r_outer='', r_inner='', variable_test = False, variable='', param_space=0, event=''):
    """
    Produce a determinisitc and ensemble of HUXt runs for a specified solar stormwatch event. For the deterministic run,
    both the full model solution, and the CME profile are saved in data>HUXt. For the ensemble, only the CME profiles
    are saved in data>HUXt, as this reduces the storage requirements significantly.
    
    Parameters
    ----------
    v_boundary: the list of solar wind speed at inner boundary
    n_ensemble: Number of ensemble memebers, defaults to 100
    variable_test: True or False. Indicates whether you're only changing one variable.
    variable: If variable_test is True, assign the variable you want to change. Only "Width", "Longitude", "Latitude", "Speed", or "Thickness" are accepted.
    param_space: If variable_test is True, assign the parameter space of the variable including the units.
    event: Tag name
    
    Returns
    -------
    A set of files in data>HUXt for the specified event.
    """

    # Get the carrington rotation number, and Earth's coordinates, at SWPCs initial observation time.
                                                                    #     cr_num = np.fix(sn.carrington_rotation_number(swpc_cme['t_obs']))
                                                                    #     ert = H.Observer('EARTH', swpc_cme['t_obs'])
    #cr_num = np.fix(sn.carrington_rotation_number(earth.time[0]))
    #ert = H.Observer('EARTH', earth.time[0])
    # This may currently need an inital run for the variables needed the above lines. 

    #print("Carrington rotation: {}".format(cr_num))
    #print("Earth Carrington Lon at init: {:3.2f}".format(ert.lon_c.to(u.deg)))
    #print("Earth HEEQ Lat at init: {:3.2f}".format(ert.lat.to(u.deg)))
    
    # Set up HUXt for a 5 day simulation
    if map_inwards == True:
        vr_inner = H.map_v_boundary_inwards(v_outer=v_boundary, r_outer=r_outer, r_inner=r_inner)
        model = H.HUXt(v_boundary=vr_inner , cr_num=cr_num, cr_lon_init=cr_lon_init, simtime=5*u.day, dt_scale=4, r_min=r_inner)      
    else:
        model = H.HUXt(v_boundary=v_boundary, cr_num=cr_num, cr_lon_init=cr_lon_init, simtime=5*u.day, dt_scale=4)
        
    # Deterministic run first:
    #---------------------------
    # Get time of CME at inner boundary, assuming fixed speed.
    #r_ib = model.r.min()
    #dt = ((r_ib - swpc_cme['r_obs']) / cme.v).to('s')
    #thickness = 5 * u.solRad
    
    # Setup a ConeCME with these parameters
    conecme = H2.ConeCME(t_launch=t_launch, longitude=longitude, latitude=latitude, width=width, v=v, thickness=thickness)

    # Run HUXt with this ConeCME
    TAG = "{}_{}".format(event , 'deterministic')
    model.solve([conecme], save=True, tag=TAG)
    
    
    # Ensemble run:
#   ---------------------------    
    # Now produce ensemble of HUXt runs with perturbed ConeCME parameters
    np.random.seed(1987)

    lon_spread = 15 * u.deg
    lat_spread = 15 * u.deg
    width_spread = 20 * u.deg
    v_spread = 100 * model.kms
    thickness_spread = 2 * u.solRad
    r_init_spread = 0 * u.solRad
   
    # Testing an ensemble with changing parameter
    if variable_test:
        lon_spread = 0 * u.deg
        lat_spread = 0 * u.deg
        width_spread = 0 * u.deg
        v_spread = 0 * model.kms
        thickness_spread = 0 * u.solRad
        r_init_spread = 0 * u.solRad
        
        if variable == "Longitude":
            lon_spread = param_space
        elif variable == 'Latitude':
            lat_spread = param_space
        elif variable == 'Width':
            width_spread = param_space
        elif variable == 'Speed':
            v_spread = param_space
        elif variable == 'Thickness':
            thickness_spread = param_space
            
        # should put in a unit check here
    
    for i in range(n_ensemble):
        
        lon = longitude + np.random.uniform(-1, 1, 1)[0] * lon_spread
        lat = latitude + np.random.uniform(-1, 1, 1)[0] * lat_spread
        width = width + np.random.uniform(-1, 1, 1)[0] * width_spread
        v = v + np.random.uniform(-1, 1, 1)[0] * v_spread
        thickness = thickness + np.random.uniform(-1, 1, 1)[0] * thickness_spread

        # Workout time of CME at inner boundary, assuming fixed speed.
        #r_init = 0*u.solRad + np.random.uniform(-1, 1, 1)[0] * r_init_spread
        #r_ib = model.r.min()
        #dt = ((r_ib - r_init) / v).to('s')
        dt=0*u.s
        
        print(i,':', width)
        # Setup the ConeCME and run the model.
        conecme = H2.ConeCME(t_launch=dt, longitude=lon, latitude=lat, width=width, v=v, thickness=thickness)
        tag = "{}_ensemble_{:02d}".format(event, i)
        model.solve([conecme],save=True, tag=tag)
    
    return

def load_cme_file(filepath):
    """
    Load in data from a saved HUXt run.

    :param filepath: The full path to a HDF5 file containing the output from HUXt.save()
    :return: cme_list: A list of instances of ConeCME
    :return: model: An instance of HUXt containing loaded results.
    """
    if os.path.isfile(filepath):

        data = h5py.File(filepath, 'r')

        # Create list of the ConeCMEs
        cme_list = []
        all_cmes = data['ConeCMEs']
        for k in all_cmes.keys():
            cme_data = all_cmes[k]
            t_launch = cme_data['t_launch'][()] * u.Unit(cme_data['t_launch'].attrs['unit'])
            lon = cme_data['longitude'][()] * u.Unit(cme_data['longitude'].attrs['unit'])
            lat = cme_data['latitude'][()] * u.Unit(cme_data['latitude'].attrs['unit'])
            width = cme_data['width'][()] * u.Unit(cme_data['width'].attrs['unit'])
            thickness = cme_data['thickness'][()] * u.Unit(cme_data['thickness'].attrs['unit'])
            thickness = thickness.to('solRad')
            v = cme_data['v'][()] * u.Unit(cme_data['v'].attrs['unit'])
            cme = H2.ConeCME(t_launch=t_launch, longitude=lon, latitude=lat, v=v, width=width, thickness=thickness)

            # Now sort out coordinates.
            # Use the same dictionary structure as defined in ConeCME._track_2d_
            coords_group = cme_data['coords']
            coords_data = {j: {'time':np.array([]), 'model_time': np.array([]) * u.s,
                               'lon_pix': np.array([]) * u.pix, 'r_pix': np.array([]) * u.pix,
                               'lon': np.array([]) * u.rad, 'r': np.array([]) * u.solRad,
                              'lat': np.array([]) * u.rad}
                               for j in range(len(coords_group))}

            for time_key, pos in coords_group.items():
                t = np.int(time_key.split("_")[2])
                coords_data[t]['time'] = Time(pos['time'][()], format="isot")
                coords_data[t]['model_time'] = pos['model_time'][()] * u.Unit(pos['model_time'].attrs['unit'])
                coords_data[t]['lon_pix'] = pos['lon_pix'][()] * u.Unit(pos['lon_pix'].attrs['unit'])
                coords_data[t]['r_pix'] = pos['r_pix'][()] * u.Unit(pos['r_pix'].attrs['unit'])
                coords_data[t]['lon'] = pos['lon'][()] * u.Unit(pos['lon'].attrs['unit'])
                coords_data[t]['r'] = pos['r'][()] * u.Unit(pos['r'].attrs['unit'])
                coords_data[t]['lat'] = pos['lat'][()] * u.Unit(pos['lat'].attrs['unit'])
                
            cme.coords = coords_data
            
            cme.compute_earth_arrival()
            cme_list.append(cme)

    else:
        # File doesnt exist return nothing
        print("Warning: {} doesnt exist.".format(filepath))
        cme_list = []

    return cme_list

def track_cme_flanks(fast=True, tag=''):
    """
    Compute the CME flank elongation for each ensemble member and save to file.
    Parameters
    ----------
    fast: Boolean, default True, of whether to use a faster version of the flank tracking algorithm. Saves a
          significant amount of time, and works for the events studied here. Might not generalise well to other events.
    Returns
    -------
    arrivial_info: global. Dataframe of the arrival time, speed, and transit time of the ensemble CMEs
    Files in data>out_data, with name format ssw_event_ensemble_sta.csv and ssw_event_ensemble_stb.csv
    """
    # Find file path of ensemble files 
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "HUXt_CR2077_*{}*_ensemble*.hdf5".format(tag))
    ensemble_files = glob.glob(path)
    n_ens = len(ensemble_files)

    # Produce a dictionary of keys of column headings for the dataframes 
    # storing the ensemble of time elonation profiles
    keys = []
    parameters = ['lon', 'r', 'el', 'lon_n', 'r_n','el_n' ,'lon_sec_flank', 'r_sec_flank','el_sec_flank']
    for param in parameters:
        for i in range(n_ens):
            keys.append("{}_{:02d}".format(param, i))       

    keys = {k: 0 for k in keys}   # I don't know what this line does

    # Loop over the ensemble files, pull out the elongation profiles.
    for i, file in enumerate(ensemble_files):
        
        cme_list = load_cme_file(file)
        cme = cme_list[0] 

        # Compute the time-elongation profiles of the CME features from STA and STB,
        # and store into dataframes for each set of ensembles
        
        if fast:
            hxta, hxtb = huxt_t_e_profile_fast(cme)
        else:
            hxta, hxtb = huxt_t_e_profile(cme)       # Don't have this module
     
        if i == 0:    
            # Make pandas array to store all ensemble t_e_profiles.
            keys['time'] = hxta['time']
            ensemble_sta = pd.DataFrame(keys)
            ensemble_stb = pd.DataFrame(keys)

        # Update the ensemble dataframes
        for key in ['lon', 'r', 'el', 'lon_n', 'r_n','el_n' ,'lon_sec_flank', 'r_sec_flank','el_sec_flank']:
            e_key = "{}_{:02d}".format(key, i)
            ensemble_sta[e_key] = hxta[key]
            ensemble_stb[e_key] = hxtb[key]
            

    out_path = project_dirs['HUXt_data']
    out_name = tag + '_ensemble_sta.csv'
    ensemble_sta.to_csv(os.path.join(out_path, out_name))
    out_name = tag + '_ensemble_stb.csv'
    ensemble_stb.to_csv(os.path.join(out_path, out_name))

    return 

def ensemble_cme_dataframes(n_ens=100, fast=True, tag=''):
    """
    
    
    """
    
    # Produce data-frames to store information
    #----------------------------------------------
    
    # CME INITIAL PARAMETERS
    cme_params = pd.DataFrame()
    
    # ERROR INFORMATION
    ensemble_errorA = pd.DataFrame()
    ensemble_errorB = pd.DataFrame()
    
    # ARRIVAL INFORMATION
    arrival_info = pd.DataFrame()
    
    
    # Deterministic event
    #----------------------------------------------
    # Find file path 
    runtag = "deterministic"
    project_dirs = H._setup_dirs_()
    path = os.path.join(project_dirs['HUXt_data'], "HUXt_CR2077_{}_deterministic.hdf5".format(tag))
    filepath = glob.glob(path)[0]
    cme_list = load_cme_file(filepath)
    cme = cme_list[0]
    
    # Store init params
    cme_params = cme_params.append({"Ensemble Run":runtag, 
                                    "Time of launch": cme.t_launch, 
                                    "Longitude (˚)":cme.longitude.to(u.deg).value,
                                    "Latitude (˚)":cme.latitude.to(u.deg).value, 
                                    "Speed (km/s)": cme.v.value,
                                    "Width (˚)": cme.width.value,
                                    "Thickness (Rs)":cme.thickness.value}, ignore_index=True)
    
    # error calculations
    if fast:
        hxta, hxtb = huxt_t_e_profile_fast(cme)
    else:
        hxta, hxtb = huxt_t_e_profile(cme)       # Don't have this module
    
    
    ts_errorA, ts_errorB, errorA, errorB = calculate_error(hxta, hxtb)
    ensemble_errorA =  ensemble_errorA.append({'file': runtag, 
                                                'total flank error':errorA['flank'][0],
                                                'total nose error':errorA['nose'][0],
                                                'total sec flank error': float('NaN'),
                                                'total N & F error': float('NaN'),
                                                'RMSE flank error': errorA['RMSE flank'][0],
                                                'RMSE nose error': errorA['RMSE nose'][0],
                                                'RMSE sec flank error': float('NaN'),
                                                'RMSE N & F error': float('NaN')},ignore_index = True)
    ensemble_errorB =  ensemble_errorB.append({'file': runtag, 
                                                'total flank error':errorB['flank'][0],
                                                'total nose error':errorB['nose'][0],
                                                'total sec flank error': errorB['sec flank'][0],
                                                'total N & F error': errorB['N & F'][0],
                                                'RMSE flank error': errorB['RMSE flank'][0],
                                                'RMSE nose error': errorB['RMSE nose'][0],
                                                'RMSE sec flank error': errorB['RMSE sec flank'][0],
                                                'RMSE N & F error': errorB['RMSE N & F'][0]},ignore_index = True)

    # arrival information
    compute_earth_arrival(cme, print_values=False, plot=False)
    arrival_info = arrival_info.append({"file": runtag,
                                        "Transit Time" : cme.earth_transit_time.value, 
                                        "Arrival Time" : cme.earth_arrival_time.jd,
                                        "Arrival Speed" : cme.earth_arrival_speed.value}, ignore_index=True)
    
    
    
    # Ensemble events
    #----------------------------------------------
    for i in range(n_ens):
        runtag = "ensemble_{:02d}".format(i)
        path = os.path.join(project_dirs['HUXt_data'], "HUXt_CR2077_{}_{}.hdf5".format(tag,runtag))
        filepath = glob.glob(path)[0]
        cme_list = load_cme_file(filepath)
        cme = cme_list[0]

        # Store init params
        cme_params = cme_params.append({"Ensemble Run":runtag, 
                                        "Time of launch": cme.t_launch, 
                                        "Longitude (˚)":cme.longitude.to(u.deg).value,
                                        "Latitude (˚)":cme.latitude.to(u.deg).value, 
                                        "Speed (km/s)": cme.v.value,
                                        "Width (˚)": cme.width.value,
                                        "Thickness (Rs)":cme.thickness.value}, ignore_index=True)
        # error calculations
        if fast:
            hxta, hxtb = huxt_t_e_profile_fast(cme)
        else:
            hxta, hxtb = huxt_t_e_profile(cme)       # Don't have this module
    
        ts_errorA, ts_errorB, errorA, errorB = calculate_error(hxta, hxtb)
        ensemble_errorA =  ensemble_errorA.append({'file': runtag, 
                                                    'total flank error':errorA['flank'][0],
                                                    'total nose error':errorA['nose'][0],
                                                    'total sec flank error': float('NaN'),
                                                    'total N & F error': errorA['N & F'][0],
                                                    'RMSE flank error': errorA['RMSE flank'][0],
                                                    'RMSE nose error': errorA['RMSE nose'][0],
                                                    'RMSE sec flank error': float('NaN'),
                                                    'RMSE N & F error': errorA['RMSE N & F'][0],},ignore_index = True)
        ensemble_errorB =  ensemble_errorB.append({'file': runtag, 
                                                    'total flank error':errorB['flank'][0],
                                                    'total nose error':errorB['nose'][0],
                                                    'total sec flank error': errorB['sec flank'][0],
                                                    'total N & F error': errorB['N & F'][0],
                                                    'RMSE flank error': errorB['RMSE flank'][0],
                                                    'RMSE nose error': errorB['RMSE nose'][0],
                                                    'RMSE sec flank error': errorB['RMSE sec flank'][0],
                                                    'RMSE N & F error': errorB['RMSE N & F'][0]},ignore_index = True)

        # arrival information
        compute_earth_arrival(cme, print_values=False, plot=False)
        arrival_info = arrival_info.append({"file": runtag,
                                            "Transit Time" : cme.earth_transit_time.value, 
                                            "Arrival Time" : cme.earth_arrival_time.jd,
                                            "Arrival Speed" : cme.earth_arrival_speed.value}, ignore_index=True) 
    # End of ensemble loop
    
    
    # Check files for NaN run. NaN elongation runs can occur due to the combination in inital parameters. 
    # If the nose and flank elongtion is NaN, add the index value to a list. Then, drop these values from the dataframe. 
    NaN_runA = []
    is_NaN_flanka = ensemble_errorA["RMSE flank error"].isnull()
    is_NaN_nosea = ensemble_errorA["RMSE nose error"].isnull()
    for i in range(len(ensemble_errorA)):
        if is_NaN_flanka[i]==True & is_NaN_nosea[i]==True:
            NaN_runA.append(i)
            
    NaN_runB = []       
    is_NaN_flankb = ensemble_errorB["RMSE flank error"].isnull()
    is_NaN_noseb = ensemble_errorB["RMSE nose error"].isnull()
    for j in range(len(ensemble_errorB)):
        if is_NaN_flankb[j]==True & is_NaN_noseb[j]==True:
            NaN_runB.append(j)
    
    for k in range(len(NaN_runA)):
        if NaN_runA[k] == NaN_runB[k]:
            ensemble_errorA = ensemble_errorA.drop([NaN_runA[k]])
            ensemble_errorB = ensemble_errorB.drop([NaN_runA[k]])
            arrival_info = arrival_info.drop([NaN_runA[k]])
    


    # Save files
    out_path = project_dirs['HUXt_data']

    out_name = tag + '_cme_params.csv'
    cme_params.to_csv(os.path.join(out_path, out_name))
    
    out_name = tag + '_ensemble_errorA.csv'
    ensemble_errorA.to_csv(os.path.join(out_path, out_name))
    out_name = tag + '_ensemble_errorB.csv'
    ensemble_errorB.to_csv(os.path.join(out_path, out_name))
    
    out_name = tag + '_arrival_info.csv'
    arrival_info.to_csv(os.path.join(out_path, out_name))
    
    return 

def load_csv_file(file, tag=''):
    
    project_dirs = H._setup_dirs_()
    
    if file == "elongation profiles":
        pathA = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_sta.csv")
        ens_profileA = pd.read_csv(r"{}".format(pathA))
        ens_profileA = ens_profileA.drop(columns=['Unnamed: 0'])

        pathB = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_stb.csv")
        ens_profileB = pd.read_csv(r"{}".format(pathB))
        ens_profileB = ens_profileB.drop(columns=['Unnamed: 0'])
        
        data = ens_profileA, ens_profileB
        
    if file == "errors":
        pathA = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_errorA.csv")
        ens_errorA = pd.read_csv(r"{}".format(pathA))
        ens_errorA = ens_errorA.drop(columns=['Unnamed: 0'])
        pathB = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_errorB.csv")
        ens_errorB = pd.read_csv(r"{}".format(pathB))
        ens_errorB = ens_errorB.drop(columns=['Unnamed: 0'])
        
        data = ens_errorA, ens_errorB

    if file == "initial parameters":
        path = os.path.join(project_dirs['HUXt_data'], tag+"_cme_params.csv")
        cme_parameters = pd.read_csv(r"{}".format(path))
        cme_parameters = cme_parameters.drop(columns=['Unnamed: 0'])
        
        data = cme_parameters
        
    if file == "arrival information":
        path = os.path.join(project_dirs['HUXt_data'], tag+"_arrival_info.csv")
        arrival_info = pd.read_csv(r"{}".format(path))
        arrival_info = arrival_info.drop(columns=['Unnamed: 0'])
        
        data = arrival_info
        
    if file == "best fit run":
        path = os.path.join(project_dirs['HUXt_data'], tag+"_bestfitA.csv")
        best_fit_runA = pd.read_csv(r"{}".format(path))
        best_fit_runA = best_fit_runA.drop(columns=['Unnamed: 0'])
        
        path = os.path.join(project_dirs['HUXt_data'], tag+"_bestfitB.csv")
        best_fit_runB = pd.read_csv(r"{}".format(path))
        best_fit_runB = best_fit_runB.drop(columns=['Unnamed: 0'])
        
        data = best_fitA, best_fitB
    
    return data


def ensemble_error(n_ens=100, save=False, tag=''):
    """
    Calculates the error between the ensemble runs and the HI observations. 
    
    Params:
    --------
    n_ens: Number of ensemble members
    save: True or False. Saves dataframes to local files.
    tag: Tag name of event
    
    Return:
    ------
    ensemble_errorA : Dataframe of the total residual-squared error and RMSE of each features, as seen from STEREO-A
    ensemble_errorB : Dataframe of the total residual-squared error and RMSE of each features, as seen from STEREO-B
    
    """
    
    # load in elongation profiles for the ensemble runs 
    #df_ensembleA, df_ensembleB, ens_timeA, ens_timeB = load_elongation_data(tag=tag)
    
    # Create dataframe for storing residual information
    ensemble_errorA = pd.DataFrame()
    ensemble_errorB = pd.DataFrame()
    
    # Attempt One!
    # --------------
    # Loop over range of ensemble files, inc. deterministic.
#     for i in range(n_ens+1):
#         if i == 0:
#             run = 'deterministic'
#             print('deterministic width: ',cme.width)
#             sta_profile, stb_profile = huxt_t_e_profile_fast(cme)
#             ts_errorA, ts_errorB, errorA, errorB = calculate_error(sta_profile, stb_profile)
        
#         elif i > 0:
#             run = "_{:02d}".format(i-1)        # ensemble run labelling starts at 0
            
#             # Create the elongation datafiles for A and B
#             sta_profile = pd.DataFrame()
#             sta_profile['time'] = ens_timeA.jd
#             sta_profile['lon'] = df_ensembleA['lon'+run]
#             sta_profile['r'] = df_ensembleA['r'+run]
#             sta_profile['el'] = df_ensembleA['el'+run]
#             sta_profile['lon_n'] = df_ensembleA['lon_n'+run]
#             sta_profile['r_n'] = df_ensembleA['r_n'+run]
#             sta_profile['el_n'] = df_ensembleA['el_n'+run]
#             sta_profile['lon_sec_flank'] = df_ensembleA['lon_sec_flank'+run]
#             sta_profile['r_sec_flank'] = df_ensembleA['r_sec_flank'+run]
#             sta_profile['el_sec_flank'] = df_ensembleA['el_sec_flank'+run]
            
#             stb_profile = pd.DataFrame()
#             stb_profile['time'] = ens_timeB.jd
#             stb_profile['lon'] = df_ensembleB['lon'+run]
#             stb_profile['r'] = df_ensembleB['r'+run]
#             stb_profile['el'] = df_ensembleB['el'+run]
#             stb_profile['lon_n'] = df_ensembleB['lon_n'+run]
#             stb_profile['r_n'] = df_ensembleB['r_n'+run]
#             stb_profile['el_n'] = df_ensembleB['el_n'+run]
#             stb_profile['lon_sec_flank'] = df_ensembleB['lon_sec_flank'+run]
#             stb_profile['r_sec_flank'] = df_ensembleB['r_sec_flank'+run]
#             stb_profile['el_sec_flank'] = df_ensembleB['el_sec_flank'+run]
            
#             # Calculate the error
#             ts_errorA, ts_errorB, errorA, errorB = calculate_error(sta_profile, stb_profile)
    
    # Attempt Two!
    # --------------
    project_dirs = H._setup_dirs_()
    for i in range(n_ens+1):
        if i == 0:
            ens_run = "deterministic"
            filename = "HUXt_CR2077_{}_deterministic.hdf5".format(tag)
            path = os.path.join(project_dirs['HUXt_data'], filename)
            filepath = glob.glob(path)[0]
            cme_list = load_cme_file(filepath)
            cme = cme_list[0] 
        else:
            # use i-1 as the first ensemble run is "ensemble_00"
            ens_run = "ensemble_{:02d}".format(i-1)
            filename = "HUXt_CR2077_{}_ensemble_{:02d}.hdf5".format(tag,i-1)
            path = os.path.join(project_dirs['HUXt_data'], filename)
            filepath = glob.glob(path)[0]
            cme_list = load_cme_file(filepath)
            cme = cme_list[0] 
         
#         print(cme.width)
        profileA, profileB = huxt_t_e_profile_fast(cme)
        ts_errorA, ts_errorB, errorA, errorB = calculate_error(profileA, profileB)
    # ------------------
    
        if cme.width.value == 37.4065736506236:
            out_path = project_dirs['HUXt_data']
            out_nameA = tag + '_TSerrorA_37.406.csv'
            ts_errorA.to_csv(os.path.join(out_path, out_nameA))
            out_nameB = tag + '_TSerrorB_37.406.csv'
            ts_errorB.to_csv(os.path.join(out_path, out_nameB))
            out_nameC = 'ST-B_profile2_37.406.csv'
            profileB.to_csv(os.path.join(out_path, out_nameC))
            
        ensemble_errorA =  ensemble_errorA.append({'file': ens_run, 
                                                    'total flank error':errorA['flank'][0],
                                                    'total nose error':errorA['nose'][0],
                                                    'total sec flank error': float('NaN'),
                                                    'RMSE flank error': errorA['RMSE flank'][0],
                                                    'RMSE nose error': errorA['RMSE nose'][0],
                                                    'RMSE sec flank error': float('NaN')},ignore_index = True)
            
        ensemble_errorB =  ensemble_errorB.append({'file': ens_run, 
                                                    'total flank error':errorB['flank'][0],
                                                    'total nose error':errorB['nose'][0],
                                                    'total sec flank error': errorB['sec flank'][0],
                                                    'RMSE flank error': errorB['RMSE flank'][0],
                                                    'RMSE nose error': errorB['RMSE nose'][0],
                                                    'RMSE sec flank error': errorB['RMSE sec flank'][0]},ignore_index = True)
        
    # add a two-feature (nose & flank) error values
    ensemble_errorA["total two feature error"] = ensemble_errorA["total flank error"] + ensemble_errorA["total nose error"]
    ensemble_errorA["RMSE two feature error"] = ensemble_errorA["RMSE flank error"] + ensemble_errorA["RMSE nose error"]
        
    ensemble_errorB["total two feature error "] = ensemble_errorB["total flank error"] + ensemble_errorB["total nose error"]
    ensemble_errorB["RMSE two feature error"] = ensemble_errorB["RMSE flank error"] + ensemble_errorB["RMSE nose error"]
    
    if save:
        # Save dataframe as a csv file in the correct file
        out_path = project_dirs['HUXt_data']
        out_name = tag + '_errors_sta.csv'
        ensemble_errorB.to_csv(os.path.join(out_path, out_name))
        out_name = tag + '_errors_stb.csv'
        ensemble_errorB.to_csv(os.path.join(out_path, out_name))
        
    return ensemble_errorA, ensemble_errorB
            
              
        