# Import libaries
# As seen in HIEnsembleHindcast/ensemble_analysis.ipynb by L.Barnard ()
import HUXt as H
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
from datetime import datetime, timedelta
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.lines as mlines

import Analysis2 as ana
import cmath


def plotACEinsitudata(arrivaltime = "2008-12-16 07:00:00", save=False, saveformat='pdf'):
    """
    Plot the in-situ data from ACE for the December 12th CME. We have downloaded data from //cdaweb.gsfc.nans.gov and used the following instruments:
        • AC_H2_MFI for the magnetic field in GSE coordinates at 1-hour intervals
        • AC_H6_SWI for the proton density and speed at 12-minute intervals 
        • [NO LONGER USED] --- AC_H2_SWE for the solar wind bluk speek at 1-hour intervals () --- 
    """

    # We load the .csv data in panda dataframes for easy data access
    
    # Load in ACE MAGNETIC observation
    df_aceMAG = pd.read_csv(r"AC_H1_MFI_91226.csv")
    df_aceMAG = df_aceMAG.rename(columns={'EPOCH__yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'BX_GSE_(@_x_component_)_nT':'Bx (nT)',
                                          'BY_GSE_(@_y_component_)_nT':'By (nT)',
                                          'BZ_GSE_(@_z_component_)_nT':'Bz (nT)'})
    df_aceMAG['time'] = pd.to_datetime(df_aceMAG['time'])

    # Load in ACE PROTON DENSITY observation
    df_aceDEN = pd.read_csv(r"AC_H6_SWI_226960.csv")
    df_aceDEN = df_aceDEN.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'PROTON_DENSITY_#/cm^3':'proton density (/cm3)'})
    df_aceDEN['time'] = pd.to_datetime(df_aceDEN['time'])

    # Load in ACE SOLAR WIND SPEED observation
    df_aceSWS = pd.read_csv(r"AC_H6_SWI_91226.csv")
    df_aceSWS = df_aceSWS.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'PROTON_SPEED_km/s':'Speed (km/s)'})
    df_aceSWS['time'] = pd.to_datetime(df_aceSWS['time'])

    # Define the arrival time of the CME - this will be plotted as a red line later in the script
    arrival = Time(arrivaltime, format='iso').datetime


    # Setup figure

    plt.rcParams.update({'font.size': 22, 'axes.labelsize':14, 'legend.fontsize':16,'xtick.labelsize': 12.0,'ytick.labelsize': 12.0,"font.family":"Times New Roman"})
    fig, axs = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(8.27, 11.69)) # Paper size is equal to A4 portrait

    axs[0].set_ylabel("Speed\n  (Km s$^{-1}$)")
    axs[0].plot(df_aceSWS["time"],df_aceSWS["Speed (km/s)"], 'k', lw=0.5)
    axs[0].set_ylim(bottom=300, top=450)
    axs[0].yaxis.set_major_locator(MultipleLocator(50))
    axs[0].yaxis.set_minor_locator(MultipleLocator(10))

    axs[1].set_ylabel("Proton density\n  (cm$^{-3}$)")
    axs[1].plot(df_aceDEN["time"],df_aceDEN["proton density (/cm3)"], 'k', lw=0.5)
    axs[1].set_ylim(bottom=0, top=25)
    axs[1].yaxis.set_major_locator(MultipleLocator(5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(1))

    axs[2].set_ylabel("Magnetic Field,\n Bx (nT)")
    axs[2].plot(df_aceMAG["time"],df_aceMAG["Bx (nT)"], 'k', lw=0.5)
    axs[2].set_ylim(bottom=-10.0, top=10)
    axs[2].yaxis.set_major_locator(MultipleLocator(5))
    axs[2].yaxis.set_minor_locator(MultipleLocator(1))

    axs[3].set_ylabel("Magnetic Field,\n By (nT)")
    axs[3].plot(df_aceMAG["time"],df_aceMAG["By (nT)"], 'k', lw=0.5)
    axs[3].set_ylim(bottom=-10, top=10)
    axs[3].yaxis.set_major_locator(MultipleLocator(5))
    axs[3].yaxis.set_minor_locator(MultipleLocator(1))

    axs[4].set_ylabel("Magnetic Field,\n Bz (nT)")
    axs[4].plot(df_aceMAG["time"],df_aceMAG["Bz (nT)"], 'k', lw=0.5)
    axs[4].set_ylim(bottom=-10, top=10)
    axs[4].yaxis.set_major_locator(MultipleLocator(5))
    axs[4].yaxis.set_minor_locator(MultipleLocator(1))


    axs[4].set_xlabel("Time")
    axs[4].set_xlim(left= df_aceMAG.time.min() , right= df_aceDEN.time.max())
    axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m/%d'))
    axs[4].xaxis.set_minor_locator(mdates.HourLocator(interval=3))

    axs[0].axvline(x=arrival, ymin=-10, ymax=10,color='r', lw=0.75)
    axs[1].axvline(x=arrival, ymin=-10, ymax=10,color='r', lw=0.75)
    axs[2].axvline(x=arrival, ymin=-10, ymax=10,color='r', lw=0.75)
    axs[3].axvline(x=arrival, ymin=-10, ymax=10,color='r', lw=0.75)
    axs[4].axvline(x=arrival, ymin=-10, ymax=10,color='r', lw=0.75)

    # axs[0].set_title('a)', loc='left', fontsize=14)
    # axs[1].set_title('b)', loc='left', fontsize=14)
    # axs[2].set_title('c)', loc='left', fontsize=14)
    # axs[3].set_title('d)', loc='left', fontsize=14)
    # axs[4].set_title('e)', loc='left', fontsize=14)

    axs[0].annotate("a)", xy=(0.01, 0.85), xycoords="axes fraction", fontsize=14)
    axs[1].annotate("b)", xy=(0.01, 0.85), xycoords="axes fraction", fontsize=14)
    axs[2].annotate("c)", xy=(0.01, 0.85), xycoords="axes fraction", fontsize=14)
    axs[3].annotate("d)", xy=(0.01, 0.85), xycoords="axes fraction", fontsize=14)
    axs[4].annotate("e)", xy=(0.01, 0.85), xycoords="axes fraction", fontsize=14)

    plt.show()

    if save:
        project_dirs = H._setup_dirs_()

        filename = "12Dec08CME_ACEobservations_plot.{}".format(saveformat)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        fig.savefig(filepath, dpi=300, bbox_inches='tight')


        
def plotACEandSTEREOinsitudata(arrivaltime = "2008-12-16 07:00:00", save=False, saveformat='pdf'):
    """
    Plot the in-situ data from ACE and STEREO for CR2077 (2008/11/20 06:56:00.000 - 2008/12/20 07:00:00.000). 
    We have downloaded data from //cdaweb.gsfc.nans.gov and used the following instruments:
        • AC_H6_SWI for the proton speed at 12-minute intervals (ACE)
        • STA_L2_PLA_1DMAX_10MIN for solar wind speed at 10-minute intervals (STEREO-A)
        • STB_L2_PLA_1DMAX_10MIN for solar wind speed at 10-minute intervals (STEREO-B)
    
    """

    # We load the .csv data in panda dataframes for easy data access
    
    # ACE 
    df_aceSWS = pd.read_csv(r"AC_H6_SWI_48482.csv")
    df_aceSWS = df_aceSWS.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'PROTON_SPEED_km/s':'Speed (km/s)'})
    df_aceSWS['time'] = pd.to_datetime(df_aceSWS['time'])
    
    # STEREO A 
    df_stereo_aSWS = pd.read_csv(r"STA_L2_PLA_1DMAX_10MIN_67449.csv")
    df_stereo_aSWS = df_stereo_aSWS.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'SPEED_km/s':'Speed (km/s)'})
    df_stereo_aSWS['time'] = pd.to_datetime(df_stereo_aSWS['time'])
    
    # STEREO A 
    df_stereo_bSWS = pd.read_csv(r"STB_L2_PLA_1DMAX_10MIN_67449.csv")
    df_stereo_bSWS = df_stereo_bSWS.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'SPEED_km/s':'Speed (km/s)'})
    df_stereo_bSWS['time'] = pd.to_datetime(df_stereo_bSWS['time'])

    # Define the arrival time of the CME - this will be plotted as a red line later in the script
    arrival = Time(arrivaltime, format='iso').datetime
    
    # Fast Steam Identification
    start= Time("2008-12-08 00:00:00", format='iso').datetime
    end= Time("2008-12-15 00:00:00", format='iso').datetime
    
    # Setup figure

    plt.rcParams.update({'font.size': 22, 'axes.labelsize':14, 'legend.fontsize':16,'xtick.labelsize': 12.0,'ytick.labelsize': 12.0,"font.family":"Times New Roman"})
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(8.27, 9)) # Paper size is equal to A4 portrait

    axs[0].plot(df_stereo_bSWS["time"],df_stereo_bSWS["Speed (km/s)"], 'k', lw=0.5)
    axs[1].plot(df_aceSWS["time"],df_aceSWS["Speed (km/s)"], 'k', lw=0.5)
    axs[2].plot(df_stereo_aSWS["time"],df_stereo_aSWS["Speed (km/s)"], 'k', lw=0.5)

    axs[2].set_xlabel("Time")
    axs[2].set_xlim(left= df_stereo_aSWS.time.min() , right= df_stereo_aSWS.time.max())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axs[2].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axs[2].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    
    for nn,ax in enumerate(axs):
        ax.set_ylabel("Speed\n  (Km s$^{-1}$)")
        ax.set_ylim(bottom=200, top=700)
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(20))
#         ax.axvline(x=arrival, ymin=-10, ymax=10,color='r', lw=0.75)

    axs[0].axvspan(start,end,alpha=0.15, color='orange', lw=0)
    axs[1].axvspan(start+timedelta(days=3),end+timedelta(days=3),alpha=0.15, color='orange', lw=0)
    axs[2].axvspan(start+timedelta(days=6),end+timedelta(days=6),alpha=0.15, color='orange', lw=0)
    
    axs[0].annotate("STEREO-B", xy=(0.01, 0.87), xycoords="axes fraction", fontsize=14)
    axs[1].annotate("ACE", xy=(0.01, 0.87), xycoords="axes fraction", fontsize=14)
    axs[2].annotate("STEREO-A", xy=(0.01, 0.87), xycoords="axes fraction", fontsize=14)

    plt.show()

    if save:
        project_dirs = H._setup_dirs_()

        filename = "12Dec08CME_ACEandSTEREO_swspeed_plot.{}".format(saveformat)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        
        
def plot_timeestimates_onACEinsitudata(arrivaltimes, arrivaluncertainties, leadtime, save=False, saveformat='pdf'):
    """
    Plot the in-situ data from ACE for the December 12th CME. We have downloaded data from //cdaweb.gsfc.nans.gov and used the following instruments:
        • AC_H2_MFI for the magnetic field in GSE coordinates at 1-hour intervals
        • AC_H6_SWI for the proton density and speed at 12-minute intervals 
        • [NO LONGER USED] --- AC_H2_SWE for the solar wind bluk speek at 1-hour intervals () --- 
    """

    # We load the .csv data in panda dataframes for easy data access
    
    # Load in ACE MAGNETIC observation
    df_aceMAG = pd.read_csv(r"AC_H1_MFI_91226.csv")
    df_aceMAG = df_aceMAG.rename(columns={'EPOCH__yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'BX_GSE_(@_x_component_)_nT':'Bx (nT)',
                                          'BY_GSE_(@_y_component_)_nT':'By (nT)',
                                          'BZ_GSE_(@_z_component_)_nT':'Bz (nT)'})
    df_aceMAG['time'] = pd.to_datetime(df_aceMAG['time'])

    # Load in ACE PROTON DENSITY observation
    df_aceDEN = pd.read_csv(r"AC_H6_SWI_226960.csv")
    df_aceDEN = df_aceDEN.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'PROTON_DENSITY_#/cm^3':'proton density (/cm3)'})
    df_aceDEN['time'] = pd.to_datetime(df_aceDEN['time'])

    # Load in ACE SOLAR WIND SPEED observation
    df_aceSWS = pd.read_csv(r"AC_H6_SWI_91226.csv")
    df_aceSWS = df_aceSWS.rename(columns={'EPOCH_yyyy-mm-ddThh:mm:ss.sssZ' : 'time',
                                          'PROTON_SPEED_km/s':'Speed (km/s)'})
    df_aceSWS['time'] = pd.to_datetime(df_aceSWS['time'])

    # Define the arrival time of the CME - this will be plotted as a red line later in the script
    if len(arrivaltimes) != 3:
        print("three arrival times arguements required: BRaVDA-HUXt 30, BRaVDA-HUXT 8, MAS-HUXt 30")
        return
    time1, time2, time3 = arrivaltimes
    arrival1 = Time(time1, format='iso').datetime
    arrival2 = Time(time2, format='iso').datetime
    arrival3 = Time(time3, format='iso').datetime
    
    if len(arrivaluncertainties) != 3:
        print("three arrival times uncertanties arguements required [unit: HOURS]: BRaVDA-HUXt 30, BRaVDA-HUXT 8, MAS-HUXt 30")
        return
    error1, error2, error3 = arrivaluncertainties
    error1 = timedelta(hours=error1)
    error2 = timedelta(hours=error2)
    error3 = timedelta(hours=error3)

    # Setup figure

    plt.rcParams.update({'font.size': 22, 'axes.labelsize':14, 'legend.fontsize':10,'xtick.labelsize': 12.0,'ytick.labelsize': 12.0,"font.family":"Times New Roman"})
    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(8, 8)) # Paper size is equal to A4 portrait

    axs[0].set_ylabel("Speed\n  (Km s$^{-1}$)")
    axs[0].plot(df_aceSWS["time"],df_aceSWS["Speed (km/s)"], 'k', lw=0.5)
    axs[0].set_ylim(bottom=300, top=450)
    axs[0].yaxis.set_major_locator(MultipleLocator(50))
    axs[0].yaxis.set_minor_locator(MultipleLocator(10))

    axs[1].set_ylabel("Proton density\n  (cm$^{-3}$)")
    axs[1].plot(df_aceDEN["time"],df_aceDEN["proton density (/cm3)"], 'k', lw=0.5)
    axs[1].set_ylim(bottom=0, top=25)
    axs[1].yaxis.set_major_locator(MultipleLocator(5))
    axs[1].yaxis.set_minor_locator(MultipleLocator(1))

    axs[2].set_ylabel("Magnetic Field,\n (nT)")
    axs[2].plot(df_aceMAG["time"],df_aceMAG["Bx (nT)"], 'k', lw=0.5, label="Bx")
    axs[2].set_ylim(bottom=-10.0, top=10)
    axs[2].yaxis.set_major_locator(MultipleLocator(5))
    axs[2].yaxis.set_minor_locator(MultipleLocator(1))

#     axs[3].set_ylabel("Magnetic Field,\n By (nT)")
    axs[2].plot(df_aceMAG["time"],df_aceMAG["By (nT)"], 'k:', lw=0.5, label="By")
#     axs[3].set_ylim(bottom=-10, top=10)
#     axs[3].yaxis.set_major_locator(MultipleLocator(5))
#     axs[3].yaxis.set_minor_locator(MultipleLocator(1))

#     axs[4].set_ylabel("Magnetic Field,\n Bz (nT)")
    axs[2].plot(df_aceMAG["time"],df_aceMAG["Bz (nT)"], '--',color='gray', lw=0.5, label="Bz")
#     axs[4].set_ylim(bottom=-10, top=10)
#     axs[4].yaxis.set_major_locator(MultipleLocator(5))
#     axs[4].yaxis.set_minor_locator(MultipleLocator(1))
    axs[2].legend(ncol=3, loc="righttop")


    axs[2].set_xlabel("Time")
    axs[2].set_xlim(left= Time("2008-12-15 12:00:00", format='iso').datetime , right= df_aceDEN.time.max())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m/%d'))
    axs[2].xaxis.set_minor_locator(mdates.HourLocator(interval=3))

    for nn, ax in enumerate(axs):
        ax.axvline(x=arrival1, ymin=-10, ymax=10,color='b', lw=0.75)
        ax.axvline(x=arrival2, ymin=-10, ymax=10,color='r', lw=0.75)
        ax.axvline(x=arrival3, ymin=-10, ymax=10,color='y', lw=0.75)
        
        ax.axvspan(arrival1-error1, arrival1+error1, alpha=0.3, color='b',lw=0)
        ax.axvspan(arrival2-error2, arrival2+error2, alpha=0.3, color='r',lw=0)
        ax.axvspan(arrival3-error3, arrival3+error3, alpha=0.3, color='y',lw=0)

    black_line = mlines.Line2D([], [], color='b',linestyle="-",markersize=0, label="BRaVDA-HUXt from 30Rs")
    black_line2 = mlines.Line2D([], [], color='r',linestyle="-", markersize=0, label="BRaVDA-HUXt from 8Rs")
    black_line3 = mlines.Line2D([], [], color='y',linestyle="-", markersize=0, label="MAS-HUXt from 30Rs")
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.22, hspace=0.2, wspace=0.15)
    fig.legend(bbox_to_anchor=(0.5, 0.06), loc='center', ncol=3, frameon=True, fontsize=12, handles=[black_line,black_line2, black_line3])
    plt.show()

    if save:
        project_dirs = H._setup_dirs_()

        filename = "12Dec08CME_ACEobservationsVsEnsembleEstimate_{}hours_plot.{}".format(leadtime,saveformat)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
