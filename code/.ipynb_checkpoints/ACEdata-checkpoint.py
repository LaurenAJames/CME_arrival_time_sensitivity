"""
This script contains a function to plot a specific time period of ACE data.
A vertical marker, labels the arrival time, can be plotted on this data

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
