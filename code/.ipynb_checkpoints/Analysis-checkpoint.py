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

def load_csv_file(file, tag=''):
    """
    Load data files with important values for ananysis of ensembe forecasting 
    Params:
    -------
    file: Name of file that need to be uploaded. Use one of the following : elongation profiles; errors; initial parameters; arrival information; best fit run.
    tag: tag name of the ensemble.
    
    Return:
    --------
    data :  dataframes of the .csv file. Some files return two dataframes and require two varibale names to be assigned. 
    """
    
    project_dirs = H._setup_dirs_()
    
    if file == "elongation profiles":
        pathA = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_sta.csv")
        ens_profileA = pd.read_csv(r"{}".format(pathA))
        ens_profileA = ens_profileA.drop(columns=['Unnamed: 0'])

        pathB = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_stb.csv")
        ens_profileB = pd.read_csv(r"{}".format(pathB))
        ens_profileB = ens_profileB.drop(columns=['Unnamed: 0'])
        
        data = ens_profileA, ens_profileB
        
    elif file == "errors":
        pathA = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_errorA.csv")
        ens_errorA = pd.read_csv(r"{}".format(pathA))
        ens_errorA = ens_errorA.drop(columns=['Unnamed: 0'])
        pathB = os.path.join(project_dirs['HUXt_data'], tag+"_ensemble_errorB.csv")
        ens_errorB = pd.read_csv(r"{}".format(pathB))
        ens_errorB = ens_errorB.drop(columns=['Unnamed: 0'])
        
        data = ens_errorA, ens_errorB

    elif file == "initial parameters":
        path = os.path.join(project_dirs['HUXt_data'], tag+"_cme_params.csv")
        cme_parameters = pd.read_csv(r"{}".format(path))
        cme_parameters = cme_parameters.drop(columns=['Unnamed: 0'])
        
        data = cme_parameters
        
    elif file == "arrival information":
        path = os.path.join(project_dirs['HUXt_data'], tag+"_arrival_info.csv")
        arrival_info = pd.read_csv(r"{}".format(path))
        arrival_info = arrival_info.drop(columns=['Unnamed: 0'])
        
        data = arrival_info
        
    elif file == "best fit run":
        path = os.path.join(project_dirs['HUXt_data'], tag+"_bestfitA.csv")
        best_fit_runA = pd.read_csv(r"{}".format(path))
        best_fit_runA = best_fit_runA.drop(columns=['Unnamed: 0'])
        
        path = os.path.join(project_dirs['HUXt_data'], tag+"_bestfitB.csv")
        best_fit_runB = pd.read_csv(r"{}".format(path))
        best_fit_runB = best_fit_runB.drop(columns=['Unnamed: 0'])
        
        path = os.path.join(project_dirs['HUXt_data'], tag+"_bestfitMISC.csv")
        best_fit_runMISC = pd.read_csv(r"{}".format(path))
        best_fit_runMISC = best_fit_runMISC.drop(columns=['Unnamed: 0'])
        
        data = best_fit_runA, best_fit_runB, best_fit_runMISC
        
    else:
        print('Error: "{}" file request not valid.'.format(file))
    
    return data


def ens_bar_plot(tag, save=False, tag_title=False):
    
    # Set model
    project_dirs = H._setup_dirs_()
    filename = "HUXt_CR2077_{}_deterministic.hdf5".format(tag)
    path = os.path.join(project_dirs['HUXt_data'], filename)
    filepath = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(filepath)
    
    # Load info
    arrival_info = load_csv_file("arrival information", tag)
    A, B, MISC = load_csv_file("best fit run", tag)

    # Ensure deterministic run is in the data
    if len(arrival_info) == 200:         # Where 50 corresponds the the number of ensemble runs. Change this to N in the future.
        # Add deterministic arrival information 
        arrival_info = arrival_info.append({"Transit Time" : cme.earth_transit_time.value, 
                                            "Arrival Time" : cme.earth_arrival_time.jd,
                                            "Arrival Speed" : cme.earth_arrival_speed.value}, ignore_index=True)

    # Sort the dataframe in terms of arrival time
    sorted_arrival = arrival_info.sort_values(by=["Arrival Time"])
    sorted_arrival = sorted_arrival.reset_index(drop=True)
    no_unique_values = len(sorted_arrival["Arrival Time"].unique())

    # Make bins based on model outpout
    bins = []
    for i in range(no_unique_values):
        if i == 0:
            astroTimebins = Time(sorted_arrival["Arrival Time"][0], format='jd')
        else:
            astroTimebins = astroTimebins + model.dt_out
            astroTimebins = Time(astroTimebins, format='jd')
        bins.append(astroTimebins.value)

    # Generate freq of values
    freq = sorted_arrival["Arrival Time"].value_counts(sort=True)
    binfreq = sorted_arrival["Arrival Time"].value_counts(bins=bins, sort=True).rename_axis('unique_values').reset_index(name='counts')
    binfreqsort = binfreq.sort_values(by=["unique_values"])
    
    # drop invalid data that impacts statistical analysis
    for index,value in enumerate(arrival_info["Arrival Time"]):
        if value == Time('3000-01-01 00:00:00.000', format='iso').jd:
            arrival_info = arrival_info.drop([index])

    # Assign time to variables
    #---------------------------
    obsACE = Time('2008-12-16 07:00:00', format='iso').jd

    df = arrival_info[arrival_info["file"]== 'deterministic']
    deter_time = Time(df["Arrival Time"], format='jd')[0]

    dfA = A[A["Feature"]== 'N & F']
    STA_time = Time(dfA["Arrival Time"], format='jd')[0]

    dfB = B[B["Feature"]== 'N & F']
    STB_time = Time(dfB["Arrival Time"], format='jd')[0]

    dfMISC = MISC[MISC["Feature"]== 'Both Spacecrafts']
    STMISC_time = Time(dfMISC["Arrival Time"][0], format='iso')
    
    ens_mean = arrival_info['Arrival Time'].mean()
    ens_std = arrival_info['Arrival Time'].std()
    ens_skew = arrival_info['Arrival Time'].skew()
    ens_min = arrival_info['Arrival Time'].min()
    ens_max = arrival_info['Arrival Time'].max()
    ens_len = len(arrival_info['Arrival Time'])

    # ----------------------------
    # Plot figure
    
    xlimmin = (arrival_info['Arrival Time'].min() - obsACE - 0.1) * 24
    xlimmax = (arrival_info['Arrival Time'].max() - obsACE + 0.1) * 24

    plt.figure(figsize=(12,6))
    plt.ylabel("Frequency")
    plt.xlabel("Time Error (hours)")
    plt.xlim(left=xlimmin, right=xlimmax)
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.ylim(top=40)
    
    plt.axvspan(xmin = - 1, xmax = 1, facecolor='grey', alpha=0.2, label="Obs. Arrival Uncertainty")
    plt.axvline(x = 0, color = 'k', linestyle = '-', linewidth = 2, label = 'Obs. Arrival Time')
    
    plt.bar((bins[0:no_unique_values-1]-obsACE)* 24, (binfreqsort["counts"].values[0:no_unique_values-1]), 
            width=34.8/60, align='edge', color='None', edgecolor='k')
    # plt.plot(bins[0:63]-obsACE,binfreqsort["counts"].values[0:63], 'kx')


    # Plot lines
    plt.axvline(x = (deter_time.jd - obsACE) *24, color='blue', label='Deterministic', linestyle=(0,(5,1)), linewidth=2)
    plt.axvline(x = (STA_time.jd - obsACE) *24, color='forestgreen', label='STRERO-A', linestyle=(0,(5,5)), linewidth=2)
    plt.axvline(x = (STB_time.jd - obsACE) *24, color='crimson', label='STRERO-B', linestyle='-.', linewidth=2)
    plt.axvline(x = (STMISC_time.jd - obsACE) *24, color='orange', label='Both', linestyle=(0,(5,10)), linewidth=2)
    
    # Mean & std
#     plt.axvspan(xmin = (ens_mean - ens_std - obsACE)*24, xmax = (ens_mean + ens_std - obsACE)*24, 
#     facecolor='m', alpha=0.025, label="Standard Deviation")
#     plt.axvline(x = (ens_mean - obsACE)*24, color='m', label='Mean', linestyle='-', linewidth=2)
    plt.text(xlimmax-0.5,37, 'Mean = {:.3f} hours'.format((ens_mean-obsACE)*24), fontsize=14, ha='right')
    plt.text(xlimmax-0.5, 35, 'Std = {:.3f} hours'.format(ens_std*24), fontsize=14, ha='right')
    plt.text(xlimmax-0.5,33, 'Skew = {:.3f}'.format(ens_skew*24), fontsize=14, ha='right')

    plt.legend(loc=[1.1,0.35], frameon=False, fontsize=14)
    if tag_title == True:
        plt.text(xlimmax-0.2,40.5, '{}'.format(tag), fontsize=14, ha='right', weight='bold')
    
    plt.text(xlimmax+1,37, 'Min = {:.3f} hours'.format((ens_min-obsACE)*24), fontsize=14, ha='left')
    plt.text(xlimmax+1,35, 'Max = {:.3f} hours'.format((ens_max-obsACE)*24), fontsize=14, ha='left')
    plt.text(xlimmax+1,33, 'Member Count = {}'.format(ens_len), fontsize=14, ha='left')
    
    
    if save:
        filename = "HUXt_{}_ens_bar_plot.png".format(tag)
        filepath = os.path.join(model._figure_dir_, filename)  
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_ens_bar_plot.pdf".format(tag)
        filepath = os.path.join(model._figure_dir_, filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
    return
        
    
def RMSEscatter(feature, view, tag, tag_overlay='', overlay=False, title=True, save=False):

    # load data
    ainfo = load_csv_file("arrival information", tag=tag)
    errorsA, errorsB = load_csv_file("errors", tag=tag)

    # drop invalid data that impacts statistical analysis
    for index,value in enumerate(ainfo["Arrival Time"]):
        if value == Time('3000-01-01 00:00:00.000', format='iso').jd:
            ainfo = ainfo.drop([index])
            errorsA = errorsA.drop([index])
            errorsB = errorsB.drop([index])

    obsACE = Time('2008-12-16 07:00:00', format='iso').jd
    x = (ainfo["Arrival Time"] - obsACE)*24
    
    if view == "A":
        if feature == "nose" or feature == "flank" or feature == "N & F":
            y = errorsA["RMSE {} error".format(feature)]
        elif feature == "sec flank":
            print("Secondary flank not available from this viewpoint.")
        else:
            print("Error: Feature not valid.")
    elif view == "B":
        if feature == "nose" or feature == "flank" or feature == "N & F" or feature == "sec flank":
            y = errorsB["RMSE {} error".format(feature)]
        else:
            print("Error: Feature not valid.")
    elif view == "both":
        if feature == "N & F":
            y = errorsA["RMSE error w/ ST-B"]
        elif feature == "nose" or feature == "flank":
            y = errorsA["RMSE {} error".format(feature)] + errorsB["RMSE {} error".format(feature)]
        elif feature == "sec flank":
            print("Secondary flank not available from all viewpoints.")
        else:
            print("Error: Feature not valid.")
    
    
    plt.figure()
    plt.scatter(x,y, color='k', s=2)

    plt.xlabel("Time Error (Hours)")
    plt.ylabel("RMSE")
    plt.xlim(left=-10, right=20)
    plt.ylim(bottom=0, top=3)
    # ax=plt.gca()
    # # ax.xaxis.set_major_locator(MultipleLocator(2))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(.2))
    # ax.yaxis.set_minor_locator(MultipleLocator(.1))
    
    if title == True:
        plt.text(19.8,3.1, '{} from {}'.format(feature, view), fontsize=14, ha='right', weight='bold')
        
    if overlay == True:
        ainfo = load_csv_file("arrival information", tag=tag_overlay)
        errorsA, errorsB = load_csv_file("errors", tag=tag_overlay)

        # drop invalid data that impacts statistical analysis
        for index,value in enumerate(ainfo["Arrival Time"]):
            if value == Time('3000-01-01 00:00:00.000', format='iso').jd:
                ainfo = ainfo.drop([index])
                errorsA = errorsA.drop([index])
                errorsB = errorsB.drop([index])

        obsACE = Time('2008-12-16 07:00:00', format='iso').jd
        x = (ainfo["Arrival Time"] - obsACE)*24

        if view == "A":
            if feature == "nose" or feature == "flank" or feature == "N & F":
                y_o = errorsA["RMSE {} error".format(feature)]
            elif feature == "sec flank":
                print("Secondary flank not available from this viewpoint.")
            else:
                print("Error: Feature not valid.")
        elif view == "B":
            if feature == "nose" or feature == "flank" or feature == "N & F" or feature == "sec flank":
                y_o = errorsB["RMSE {} error".format(feature)]
            else:
                print("Error: Feature not valid.")
        elif view == "both":
            if feature == "N & F":
                y_o = errorsA["RMSE error w/ ST-B"]
            elif feature == "nose" or feature == "flank":
                y_o = errorsA["RMSE {} error".format(feature)] + errorsB["RMSE {} error".format(feature)]
            elif feature == "sec flank":
                print("Secondary flank not available from all viewpoints.")
            else:
                print("Error: Feature not valid.")
        
        plt.scatter(x,y_o, color='red', s=2)
        
        plt.legend(["2x Parameter Uncertainty", "1x Parameter Uncertainty"], bbox_to_anchor=(0.5, 0.06), loc='center', ncol=2, fontsize=10)
    
    project_dirs = H._setup_dirs_()
    
    if save:
        filename = "HUXt_{}_ens_scat_plot_{}{}.png".format(tag,feature,view)
        path = os.path.join(project_dirs['HUXt_figures'], filename)
        filepath = glob.glob(path)[0]
#         filepath = os.path.join(model._figure_dir_, filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_ens_scat_plot_{}{}.pdf".format(tag,feature,view)
        path = os.path.join(project_dirs['HUXt_figures'], filename)
        filepath = glob.glob(path)[0]
#         filepath = os.path.join(model._figure_dir_, filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    return

#------------------------------------
#---------- PAPER PLOTS -------------
#------------------------------------

def paper_ens_bar(tag, save=False, tag_title=False):
    
    # Set model
    project_dirs = H._setup_dirs_()
    filename = "HUXt_CR2077_{}_deterministic.hdf5".format(tag)
    path = os.path.join(project_dirs['HUXt_data'], filename)
    filepath = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(filepath)
    
    # Load info
    arrival_info = load_csv_file("arrival information", tag)
    A, B, MISC = load_csv_file("best fit run", tag)

    # Ensure deterministic run is in the data
    if len(arrival_info) == 200:         # Where 50 corresponds the the number of ensemble runs. Change this to N in the future.
        # Add deterministic arrival information 
        arrival_info = arrival_info.append({"Transit Time" : cme.earth_transit_time.value, 
                                            "Arrival Time" : cme.earth_arrival_time.jd,
                                            "Arrival Speed" : cme.earth_arrival_speed.value}, ignore_index=True)

    # Sort the dataframe in terms of arrival time
    sorted_arrival = arrival_info.sort_values(by=["Arrival Time"])
    sorted_arrival = sorted_arrival.reset_index(drop=True)
    no_unique_values = len(sorted_arrival["Arrival Time"].unique())

    # Make bins based on model outpout
    bins = []
    for i in range(no_unique_values):
        if i == 0:
            astroTimebins = Time(sorted_arrival["Arrival Time"][0], format='jd')
        else:
            astroTimebins = astroTimebins + model.dt_out
            astroTimebins = Time(astroTimebins, format='jd')
        bins.append(astroTimebins.value)

    # Generate freq of values
    freq = sorted_arrival["Arrival Time"].value_counts(sort=True)
    binfreq = sorted_arrival["Arrival Time"].value_counts(bins=bins, sort=True).rename_axis('unique_values').reset_index(name='counts')
    binfreqsort = binfreq.sort_values(by=["unique_values"])
    
    # drop invalid data that impacts statistical analysis
    for index,value in enumerate(arrival_info["Arrival Time"]):
        if value == Time('3000-01-01 00:00:00.000', format='iso').jd:
            arrival_info = arrival_info.drop([index])

    # Assign time to variables
    #---------------------------
    obsACE = Time('2008-12-16 07:00:00', format='iso').jd

    df = arrival_info[arrival_info["file"]== 'deterministic']
    deter_time = Time(df["Arrival Time"], format='jd')[0]

    dfA = A[A["Feature"]== 'N & F']
    STA_time = Time(dfA["Arrival Time"], format='jd')[0]

    dfB = B[B["Feature"]== 'N & F']
    STB_time = Time(dfB["Arrival Time"], format='jd')[0]

    dfMISC = MISC[MISC["Feature"]== 'Both Spacecrafts']
    STMISC_time = Time(dfMISC["Arrival Time"][0], format='iso')
    
    ens_mean = arrival_info['Arrival Time'].mean()
    ens_std = arrival_info['Arrival Time'].std()
    ens_skew = arrival_info['Arrival Time'].skew()
    ens_min = arrival_info['Arrival Time'].min()
    ens_max = arrival_info['Arrival Time'].max()
    ens_len = len(arrival_info['Arrival Time'])

    # ----------------------------
    # Plot figure
    
    xlimmin = (arrival_info['Arrival Time'].min() - obsACE - 0.1) * 24
    xlimmax = (arrival_info['Arrival Time'].max() - obsACE + 0.1) * 24

    plt.figure(figsize=(12,6))
    plt.ylabel("Frequency")
    plt.xlabel("Time Error (hours)")
    plt.xlim(left=xlimmin, right=xlimmax)
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.ylim(top=40)
    
    plt.axvspan(xmin = - 1, xmax = 1, facecolor='grey', alpha=0.2, label="Obs. Arrival Uncertainty")
    plt.axvline(x = 0, color = 'k', linestyle = '-', linewidth = 2, label = 'Obs. Arrival Time')
    
    plt.bar((bins[0:no_unique_values-1]-obsACE)* 24, (binfreqsort["counts"].values[0:no_unique_values-1]), 
            width=34.8/60, align='edge', color='None', edgecolor='k')
    # plt.plot(bins[0:63]-obsACE,binfreqsort["counts"].values[0:63], 'kx')


    # Plot lines
    plt.axvline(x = (deter_time.jd - obsACE) *24, color='blue', label='Deterministic', linestyle=(0,(5,1)), linewidth=2)
    plt.axvline(x = (STA_time.jd - obsACE) *24, color='forestgreen', label='STRERO-A', linestyle=(0,(5,5)), linewidth=2)
    plt.axvline(x = (STB_time.jd - obsACE) *24, color='crimson', label='STRERO-B', linestyle='-.', linewidth=2)
    plt.axvline(x = (STMISC_time.jd - obsACE) *24, color='orange', label='Both', linestyle=(0,(5,10)), linewidth=2)
    
    # Mean & std
#     plt.axvspan(xmin = (ens_mean - ens_std - obsACE)*24, xmax = (ens_mean + ens_std - obsACE)*24, 
#     facecolor='m', alpha=0.025, label="Standard Deviation")
#     plt.axvline(x = (ens_mean - obsACE)*24, color='m', label='Mean', linestyle='-', linewidth=2)
    plt.text(xlimmax-0.5,37, 'Mean = {:.3f} hours'.format((ens_mean-obsACE)*24), fontsize=14, ha='right')
    plt.text(xlimmax-0.5, 35, 'Std = {:.3f} hours'.format(ens_std*24), fontsize=14, ha='right')
    plt.text(xlimmax-0.5,33, 'Skew = {:.3f}'.format(ens_skew*24), fontsize=14, ha='right')

    plt.legend(loc=[1.1,0.35], frameon=False, fontsize=14)
    if tag_title == True:
        plt.text(xlimmax-0.2,40.5, '{}'.format(tag), fontsize=14, ha='right', weight='bold')
    
    plt.text(xlimmax+1,37, 'Min = {:.3f} hours'.format((ens_min-obsACE)*24), fontsize=14, ha='left')
    plt.text(xlimmax+1,35, 'Max = {:.3f} hours'.format((ens_max-obsACE)*24), fontsize=14, ha='left')
    plt.text(xlimmax+1,33, 'Member Count = {}'.format(ens_len), fontsize=14, ha='left')
    
    
    if save:
        filename = "HUXt_{}_ens_bar_plot.png".format(tag)
        filepath = os.path.join(model._figure_dir_, filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_ens_bar_plot.pdf".format(tag)
        filepath = os.path.join(model._figure_dir_, filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
    return

