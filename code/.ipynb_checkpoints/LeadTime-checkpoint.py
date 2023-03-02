"""
Script made for the analysisf of lead time work; making up the third chpater of the thesis.

Using ensemble of HUXt runs (from the ghostfronts.ipynb script), the cumulative RMSE for each observation is stored. 
We use this dataset to analyse the affect more observations has on the arrival time estimate of CME at Earth.

"""

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
from scipy.optimize import curve_fit
import glob
import math
import sunpy.coordinates.sun as sn
import h5py
import ensemble as ens
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import cmath
import Analysis2 as ana

# ______________________________________

def loopLT(tag, feature="N&F", percent=25, lowestsubset=True, xaxis="time"):
    """
    Create a loop through the number of observation timesteps to run lead time analysis
    This will:
    1. Find the timestamp (index) required to run a single ensemble analysis
    2. Create a subset of data, from the timestamp
    3. Create another subset (now a sub-subset), seeking the lowest 25% of binned data
    4. Create a scatter of cumulate RMSE and L1 arrival error, and seek the correlation of the fit
    5. Find the ensemble estimate of arrival
    6. Store into a dataframe 
    
    Return:
    ========
    the dataframe informing the last HI timestamp, ensemble estimate, ensemble estimate RMSE. This should be on length equal to number of observations
    """
    # These are hard-set variables for the 12Dec2008 event
    obsACE = Time('2008-12-16 07:00:00', format='iso').jd
    feature = "N&F"
    spacecraft = "STEREO"
    
    # Create dataframe
    ens_estimates = pd.DataFrame()
    
    # Step 1,2 and 3. 
    for index in range(42):
        print("Progress: {} out of 42".format(index))
        dataframes = bindataLT(tag, index, feature, percent, lowestsubset)
        if dataframes is not None:
            arrival_info, timeres_subset, histoinfo, lowestmembers = dataframes
            lowestmembers = lowestmembers.dropna()
            timestamp = arrival_info["HI Time B"][index]
        
            # Step 4
            if xaxis == "time":
                x1 = (arrival_info["Arrival Time"] - obsACE)*24
                y1 = (arrival_info["RMSE {} {}".format(feature, spacecraft)])

                xlow1 = (lowestmembers["Arrival Time"] - obsACE)*24
                ylow1 = (lowestmembers["RMSE {} {}".format(feature, spacecraft)])

                deter1 = arrival_info[arrival_info["file"] == "deterministic"]
                deterATE1 = (deter1["Arrival Time"][0] - obsACE)*24

            elif xaxis == "speed":
                x1 = (arrival_info["Arrival Speed Error (km/s)"])
                y1 = (arrival_info["RMSE {} {}".format(feature, spacecraft)])

                xlow1 = (lowestmembers["Arrival Speed Error (km/s)"])
                ylow1 = (lowestmembers["RMSE {} {}".format(feature, spacecraft)])

                deter1 = arrival_info[arrival_info["file"] == "deterministic"]
                deterATE1 = (deter1["Arrival Speed Error (km/s)"][0])
                print("Warning: Functionality for speed analysis is yet to be developed.")

            else:
                print("xaxis is not valid. Use 'time' or 'speed'.")
                return

            # Find quadratic curve of regression    
            popt, fx_df, x_solution = ana.quadraticcurvefit(xlow1, ylow1, timeres_subset, obsACE)
            xsol1, xuncert68, xuncert95 = x_solution
            # define a sequence of inputs between the smallest and largest known inputs
    #             fxX1 = np.arange(min(x1), max(x1)+1, 0.5)
            # calculate the output for the range
            a, b, c = popt
    #             fxY1 = ana.objective(fxX1, a, b, c)

            # Store into dataframe
            ens_estimates = ens_estimates.append({"obs timestamp": timestamp ,
                                                  "ens estimate time": xsol1,
                                                  "time uncertainty 1SD": xuncert68}, ignore_index=True)
        else:
            print("No data valid for this timestep.")
        # end loop
    # end loop
    
    return ens_estimates


def feedbindata(tag, index):
    """
    To bin the data for ensemble analyis, we require the have a dataframe of index length 201 (i.e., one data set for each run).
    This will be reduced based off the time of observation for the STEREO HI images
    """

    # Load dataframe
    cumulative_error = ana.load_csv_file("cumulative error",tag)
    
    # Create array of time observations. 
    # Use STEREO-B timestamp, as this is 36 seconds later
    timestamp = cumulative_error['TIME B'][0:43].to_numpy()
    
    # Create subset of dataframe based off the index of the timestamp
    if len(timestamp)<index:
        print("Index is too big. Must be less than {}".format(len(timestamp)+1))
        return
    
    timestamp_error = cumulative_error[cumulative_error["TIME B"]==timestamp[index]]
    
    return timestamp_error


    
def bindataLT(tag, index, feature="N&F", percent=100, lowestsubset=False):
    """ 
    Produces dataframes that tell us the distribution of member arrival times in terms of model resolution  
    and produces a subset of data points for anaysis, descibable by the lowest N% of RMSE
    RETURNS:
    =========
    arrival_info: Merge of the arrival information and N&F RMSE for the spacecrafts for each member
    histinfo: a dataframe detailiing the bin width (includes the first value), frequency, and n% of the frequency
    lowestmembers: a dataframe of the lowest N% of RMSE producing members
    """
    
    # Set model so we can identify the model time resoltuion.
    project_dirs = H._setup_dirs_()
    filename = "HUXt_CR2077_{}_deterministic.hdf5".format(tag)
    path = os.path.join(project_dirs['HUXt_data'], filename)
    filepath = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(filepath)
    
    # Load dataframe 
    arrival_info = ana.load_csv_file("arrival information", tag)
    cumulative_error = feedbindata(tag, index)
    cumulative_error.reset_index(inplace=True)
    
    if math.isnan(cumulative_error["RMSE combined"][0]):
        return
        

    # Ensure deterministic run is in the data
    if len(arrival_info) == 200:         
        # Add deterministic arrival information 
        arrival_info = arrival_info.append({"Transit Time" : cme.earth_transit_time.value, 
                                            "Arrival Time" : cme.earth_arrival_time.jd,
                                            "Arrival Speed" : cme.earth_arrival_speed.value}, ignore_index=True)
    
    # Combine the two dataframes for easier functionalitly. Only add the combined RMSE
    arrival_info["RMSE N&F STEREO"] = cumulative_error["RMSE combined"]
    arrival_info["HI Time B"] = cumulative_error["TIME B"]
        
    # drop invalid data that impacts statistical analysis
    for index,value in enumerate(arrival_info["Arrival Time"]):
        if value == Time('3000-01-01 00:00:00.000', format='iso').jd:
            arrival_info = arrival_info.drop([index])
    
    # Make a list of model time output, format 'jd'
    timeres = []
    for index, value in enumerate(model.time_out):
        if index == 0:
            astroTimebins = Time(model.time_init, format='jd')
        else:
            astroTimebins = astroTimebins + model.dt_out
            astroTimebins = Time(astroTimebins, format='jd')
        timeres.append(astroTimebins.value)
    
    # Find the subset of time required for binning arrival time data
    binmin = arrival_info["Arrival Time"].min()
    binmax = arrival_info["Arrival Time"].max()   
    list1 = []
    for index,value in enumerate(timeres):
        if value > binmin and value < binmax:
            list1.append(index)

    indexTmin = list1[0]-1
    indexTmax = list1[-1]+2
    timeres_subset = timeres[indexTmin: indexTmax]
    
    # Sort the dataframe in terms of RMSE N&F STEREO
    sortedRMSE_arrival = arrival_info.sort_values(by=["RMSE N&F STEREO"])
    sortedRMSE_arrival = sortedRMSE_arrival.reset_index(drop=True)
    
    # Loop throught the time res to bin data
    # Create new dataframes
    histoinfo = pd.DataFrame()
    lowestmembers = pd.DataFrame()
   
    # Loop through the bin ranges:
    for i in range(len(timeres_subset)-1):
        temp= pd.DataFrame()
        # Does the ens.member arrival fit the bin? Yes- store in temporary df
        temp = sortedRMSE_arrival[(timeres_subset[i]<=sortedRMSE_arrival["Arrival Time"]) 
                                  & (sortedRMSE_arrival["Arrival Time"]< timeres_subset[i+1])]
        # Find information for histogram
        frequency = len(temp)
        
        if lowestsubset == False:
            histoinfo = histoinfo.append({"Arrival Time Bin": (timeres_subset[i], timeres_subset[i+1]),
                                          "Frequency": frequency}, ignore_index=True)
            returnDF = arrival_info, timeres_subset, histoinfo
        
        elif lowestsubset == True: 
            if percent <= 100 and percent >= 0:
                percent_freq = int(np.round(frequency*(percent/100), decimals=0))
            else:
                print("Percentage value given must be between 0 - 100.")
                percent_freq = int(frequency)
            histoinfo = histoinfo.append({"Arrival Time Bin": (timeres_subset[i], timeres_subset[i+1]),
                                          "Frequency": frequency,
                                          "{}%".format(percent): percent_freq}, ignore_index=True)
            # Find the members with lowest RMSE
            lowestmembers = lowestmembers.append(temp[0:percent_freq])
            
            returnDF = arrival_info, timeres_subset, histoinfo, lowestmembers
    
    return returnDF