"""
This script includes functions that has been used in the analysis of ensembles runs on HUXt.


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





def load_csv_file(file, tag=''):
    """
    Load data files with important values for ananysis of ensemble forecasting 
    
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




def bindata(tag, feature, percent=100, lowestsubset=False):
    """ 
    Produces dataframes that tell us the distribution of member arrival times in terms of model resolution  
    and produces a subset of data points for anaysis, descibable by the lowest N% of RMSE
    
    Function is used in paper_RMSEcompare( )
    
    
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
    arrival_info = load_csv_file("arrival information", tag)
    errorA, errorB = load_csv_file("errors",tag)

    # Ensure deterministic run is in the data
    if len(arrival_info) == 200:         
        # Add deterministic arrival information 
        arrival_info = arrival_info.append({"Transit Time" : cme.earth_transit_time.value, 
                                            "Arrival Time" : cme.earth_arrival_time.jd,
                                            "Arrival Speed" : cme.earth_arrival_speed.value}, ignore_index=True)
    
    # Combine the two dataframes for easier functionalitly.
    arrival_info['RMSE N&F A'] = errorA['RMSE N & F error']
    arrival_info['RMSE nose A'] = errorA['RMSE nose error']
    arrival_info['RMSE flank A'] = errorA['RMSE flank error']
    arrival_info['RMSE sec.flank A'] = errorA['RMSE sec flank error']
    
    arrival_info['RMSE N&F B'] = errorB['RMSE N & F error']
    arrival_info['RMSE nose B'] = errorB['RMSE nose error']
    arrival_info['RMSE flank B'] = errorB['RMSE flank error']
    arrival_info['RMSE sec.flank B'] = errorB['RMSE sec flank error']
    
    arrival_info['RMSE N&F STEREO'] = errorA['RMSE error w/ ST-B']
    arrival_info['RMSE nose STEREO'] = errorA['RMSE nose error']+errorB['RMSE nose error']
    arrival_info['RMSE flank STEREO'] = errorA['RMSE flank error']+errorB['RMSE flank error']
    
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
    sortedRMSE_arrival = arrival_info.sort_values(by=["RMSE {} STEREO".format(feature)])
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





# Define true objective function 
def objective(x, a, b, c):
    """
    Quadratic curve function
    
    Function is used in paper_RMSEcompare( ).
    """
    return b * x + a * x**2 + c





# Define error of true objective function 
def errorofobjective(x, a, b, c):
    """
    Error of coefficients from the quadratic curve function
    
    Function is used in xsolution( ).
    
    args:
    ------
    x: x value
    a: standard devation of a coeffecient
    b: standard devation of b coeffecient
    c: standard devation of c coeffecient
    """  
    return (a**2 * x**4) + (b**2 * x**2) + (c**2)





def quadraticcurvefit(x,y, timeres_stubset,obsACE):
    """
    Using non-linear least square to fit a curve to the data. Blog post by
    https://machinelearningmastery.com/curve-fitting-with-python/
    
    Function is used in paper_RMSEcompare( )
    
    Return:
    ---------
    prints: Curve equation and solutions
    a,b,c: Intergers of the quadratic curve
    df: uncertainty of f(x) where x is equal to model time resolution. Push forwarded from xsolution() return.
    """
    try:
        # curve fit
        popt, pcov = curve_fit(objective,x,y)
    
        #summerise parameter values
        a,b,c = popt
    
        curve = "{:.3f}x^2 + {:.3f}x + {:.3f}".format(a,b,c)
        print ("The curve has fit f(x) = {}.".format(curve))
    
        fx_df, x_solution = xsolution(popt,pcov,timeres_stubset, obsACE)
        
    except TypeError:
        print("The scatter has no curve fit")
        popt = float('NaN'), float('NaN'), float('NaN')
        fx_df = pd.DataFrame()
        x_solution = float('NaN'), float('NaN'), float('NaN')
            
    return popt, fx_df, x_solution





def xsolution(popt,pcov, xbinvalues, obsACE):
    """ 
    Calculate the real x solution of a quadratic curve, plus the uncertainty within 1SD or 2SD of the coefficient uncertainty.
    
    Used in quadraticcurvefit().
    
    Returns:
    ----------
    print: the x solution and uncertainty.
    fxdf: uncertainty of f(x) where x is equal to model time resolution
    
    """
    # Find x solution
    a,b,c = popt
    
    # Function to print the Maximum and Minimum values of the quadratic function
    # Calculate the value of second part
    secondPart = c * 1.0 - (b * b / (4.0 * a));
    # calculate the discriminant
    d = (b**2) - (4*a*c)
    # find two solutions
    sol1 = (-b-cmath.sqrt(d))/(2*a)
    sol2 = (-b+cmath.sqrt(d))/(2*a)
    
    xsol1 = np.real(sol1)
    xsol2 = np.real(sol2)
    
    # Find uncertainty of 1sd of quadratic coeffecience
    perr = np.sqrt(np.diag(pcov))
    asd, bsd, csd = perr
    
    # Define xvalue range 
    xvalues = xbinvalues 
    for i in range(len(xvalues)):
        xvalues[i] = (xvalues[i] - obsACE) * 24
    xvalues.insert(0, xsol1)
    
    # Compute curve function
    fxdf = pd.DataFrame()
    for index, value in enumerate(xvalues):
        # Solve equations for y value and y error
        y = objective(value, a, b, c)
        y1error = errorofobjective(value, asd, bsd, csd)
        y2error = errorofobjective(value, (2*asd), (2*bsd), (2*csd))
        
        if index == 0:
        # create variable to compare values with minimum to see if signigicantly different.
            xminy1SD = y + y1error
            xminy2SD = y + y2error
    
        # Add information to a dataframe
        fxdf = fxdf.append({"x value":value,
                            "f(x)": y,
                            "uncertainty 1SD": y1error,
                            "uncertainty 2SD": y2error,
                            "error difference 1SD": (y - y1error) - xminy1SD, 
                            "error difference 2SD": (y - y2error) - xminy2SD}, ignore_index=True)
        
    fxdf = fxdf[["x value", "f(x)", "uncertainty 1SD", "error difference 1SD", "uncertainty 2SD", "error difference 2SD"]]
    
    try:
        # Find x value of smallest postive error difference
        min_error_diff_1sd = fxdf.loc[fxdf['error difference 1SD'] > 0, 'error difference 1SD'].min()
        x_error_diff_1sd = fxdf["x value"][fxdf["error difference 1SD"] == min_error_diff_1sd].values[0]
    except IndexError:
        x_error_diff_1sd = float('NaN')

    try:
        min_error_diff_2sd = fxdf.loc[fxdf['error difference 2SD'] > 0, 'error difference 2SD'].min()
        x_error_diff_2sd = fxdf["x value"][fxdf["error difference 2SD"] == min_error_diff_2sd].values[0]
    except IndexError:
        x_error_diff_2sd = float('NaN')

    # Calculate x uncertainty
    xuncertainty1 = np.abs(xsol1 - x_error_diff_1sd)
    xuncertainty2 = np.abs(xsol1 - x_error_diff_2sd)

    print("The x solution is {:.3f}units ±{:.3f}units with 68% CI or ±{:.3f}units for 95% CI. Units are either hours for time for km/s for speed".format(xsol1, xuncertainty1, xuncertainty2))
    print("At the curve minimum, the RMSE (i.e., the y value) is {:.3f}˚".format(fxdf["f(x)"][0]))   # Since x-solution was the first value in the list of xvalues to be solved

    x_solution = xsol1, xuncertainty1, xuncertainty2
    
    return fxdf, x_solution



#------------------------------------
#---------- PAPER PLOTS -------------
#------------------------------------


def paper_RMSEcompare(tag1, tag2, obsACE, feature, spacecraft, xaxis="Time", percent=100, save=False):
    """
    This function produces a 2x1 figure, with both plots showing a scatter plot for the RMSE (of HI1 time-elongation 
    profile) against arrival error (either Time or Speed at L1) for all ensemble members.
    A subset of data can be defined, using percent. Then, the lowest <percent> of binned data is located. 
    A quadratic line of best fit is found to the subset of data, and the value that minimises the curve is returned.
    
    Function is used in EnsembleAnalysis.ipybn.
    
    
    params:
    --------
    tag1: The ensemble tag name for the leftside plot. i.e., "12Dec08_n=200_r=30_PS=1"
    tag2: The ensemble tag name for the rightside plot. i.e., "12Dec08_n=200_r=30_PS=2"
    obsACE: The arrival time of the CME at L1, defined by the user.
    feature: The time-elongation profile used to get RMSE value. Either flank, nose, or N&F.
    spacecraft: The spacecraft which observed the feature. Either A, B or STEREO.
    xaxis: The variable on the x-axis. Either Time or Speed. Time is automatically set if not defined.
    percent: The percentage of data that will create the subset. Automatically, 100% is set. 
    save: If True, figure will save to repository.

    """
    
    # Check function request are valid
    if feature != "flank" and feature != "nose" and  feature != "N&F":
        print("Error: feature not valid. Try 'flank', 'nose', or 'N&F'.")
        return
    if spacecraft != "A" and spacecraft != "B" and spacecraft != "STEREO":
        print("Error: spacecraft not valid. Try 'A', 'B', or 'STEREO'.")
        return
    if xaxis != "time" and xaxis!="speed":
        print("Error: xaxis variable not valid. Try 'time' or 'speed'.")
        return
    
    # Retrieve data
    arrival_info1, timeres_subset1, histoinfo1, lowestmembers1 = bindata(tag1, feature, percent, lowestsubset=True)
    arrival_info2, timeres_subset2, histoinfo2, lowestmembers2 = bindata(tag2, feature, percent, lowestsubset=True)
    
    # set plotting variables
    if xaxis == "time":
        # SUBPLOT 1
        x1 = (arrival_info1["Arrival Time"] - obsACE)*24
        y1 = (arrival_info1["RMSE {} {}".format(feature, spacecraft)])

        xlow1 = (lowestmembers1["Arrival Time"] - obsACE)*24
        ylow1 = (lowestmembers1["RMSE {} {}".format(feature, spacecraft)])

        deter1 = arrival_info1[arrival_info1["file"] == "deterministic"]
        deterATE1 = (deter1["Arrival Time"][0] - obsACE)*24
        
    if xaxis == "speed":
        x1 = (arrival_info1["Arrival Speed Error (km/s)"])
        y1 = (arrival_info1["RMSE {} {}".format(feature, spacecraft)])
        
        xlow1 = (lowestmembers1["Arrival Speed Error (km/s)"])
        ylow1 = (lowestmembers1["RMSE {} {}".format(feature, spacecraft)])

        deter1 = arrival_info1[arrival_info1["file"] == "deterministic"]
        deterATE1 = (deter1["Arrival Speed Error (km/s)"][0])
        
    
    # Find quadratic curve of regression    
    popt, fx_df, x_solution = quadraticcurvefit(xlow1, ylow1, timeres_subset1, obsACE)
    xsol1, xuncert68, xuncert95 = x_solution
    # define a sequence of inputs between the smallest and largest known inputs
    fxX1 = np.arange(min(x1), max(x1)+1, 0.5)
    # calculate the output for the range
    a, b, c = popt
    fxY1 = objective(fxX1, a, b, c)
    
    # SUBPLOT 2
    if xaxis == "time":
        x2 = (arrival_info2["Arrival Time"] - obsACE)*24
        y2 = (arrival_info2["RMSE {} {}".format(feature, spacecraft)])

        xlow2 = (lowestmembers2["Arrival Time"] - obsACE)*24
        ylow2 = (lowestmembers2["RMSE {} {}".format(feature, spacecraft)])

        deter2 = arrival_info2[arrival_info2["file"] == "deterministic"]
        deterATE2 = (deter2["Arrival Time"][0] - obsACE)*24
    
    if xaxis == "speed":
        x2 = (arrival_info2["Arrival Speed Error (km/s)"])
        y2 = (arrival_info2["RMSE {} {}".format(feature, spacecraft)])
        
        xlow2 = (lowestmembers2["Arrival Speed Error (km/s)"])
        ylow2 = (lowestmembers2["RMSE {} {}".format(feature, spacecraft)])

        deter2 = arrival_info2[arrival_info2["file"] == "deterministic"]
        deterATE2 = (deter2["Arrival Speed Error (km/s)"][0])
    
    # Find quadratic curve of regression    
    popt, fx_df, x_solution = quadraticcurvefit(xlow2, ylow2, timeres_subset2, obsACE)
    xsol2, xuncert68, xuncert95 = x_solution
    # define a sequence of inputs between the smallest and largest known inputs
    fxX2 = np.arange(min(x2), max(x2)+1, 0.5)
    # calculate the output for the range
    a, b, c = popt
    fxY2 = objective(fxX2, a, b, c)
    
    print("Min X1 = {:.3f}({:.3f}), Max X1 = {:.3f}({:.3f})".format(x1.min(), xlow1.min(), x1.max(), xlow1.max()) )
    print("Min X2 = {:.3f}({:.3f}), Max X2 = {:.3f}({:.3f})".format(x2.min(), xlow2.min(), x2.max(), xlow2.max()))
    
    # Setup figure
    plt.rcParams.update({'font.size': 22, 'axes.labelsize':20, 'legend.fontsize':20,'xtick.labelsize': 20.0,'ytick.labelsize':20.0,"font.family":"Times New Roman"})
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
    
    
    ax1.annotate("a)", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=18)
    ax2.annotate("b)", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=18)
    
    
    ax1.tick_params(axis= 'y', which='major', direction="out", width=2, length=7)
    ax1.tick_params(axis= 'x', which='major', direction="out", width=2, length=7)
    ax1.tick_params(axis= 'y', which='minor', direction="out", width=1, length=3)
    ax1.tick_params(axis= 'x', which='minor', direction="out", width=1, length=3)
    
    ax2.tick_params(axis= 'y', which='major', direction="out", width=2, length=7)
    ax2.tick_params(axis= 'x', which='major', direction="out", width=2, length=7)
    ax2.tick_params(axis= 'y', which='minor', direction="out", width=1, length=3)
    ax2.tick_params(axis= 'x', which='minor', direction="out", width=1, length=3)
    
    if xaxis == "time":
        ax1.set_xlabel("Time Error (Hours)")
        ax2.set_xlabel("Time Error (Hours)")
        ax1.set_xlim(left=-8, right=20)
        ax2.set_xlim(left=-8, right=20)
        
        ax1.set_ylabel("RMSE (˚)")
        ax1.set_ylim(bottom=0, top=3)
        ax2.set_ylim(bottom=0, top=3)
        
        ax1.xaxis.set_major_locator(MultipleLocator(2))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.xaxis.set_major_locator(MultipleLocator(2))
        ax2.xaxis.set_minor_locator(MultipleLocator(1))
        
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
        
    elif xaxis == "speed":
        ax1.set_xlabel("Speed Error (km/s)")
        ax2.set_xlabel("Speed Error (km/s)")
        
        ax1.set_xlim(left=-20, right=100)
        ax2.set_xlim(left=-20, right=100)
        
        ax1.set_ylabel("RMSE (˚)")
        ax1.set_ylim(bottom=0, top=3)
        ax2.set_ylim(bottom=0, top=3)
        
        ax1.xaxis.set_major_locator(MultipleLocator(20))
        ax1.xaxis.set_minor_locator(MultipleLocator(5))
        ax2.xaxis.set_major_locator(MultipleLocator(20))
        ax2.xaxis.set_minor_locator(MultipleLocator(5))
        
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
    
    
    # Plot scatter
    ax1.plot(x1,y1,'.', color='darkgray', zorder=5, ms = 3)
    ax1.plot(xlow1,ylow1,'.', color='k', zorder=6)
    # Plot regression curve
    ax1.plot(fxX1, fxY1, '--', color='r', zorder=4, label = "Data best fit")
    # Plot ATE vertical markers
    ax1.axvline(x=deterATE1, color="orange", label = "Deterministic Arrival Estimate")
    ax1.axvline(x=xsol1, color="blue", label = "Ensemble Arrival Estimate")
    
    # Plot scatter
    ax2.plot(x2,y2,'.', color='darkgray', zorder=5, ms = 3)
    ax2.plot(xlow2,ylow2,'.', color='k', zorder=6)
    # Plot regression curve
    ax2.plot(fxX2, fxY2, '--', color='r', zorder=4)
    # Plot ATE vertical markers
    ax2.axvline(x=deterATE2, color="orange")
    ax2.axvline(x=xsol2, color="blue")
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.3, hspace=0.3, wspace=0.15)
    fig.legend(bbox_to_anchor=(0.5, 0.06), loc='center', ncol=3)
    
    if save:
        project_dirs = H._setup_dirs_()
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}_{}%curve.png".format(tag1,xaxis,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}_{}%curve.pdf".format(tag1,xaxis,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)           
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
    plt.show()
    
 