# Copy of "Analysis.py" from the RACC. New funtions for plotting RMSE-ATE for the lowest N% of values. Copied on 14/03/2022


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
        
    elif file == "cumulative error":
        path = os.path.join(project_dirs['HUXt_data'], tag+"_cumulativeRMSE.csv")
        cumulativeerror = pd.read_csv(r"{}".format(path))
        cumulativeerror = cumulativeerror.drop(columns=['Unnamed: 0'])
        
        data = cumulativeerror
        
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
        
#--------------------------    
def RMSEscatter(feature, spacecraft, tag, tag_overlay='', overlay=False, fit_data=False, title=True, save=False):

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
    
    if spacecraft == "A":
        if feature == "nose" or feature == "flank" or feature == "N & F":
            y = errorsA["RMSE {} error".format(feature)]
        elif feature == "sec flank":
            print("Secondary flank not available from this viewpoint.")
        else:
            print("Error: Feature not valid.")
    
    elif spacecraft == "B":
        if feature == "nose" or feature == "flank" or feature == "N & F" or feature == "sec flank":
            y = errorsB["RMSE {} error".format(feature)]
        else:
            print("Error: Feature not valid.")
    
    elif spacecraft == "STEREO":
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
    
    # ================ 
    if title == True:
        plt.text(19.8,3.1, '{} from {}'.format(feature, view), fontsize=14, ha='right', weight='bold')
    
    # ================   
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
        
        plt.legend(["2x Parameter Uncertainty", "1x Parameter Uncertainty"], bbox_to_anchor=(0.5, 0.06),
                   loc='center', ncol=2, fontsize=10)
    
    # ================ 
    if fit_data == True:
        """
        Using non-linear least square to fit a curve to the data.Blog post by
        https://machinelearningmastery.com/curve-fitting-with-python/ 
        """

        # Define true objective function 
        def objective(x, a, b, c):
            return b * x + a * x**2 + c

        # curve fit
        popt, _ = curve_fit(objective,x,y)
        #summerise parameter values
        a,b,c = popt
        print('y = %.5f * x + %.5f * x^2 + %.5f' % (b, a, c))
        def PrintMaxMinValue(a, b, c) :
            # Function to print the Maximum and Minimum values of the quadratic function
            # Calculate the value of second part
            secondPart = c * 1.0 - (b * b / (4.0 * a));

            # Print the values
            if (a > 0) :
                # Open upward parabola function
                print("MaxYvalue =", "Infinity");
                print("MinYvalue = ", secondPart);

            elif (a < 0) :
                # Open downward parabola function
                print("MaxYvalue = ", secondPart);
                print("MinYvalue =", "-Infinity");

            else :
                # If a=0 then it is not a quadratic function
                print("Not a quadratic function");
                
            # calculate the discriminant
            d = (b**2) - (4*a*c)

            # find two solutions
            sol1 = (-b-cmath.sqrt(d))/(2*a)
            sol2 = (-b+cmath.sqrt(d))/(2*a)

            print('The solution are {0} and {1}'.format(sol1,sol2))
            
            return
        PrintMaxMinValue(a, b, c)
        
        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.arange(min(x), max(x), 0.5)
        # calculate the output for the range
        y_line = objective(x_line, a, b, c)
        # create a line plot for the mapping function
        plt.plot(x_line, y_line, '--', color='red')
        plt.show()

    
    project_dirs = H._setup_dirs_()
    
    # ================ 
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


#--------------

def bindata(tag, feature, percent=100, lowestsubset=False):
    """ Produces dataframes that tell us the distribution of member arrival times in terms of model resolution  
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
    """
    return b * x + a * x**2 + c

# Define error of true objective function 
def errorofobjective(x, a, b, c):
    """
    Error of coefficients from the quadratic curve function
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
    Using non-linear least square to fit a curve to the data.Blog post by
    https://machinelearningmastery.com/curve-fitting-with-python/
    
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
    """ Calculate the real x solution of a quadratic curve, plus the uncertainty within 1SD or 2SD of the coefficient uncertainty.
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



    
def RMSEandATEregression(tag, obsACE, feature, spacecraft, percent=100, plot=False, save=False):
    """
    
    """
    # Check data requests are valid
    if feature != "flank" and feature != "nose" and  feature != "N&F":
        print("Error: feature not valid. Try 'flank', 'nose', or 'N&F'.")
        return
    if spacecraft != "A" and spacecraft != "B" and spacecraft != "STEREO":
        print("Error: spacecraft not valid. Try 'A', 'B', or 'STEREO'.")
        return
    
    # Retrieve required data
    arrival_info, timeres_subset, histoinfo, lowestmembers = bindata(tag, feature, percent, lowestsubset=True)
    
    # Assign x and y variables
    x = (arrival_info["Arrival Time"] - obsACE)*24
    y = (arrival_info["RMSE {} {}".format(feature, spacecraft)])
    
    xlow = (lowestmembers["Arrival Time"] - obsACE)*24
    ylow = (lowestmembers["RMSE {} {}".format(feature, spacecraft)])
    
    if len(ylow) == 0:
        print("Error: There is no values in the subset to be analysed. Try with a larger percentage.")
        return
    
    # Find quadratic curve of regression    
    popt, fx_df, x_solution = quadraticcurvefit(xlow, ylow, timeres_subset, obsACE)
    xsol, xuncert68, xuncert95 = x_solution
    a,b,c = popt
    curve = "{:.3f}x^2 + {:.3f}x + {:.3f}".format(a,b,c)
    
    # Put important information in a dataframe
    xsolution_df = pd.DataFrame()
    xsolution_df = xsolution_df.append({"data length": len(xlow),
                                       "% of total data": percent,
                                       "x solution": xsol,
                                       "x uncertainty 1SD": xuncert68, 
                                       "x uncertainty 2SD": xuncert95, 
                                       "f(x) eqn.": curve}, ignore_index=True)
    
    xsolution_df = xsolution_df[["% of total data", "data length", "x solution","x uncertainty 1SD","x uncertainty 2SD","f(x) eqn."]]
    
    if plot:
        # Find determinsitc ATE
        deter = arrival_info[arrival_info["file"] == "deterministic"]
        deterATE = (deter["Arrival Time"][0] - obsACE)*24
        
        information = tag, feature, spacecraft, percent, deterATE, xsol
        lowestmembersplot(x,y,xlow,ylow,popt,fx_df,information,save)
    
    return xsolution_df
    

def lowestmembersplot(x1, y1, x2, y2,popt, fx_df, information, save=False):
    """
    
    """
    a,b,c = popt
    tag, feature, spacecraft, percent, deterATE, xsol = information
    
    # Figure setup
    plt.figure()
    plt.xlabel("Time Error (Hours)")
    plt.ylabel("RMSE")
    plt.xlim(left=x1.min()-1, right=x1.max()+1)
    plt.minorticks_on()
    plt.ylim(bottom=y1.min()-0.2, top=y1.max()+0.2)
#     plt.set_major_locator(MultipleLocator(5))
#     plt.set_minor_locator(MultipleLocator(1))
    plt.title("{} RMSE against arrival time error for spacecraft {}".format(feature, spacecraft))
    
    # Plot scatter
    plt.scatter(x1,y1, color='darkgray', s=2, zorder=5)
    plt.scatter(x2,y2, color='k', s=2, zorder=6)
    
    # Plot regression curve
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x1), max(x1)+1, 0.5)
    # calculate the output for the range
    y_line = objective(x_line, a, b, c)
    # create a line plot for the mapping function
    plt.plot(x_line, y_line, '--', color='r', zorder=4)
    
    # Plot curve uncertainty
    yhigh = fx_df["f(x)"] + fx_df["uncertainty 1SD"]
    ylow = fx_df["f(x)"] - fx_df["uncertainty 1SD"]
    plt.fill_between(fx_df["x value"], ylow, yhigh, facecolor='peachpuff', zorder=1)
    
    # Plot ATE vertical markers
    plt.axvline(x=deterATE, color="orange")
    plt.axvline(x=xsol, color="blue")
    
    if save:
        project_dirs = H._setup_dirs_()
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}%curve.png".format(tag,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}%curve.pdf".format(tag,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)           
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()
    return
    
    
def lowestmemberplot(tag, obsACE, feature, spacecraft, percent=100, curve=False, save=False):
    """
    Produces the scatter plot of all members and curve fit to the lower N% of the members.
    Prints quadratic curve equation and solutions 
    """
    if feature != "flank" and feature != "nose" and  feature != "N&F":
        print("Error: feature not valid. Try 'flank', 'nose', or 'N&F'.")
        return
    if spacecraft != "A" and spacecraft != "B" and spacecraft != "STEREO":
        print("Error: spacecraft not valid. Try 'A', 'B', or 'STEREO'.")
        return
    
    # Run required functions
    arrival_info, timeres_subset, histoinfo, lowestmembers = bindata(tag, percent, lowestsubset=True)
    
    #Plot figure
    x = (arrival_info["Arrival Time"] - obsACE)*24
    y = (arrival_info["RMSE {} {}".format(feature, spacecraft)])
    
    xlow = (lowestmembers["Arrival Time"] - obsACE)*24
    ylow = (lowestmembers["RMSE {} {}".format(feature, spacecraft)])
    
    if len(ylow) == 0:
        print("Error: There is no values in the subset to be plotted. Try with a larger percentage.")
        
        return
    
    plt.figure()
    plt.scatter(x,y, color='darkgray', s=2, zorder=5)
    plt.scatter(xlow,ylow, color='k', s=2, zorder=6)

    plt.xlabel("Time Error (Hours)")
    plt.ylabel("RMSE")
    plt.xlim(left=x.min()-1, right=x.max()+1)
    plt.ylim(bottom=y.min()-0.2, top=y.max()+0.2)
    
    plt.title("{} RMSE against arrival time error for spacecraft {}".format(feature, spacecraft))
    
    if curve == True:
        a,b,c, df = quadraticcurvefit(xlow, ylow, timeres_subset, obsACE)
    
        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.arange(min(x), max(x), 0.5)
        # calculate the output for the range
        y_line = objective(x_line, a, b, c)
        # create a line plot for the mapping function
        plt.plot(x_line, y_line, '--', color='r', zorder=4)
        
#         yminussd = df["f(x)"] - df["f(x-1sd)"]
#         yplussd = df["f(x+1sd)"] - df["f(x)"]
#         plt.errorbar(df["x value"],df["f(x)"], yerr = (yminussd, yplussd),fmt='none', color='skyblue')

        plt.fill_between(df["x value"], df["f(x-1sd)"], df["f(x+1sd)"], facecolor='peachpuff', zorder=1)
    
    # ================ 
    if save:
        project_dirs = H._setup_dirs_()
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}%curve.png".format(tag,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}%curve.pdf".format(tag,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)           
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return



#------------------------------------
#---------- PAPER PLOTS -------------
#------------------------------------

def paper_ens_bar(tag, obsACE, save=False, tag_title=False):
    
    # Set model
    project_dirs = H._setup_dirs_()
    filename = "HUXt_CR2077_{}_deterministic.hdf5".format(tag)
    path = os.path.join(project_dirs['HUXt_data'], filename)
    filepath = glob.glob(path)[0]
    model, cme_list = H.load_HUXt_run(filepath)
    
    # Load info
    arrival_info, timeres_subset, histoinfo  = bindata(tag)
    A, B, MISC = load_csv_file("best fit run", tag)
    bins = timeres_subset

    # Assign time to variables
    #---------------------------
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
    
    plt.bar((bins[0:len(bins)-1]-obsACE)* 24, (histoinfo["Frequency"].values), 
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


def paper_RMSEcompare(tag1, tag2, obsACE, feature, spacecraft, xaxis, percent=100, save=False):
    """
    
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
    
#     if len(ylow1) == 0:
#         print("Error: There is no values in the subset to be analysed. Try with a larger percentage.")
#         return

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
    
#     ax1.text(xsol1, 3.1, "Ensemble ATE", rotation=30, fontsize=14)
#     ax1.text(deterATE1, 3.1, "Deterministic ATE", rotation=30, fontsize=14)
            
#     ax2.text(xsol2, 3.1, "Ensemble ATE", rotation=30, fontsize=14)
#     ax2.text(deterATE2, 3.1, "Deterministic ATE", rotation=30, fontsize=14)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.3, hspace=0.3, wspace=0.15)
    fig.legend(bbox_to_anchor=(0.5, 0.06), loc='center', ncol=3)
#     plt.tight_layout()
    
    if save:
        project_dirs = H._setup_dirs_()
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}_{}%curve.png".format(tag1,xaxis,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_{}_ens_scat_plot_{}_{}%curve.pdf".format(tag1,xaxis,feature, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)           
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
def paper_RMSEATEdouble(tag1, tag2, tag3, tag4, obsACE, feature, spacecraft,xaxis, percent=100, save=False):
    """
    
    """
    # Check function request are valid
    if feature != "flank" and feature != "nose" and  feature != "N&F":
        print("Error: feature not valid. Try 'flank', 'nose', or 'N&F'.")
        return
    if spacecraft != "A" and spacecraft != "B" and spacecraft != "STEREO":
        print("Error: spacecraft not valid. Try 'A', 'B', or 'STEREO'.")
        return
    if xaxis != "time" and xaxis != "speed":
        print("Error: xaxis variable not valid. Try 'time' or 'speed'.")
        return
    
    # Retrieve data
    arrival_info1, timeres_subset1, histoinfo1, lowestmembers1 = bindata(tag1, feature, percent, lowestsubset=True)
    arrival_info2, timeres_subset2, histoinfo2, lowestmembers2 = bindata(tag2, feature, percent, lowestsubset=True)
    arrival_info3, timeres_subset3, histoinfo3, lowestmembers3 = bindata(tag3, feature, percent, lowestsubset=True)
    arrival_info4, timeres_subset4, histoinfo4, lowestmembers4 = bindata(tag4, feature, percent, lowestsubset=True)
    
    # set plotting variables 
    
    if xaxis == "time":
        # SUBPLOT 1
        x1 = (arrival_info1["Arrival Time"] - obsACE)*24
        y1 = (arrival_info1["RMSE {} {}".format(feature, spacecraft)])

        xlow1 = (lowestmembers1["Arrival Time"] - obsACE)*24
        ylow1 = (lowestmembers1["RMSE {} {}".format(feature, spacecraft)])

        deter1 = arrival_info1[arrival_info1["file"] == "deterministic"]
        deterATE1 = (deter1["Arrival Time"][0] - obsACE)*24
        
        # SUBPLOT 2
        x2 = (arrival_info2["Arrival Time"] - obsACE)*24
        y2 = (arrival_info2["RMSE {} {}".format(feature, spacecraft)])

        xlow2 = (lowestmembers2["Arrival Time"] - obsACE)*24
        ylow2 = (lowestmembers2["RMSE {} {}".format(feature, spacecraft)])

        deter2 = arrival_info2[arrival_info2["file"] == "deterministic"]
        deterATE2 = (deter2["Arrival Time"][0] - obsACE)*24
        
        # SUBPLOT 3
        x3 = (arrival_info3["Arrival Time"] - obsACE)*24
        y3 = (arrival_info3["RMSE {} {}".format(feature, spacecraft)])

        xlow3 = (lowestmembers3["Arrival Time"] - obsACE)*24
        ylow3 = (lowestmembers3["RMSE {} {}".format(feature, spacecraft)])

        deter3 = arrival_info3[arrival_info3["file"] == "deterministic"]
        deterATE3 = (deter3["Arrival Time"][0] - obsACE)*24
        
        # SUBPLOT 4
        x4 = (arrival_info4["Arrival Time"] - obsACE)*24
        y4 = (arrival_info4["RMSE {} {}".format(feature, spacecraft)])

        xlow4 = (lowestmembers4["Arrival Time"] - obsACE)*24
        ylow4 = (lowestmembers4["RMSE {} {}".format(feature, spacecraft)])

        deter4 = arrival_info4[arrival_info4["file"] == "deterministic"]
        deterATE4 = (deter4["Arrival Time"][0] - obsACE)*24
    
   

    elif xaxis == "speed":
        x1 = (arrival_info1["Arrival Speed Error (km/s)"])
        y1 = (arrival_info1["RMSE {} {}".format(feature, spacecraft)])
        
        xlow1 = (lowestmembers1["Arrival Speed Error (km/s)"])
        ylow1 = (lowestmembers1["RMSE {} {}".format(feature, spacecraft)])

        deter1 = arrival_info1[arrival_info1["file"] == "deterministic"]
        deterATE1 = (deter1["Arrival Speed Error (km/s)"][0])
        
        x2 = (arrival_info2["Arrival Speed Error (km/s)"])
        y2 = (arrival_info2["RMSE {} {}".format(feature, spacecraft)])
        
        xlow2 = (lowestmembers2["Arrival Speed Error (km/s)"])
        ylow2 = (lowestmembers2["RMSE {} {}".format(feature, spacecraft)])

        deter2 = arrival_info2[arrival_info2["file"] == "deterministic"]
        deterATE2 = (deter2["Arrival Speed Error (km/s)"][0])
        
        x3 = (arrival_info3["Arrival Speed Error (km/s)"])
        y3 = (arrival_info3["RMSE {} {}".format(feature, spacecraft)])
        
        xlow3 = (lowestmembers3["Arrival Speed Error (km/s)"])
        ylow3 = (lowestmembers3["RMSE {} {}".format(feature, spacecraft)])

        deter3 = arrival_info3[arrival_info3["file"] == "deterministic"]
        deterATE3 = (deter3["Arrival Speed Error (km/s)"][0])
        
        x4 = (arrival_info4["Arrival Speed Error (km/s)"])
        y4 = (arrival_info4["RMSE {} {}".format(feature, spacecraft)])
        
        xlow4 = (lowestmembers4["Arrival Speed Error (km/s)"])
        ylow4 = (lowestmembers4["RMSE {} {}".format(feature, spacecraft)])

        deter4 = arrival_info4[arrival_info2["file"] == "deterministic"]
        deterATE4 = (deter4["Arrival Speed Error (km/s)"][0])



    # Find quadratic curve of regression    
    popt, fx_df, x_solution = quadraticcurvefit(xlow1, ylow1, timeres_subset1, obsACE)
    xsol1, xuncert68, xuncert95 = x_solution
    # define a sequence of inputs between the smallest and largest known inputs
    fxX1 = np.arange(min(x1), max(x1)+1, 0.5)
    # calculate the output for the range
    a, b, c = popt
    fxY1 = objective(fxX1, a, b, c)
 
    
    # Find quadratic curve of regression    
    popt, fx_df, x_solution = quadraticcurvefit(xlow2, ylow2, timeres_subset2, obsACE)
    xsol2, xuncert68, xuncert95 = x_solution
    # define a sequence of inputs between the smallest and largest known inputs
    fxX2 = np.arange(min(x2), max(x2)+1, 0.5)
    # calculate the output for the range
    a, b, c = popt
    fxY2 = objective(fxX2, a, b, c)
    
    
    # Find quadratic curve of regression    
    popt, fx_df, x_solution = quadraticcurvefit(xlow3, ylow3, timeres_subset3, obsACE)
    xsol3, xuncert68, xuncert95 = x_solution
    # define a sequence of inputs between the smallest and largest known inputs
    fxX3 = np.arange(min(x3), max(x3)+1, 0.5)
    # calculate the output for the range
    a, b, c = popt
    fxY3 = objective(fxX3, a, b, c)
    
    # Find quadratic curve of regression    
    popt, fx_df, x_solution = quadraticcurvefit(xlow4, ylow4, timeres_subset4, obsACE)
    xsol4, xuncert68, xuncert95 = x_solution
    # define a sequence of inputs between the smallest and largest known inputs
    fxX4 = np.arange(min(x4), max(x4)+1, 0.5)
    # calculate the output for the range
    a, b, c = popt
    fxY4 = objective(fxX4, a, b, c)
    
    
    # Setup figure
    plt.rcParams.update({'font.size': 22, 'axes.labelsize':26, 'legend.fontsize':26, 'xtick.labelsize': 24.0, 'ytick.labelsize': 24.0, "font.family":"Times New Roman"}) # POSTER SIZE FONT
#     plt.rcParams.update({'font.size': 22, 'axes.labelsize':20, 'legend.fontsize':16,'xtick.labelsize': 16.0,'ytick.labelsize': 16.0,"font.family":"Times New Roman"})  # ARTICLE SIZE FONT
   
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,10))
    
    # Specific settings for figure
    
    if xaxis == "time":
        
        if x2.min() < x4.min():
            xlimmin = x2.min()-1
        else:
            xlimmin = x4.min()-1

        if x2.max() > x4.max():
            xlimmax = x2.max()+1
        else:
            xlimmax = x4.max()+1
        
        ax1.xaxis.set_major_locator(MultipleLocator(5))
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax2.xaxis.set_major_locator(MultipleLocator(5))
        ax2.xaxis.set_minor_locator(MultipleLocator(1))
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax3.xaxis.set_major_locator(MultipleLocator(5))
        ax3.xaxis.set_minor_locator(MultipleLocator(1))
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax4.xaxis.set_major_locator(MultipleLocator(5))
        ax4.xaxis.set_minor_locator(MultipleLocator(1))
        ax4.yaxis.set_major_locator(MultipleLocator(0.5))
        ax4.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax3.set_xlabel("Time Error (Hours)")
        ax4.set_xlabel("Time Error (Hours)")
        
    elif xaxis == "speed":
        
        if x2.min() < x4.min():
            xlimmin = x2.min()-10
        else:
            xlimmin = x4.min()-10

        if x2.max() > x4.max():
            xlimmax = x2.max()+10
        else:
            xlimmax = x4.max()+10
            
        ax1.xaxis.set_major_locator(MultipleLocator(20))
        ax1.xaxis.set_minor_locator(MultipleLocator(5))
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax2.xaxis.set_major_locator(MultipleLocator(20))
        ax2.xaxis.set_minor_locator(MultipleLocator(5))
        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax3.xaxis.set_major_locator(MultipleLocator(20))
        ax3.xaxis.set_minor_locator(MultipleLocator(5))
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax4.xaxis.set_major_locator(MultipleLocator(20))
        ax4.xaxis.set_minor_locator(MultipleLocator(5))
        ax4.yaxis.set_major_locator(MultipleLocator(0.5))
        ax4.yaxis.set_minor_locator(MultipleLocator(0.25))
        
        ax3.set_xlabel("Speed Error (km/s)")
        ax4.set_xlabel("Speed Error (km/s)")

    # General Settings for figure
    ax1.set_ylabel("RMSE (˚)", labelpad = 20)
    ax3.set_ylabel("RMSE (˚)", labelpad = 20)
    ax1.set_ylim(bottom=0, top=3)
    ax2.set_ylim(bottom=0, top=3)
    ax3.set_ylim(bottom=0, top=3)
    ax4.set_ylim(bottom=0, top=3)
    
    
    ax1.set_xlim(left=xlimmin, right=xlimmax)
    ax2.set_xlim(left=xlimmin, right=xlimmax)
    ax3.set_xlim(left=xlimmin, right=xlimmax)
    ax4.set_xlim(left=xlimmin, right=xlimmax)
    
    ax1.annotate("a)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=22)
    ax2.annotate("b)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=22)
    ax3.annotate("c)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=22)
    ax4.annotate("d)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=22)
    
    # Plot scatter
    ax1.plot(x1,y1,'.', color='darkgray', zorder=5, ms = 3)
    ax1.plot(xlow1,ylow1,'.', color='k', zorder=6)
    # Plot regression curve
    ax1.plot(fxX1, fxY1, '--', color='r', zorder=4, label = "Data best fit")
    # Plot ATE vertical markers
    ax1.axvline(x=deterATE1, color="orange")
    ax1.axvline(x=xsol1, color="blue")
    
    # Plot scatter
    ax2.plot(x2,y2,'.', color='darkgray', zorder=5, ms = 3)
    ax2.plot(xlow2,ylow2,'.', color='k', zorder=6)
    # Plot regression curve
    ax2.plot(fxX2, fxY2, '--', color='r', zorder=4)
    # Plot ATE vertical markers
    ax2.axvline(x=deterATE2, color="orange")
    ax2.axvline(x=xsol2, color="blue")
    
    # Plot scatter
    ax3.plot(x3,y3,'.', color='darkgray', zorder=5, ms = 3)
    ax3.plot(xlow3,ylow3,'.', color='k', zorder=6)
    # Plot regression curve
    ax3.plot(fxX3, fxY3, '--', color='r', zorder=4)
    # Plot ATE vertical markers
    ax3.axvline(x=deterATE3, color="orange")
    ax3.axvline(x=xsol3, color="blue")
    
    # Plot scatter
    ax4.plot(x4,y4,'.', color='darkgray', zorder=5, ms = 3)
    ax4.plot(xlow4,ylow4,'.', color='k', zorder=6)
    # Plot regression curve
    ax4.plot(fxX4, fxY4, '--', color='r', zorder=4)
    # Plot ATE vertical markers
    ax4.axvline(x=deterATE4, color="orange", label = "Deterministic arrival estimate")
    ax4.axvline(x=xsol4, color="blue", label = "Ensemble arrival estimate")
    
    ax1.tick_params(axis= 'y', which='major', direction="out", width=2, length=7)
    ax1.tick_params(axis= 'x', which='major', direction="out", width=2, length=7)
    ax1.tick_params(axis= 'y', which='minor', direction="out", width=1, length=3)
    ax1.tick_params(axis= 'x', which='minor', direction="out", width=1, length=3)
    
    ax2.tick_params(axis= 'y', which='major', direction="out", width=2, length=7)
    ax2.tick_params(axis= 'x', which='major', direction="out", width=2, length=7)
    ax2.tick_params(axis= 'y', which='minor', direction="out", width=1, length=3)
    ax2.tick_params(axis= 'x', which='minor', direction="out", width=1, length=3)
    
    ax3.tick_params(axis= 'y', which='major', direction="out", width=2, length=7)
    ax3.tick_params(axis= 'x', which='major', direction="out", width=2, length=7)
    ax3.tick_params(axis= 'y', which='minor', direction="out", width=1, length=3)
    ax3.tick_params(axis= 'x', which='minor', direction="out", width=1, length=3)
    
    ax4.tick_params(axis= 'y', which='major', direction="out", width=2, length=7)
    ax4.tick_params(axis= 'x', which='major', direction="out", width=2, length=7)
    ax4.tick_params(axis= 'y', which='minor', direction="out", width=1, length=3)
    ax4.tick_params(axis= 'x', which='minor', direction="out", width=1, length=3)
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25, hspace=0.1, wspace=0.05)
#     plt.subplots_adjust(bottom=0.20, hspace=0.4)
    fig.legend(bbox_to_anchor=(0.5, 0.06), loc='center', ncol=3)
    
    if save:
        project_dirs = H._setup_dirs_()
        
        filename = "HUXt_{}_{}_{}_ens_scat_plot_{}%curve.png".format(tag1,feature,xaxis, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)            
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        filename = "HUXt_{}_{}_{}_ens_scat_plot_{}%curve.pdf".format(tag1,feature,xaxis, percent)
        filepath = os.path.join(project_dirs['HUXt_figures'], filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
    plt.show()
    
    
    

    