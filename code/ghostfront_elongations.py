
def plot_elongation_profiles(, FOV=None, sec_flank=True):
    """
    This function will plot the leading-edge elongtion angles of the HUXt-modelled CME as seen from the FOV for STEREO-A and STEREO-B. Automatically, the nose and inital flank will be shown whilst the secondary flank is an optional attrubute. The Solar Stormwatch data - showing averaged observational elongation angles - is also plotted with error bars. 
    
    :FOV: The range of angles being obserevered. Allowed: HI1, HI2.
    :sec_flank: Optional extra elongation plot that shows the elgonation of the secondary flank.
    """

if FOV == HI1:
# HI-1: maximum elongation of 25
    angle_max = 25.0
    for i in range(len(sta_profile)):
        if sta_profile.el[i] < 30:
            FOVlimit_a = i
        if stb_profile.el[i] < 30:
            FOVlimit_b = i            
elif FOV = HI2:
# HI-2: maximum elongation of ?˚
    # use the same code as above, but also include "if maximum elongation is much less than FOV limit, plot maximum elongation" to ensure the figure is filled with the data.

else:
    print ("Error: Field of view not accepted. Try using HI1 or HI2")

# Format time
time_a = sta.time.to_value('datetime')[0:FOVlimit_a]
time_b = stb.time.to_value('datetime')[0:FOVlimit_b]
myFmt = mdates.DateFormatter('%d-%m-%y %H:%M')

# Load Solar Stormwatch front data

# Plot figure
fig, (axA, axB) = plt.subplots(1, 2, figsize = [10,4.8])

axA.set_title('ST-A')
axA.set_xlabel('Time(step)')
axA.set_ylabel('Elongation (˚)')
axA.set_ylim(top=angle_max)                                 
#axA.set_xlim(right=xmax)                                # Change to time
axA.plot(time_a, sta_profile.el[0:FOVlimit_a], label='Initial Flank')
axA.plot(time_a, sta_profile.el_n[0:FOVlimit_a], 'r', label='Nose')
#axA.plot(time_a, sta_profile.el_sec_flank, 'r', label='Secondary Flank')
plt.gca().xaxis.set_major_formatter(myFmt)

axB.set_title('ST-B')
axB.set_xlabel('Time(step)')
axB.set_ylabel('Elongation (˚)')
axB.set_ylim(top=angle_max)                                 
#axB.set_xlim(right=60)                                # Change to time
axB.plot(time_b, stb_profile.el[0:FOVlimit_b], label='Initial Flank')
axB.plot(time_b, stb_profile.el_n[0:FOVlimit_b], 'r', label='Nose')
axB.plot(time_b, stb_profile.el_sec_flank[0:FOVlimit_b], 'g', label='Secondary Flank')
plt.gca().xaxis.set_major_formatter(myFmt)

if sec_flank == True:
    axA.plot(time_a, sta_profile.el_sec_flank, 'r', label='Secondary Flank')
    axB.plot(time_b, stb_profile.el_sec_flank[0:FOVlimit_b], 'g', label='Secondary Flank')

#axB.legend(loc='best', bbox_to_anchor=[1.6,0.5])
axB.legend(loc=[1.1,0.35])
fig.tight_layout()
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(myFmt)