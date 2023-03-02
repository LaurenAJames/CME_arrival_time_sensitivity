import os
import pandas as pd
import matplotlib as mpl
import hi_processing.images as hip
import astropy.units as u
from datetime import datetime, timedelta
from PIL import Image


def get_cme_details(helcats_name):
    """Returns the craft and date of cme helcats_name.

    Parameters:
    helcats_name (str): helcats id of cme e.g. HCME_B__20130830_01
    
    Returns:
    craft (str): either 'sta' or 'stb'
    date (datetime): first date cme observed in HI1 by HELCATS
    """
    if helcats_name[5] == 'A':
        craft = 'sta'
    elif helcats_name[5] == 'B':
        craft = 'stb'
    date = pd.datetime.strptime(helcats_name[8:16], '%Y%m%d')
    return craft, date


def make_diff_img(helcats_name, time, camera='hi1', background_type=1,
                  save=False):
    """Makes the differenced image for the required CME at time specified
    using the .FTS files on the hard-drives.
    
    Parameters:
    helcats_name (str): HELCATS id for CME e.g.'HCME_B__20081212_01',
                            'HCME_A__20081212_01'
    time (datetime): datetime object with date and time
    craft (str): "sta" or "stb"
    camera (str): "hi1" or "hi2"
    background_type (int): 1 or 11
    save (str): False or r"path/to/folder" to save into
    
    Returns:
    out_img (Image): differenced image
    hi_map (Sunpy Map): contains coordinate info for the image
    """
    craft, hi1_start_date = get_cme_details(helcats_name)
    out_img = False
    out_name = False
    hi_map = False
    # Search hard drive for matching HI image
    # to create a differenced image, need to take away the previous image 
    # so look 2 hrs behind
    hi_files = hip.find_hi_files(time - timedelta(hours=2), time,
                                 craft=craft, camera=camera,
                                 background_type=background_type)
    if len(hi_files) == 0:
        # Try 5 mins later, should avoid errors with the seconds being wrong
        # this is okay as cadence ~40 mins
        hi_files = hip.find_hi_files(time - timedelta(hours=2),
                                     time + timedelta(minutes=5),
                                     craft=craft, camera=camera,
                                     background_type=background_type)   
    if len(hi_files) > 1:
        # Loop over the hi_files, make image
        fc = hi_files[len(hi_files)-1]  # current frame files, last in list
        fp = hi_files[len(hi_files)-2]  # previous frame files, 2nd last
        # Make the differenced image
        hi_map = hip.get_image_diff(fc, fp, align=True, smoothing=True)
        # Make it grey
        diff_normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
        out_img = mpl.cm.gray(diff_normalise(hi_map.data), bytes=True)
        out_img = Image.fromarray(out_img)
        if save != False:
            # Save the image
            out_name = "_".join([helcats_name, craft,
                                 hi_map.date.strftime('%Y%m%d_%H%M%S')])+'.jpg'
            out_path = os.path.join(save, out_name)
            out_img = out_img.convert(mode='RGB')
            out_img.save(out_path)    
    return out_img, hi_map


# example
img, hi_map = make_diff_img('HCME_B__20081212_01', datetime(2008, 12, 13, 00, 49, 36))
img.show()
# frame times on the hard drive - can edit to produce all frames between
# start and end time

# to convert el, pa coords to pixel coords
# inputs need to have astropy units
pix_coords = hip.convert_hpr_to_pix(6*u.deg, 264*u.deg, hi_map)
print(pix_coords.x, pix_coords.y)
