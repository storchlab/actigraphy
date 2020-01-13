# -*- coding: utf-8 -*-
"""
This script allows for conversion of actigraphy data from GENEActiv devices 
into inactograms, to better visualize patterns of inactivity accross days.
It is designed to work with binary files extracted with the GENEActiv software.
It can also work with pre-analysed CSV files form the GENEActiv software, but
these files do not contain enough information to calculate sleep bouts, and
thus only activity can be plotted.



Clément Bourguignon, The Storch Lab, McGill University, 2020
MIT License

Copyright (c) 2020 Clément Bourguignon, The Storch Lab, McGill University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def extract_bin(filepath):
    """Extract data from a GENEActiv binary file."""

    # Import the whole file as text
    with open(filepath, 'r') as f:
        data = f.readlines()

    # Only works with GENEActiv devices (at least for now)
    if not data[2].strip().split(':')[-1] == 'GENEActiv':
        raise Exception('Not a GENEActiv device')
        return
    
    # Create a calibration dictionnary from the header
    calibration = {}
    for i in range(47,55):
        tmp = data[i].strip().split(':')
        calibration[tmp[0]]=int(tmp[1])

    # Find number of pages of recording, frequency, and length in lines of a page
    n_pages = int(data[57].strip().split(':')[-1])
    fs = float(data[19].strip().split(':')[-1].split(' Hz')[0])
    block_length = (len(data)-59)//n_pages

    # Initialize numpy arrays for storing raw values
    x = np.empty(n_pages*300)
    y = np.empty(n_pages*300)
    z = np.empty(n_pages*300)
    light = np.empty(n_pages*300)
#    button = np.empty(n_pages*300)
#    temperature = np.empty(n_pages)

    for p in range(n_pages):
        '''
        A hexstring is 300 values encoded as following:
            12 bits for x (signed)
            12 bits for y (signed)
            12 bits for z (signed)
            10 bits for light
            1 bit  for button
            1 bit  for reserved 0
        resulting in a 48-bit or 6-byte block encoded in 12 hexadecimal values,
        so we slice the string every 12 character.
        '''

        hexstring = data[68 + (block_length*p)].strip()
        # temperature[p] = float(data[64 + (block_length*p)].strip().split(':')[-1])

        for i in range(300):
            '''
            For signed values, remove 4096 if the first bit is 1.
            x is in the 12 first bits, we just need to bitshift 36 times.
            y, z, and light are in the middle so retain the last bits using
            bitwise logic after bit-shifting
            '''
            d = int(hexstring[i*12:(i+1)*12],16)

            x[300*p + i] = -4096 * (d >> 47) + (d >> 36)
            y[300*p + i] = -4096 * (d >> 35 & 1) + (d >> 24 & 0xFFF)
            z[300*p + i] = -4096 * (d >> 23 & 1) + (d >> 12 & 0xFFF)
            light[300*p + i] = d >> 2 & 0x3FF
#            button.append(d >> 1 & 1)

#        temperature = temperature.repeat(300)

    start_date = pd.to_datetime(data[62].strip().split('Time:')[1], format='%Y-%m-%d %H:%M:%S:%f')
    timestamps = pd.date_range(start=start_date, periods=n_pages*300, freq=f'{1/fs}S')

#    data_df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'light':light, 'button':button, 'temperature':temperature},
#                           index=timestamps)
    data_df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'light':light}, index=timestamps)

    data_df['x'] = (data_df['x'] * 100 - calibration['x offset']) / calibration['x gain']
    data_df['y'] = (data_df['y'] * 100 - calibration['y offset']) / calibration['y gain']
    data_df['z'] = (data_df['z'] * 100 - calibration['z offset']) / calibration['z gain']
    data_df['light'] = data_df['light'] * calibration['Lux'] / calibration['Volts']

    data_smooth = data_df.rolling('5S').median()

    data_smooth['svmg'] = np.abs(np.sqrt(data_df['x']**2 + data_df['y']**2 + data_df['z']**2) - 1)

    return data_smooth

def extract_csv(filepath):
    data = pd.read_csv(filepath, header=100, index_col=0, parse_dates=True,
                       names=['x', 'y', 'z', 'light', 'button', 'temperature',
                              'svmg', 'x_dev', 'y_dev', 'z_dev', 'peak_lux'])
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S:000', infer_datetime_format=True)

    return data

def calc_angle(data, method='max', win_size=5):
    """"Extract data from a CSV file."""

    data['angle'] = np.arctan(data['x'] / np.sqrt(data['x'] ** 2 + data['y'] ** 2)) * 180 / np.pi

    angle_change = data['angle'].resample(f'5S').mean().diff().abs().rename('angle_change')
    
    win_size *= 12
    
    if method == 'median':
        roll = angle_change.rolling(win_size + 1, center=True).median().rename('rolling')
    elif method == 'max':
        roll = angle_change.rolling(win_size + 1, center=True).max().rename('rolling')

    return data, angle_change, roll

def calc_sleep(roll, thresh_method='fixed', thresh_value=5):
    if thresh_method == 'perc':
        thresh = np.nanpercentile(roll, 10)*15
    elif thresh_method == 'fixed':
        thresh = thresh_value

    sleep = (roll < thresh) * 1
    return sleep.rename('sleep')

def plot_actogram(data, tz_off=-5, binsize=5, doubleplot=1, scale=5, title=1):
    """"Prepare data and plot an inactogram"""

    if not type(data) == pd.core.series.Series:
        print('Wrong data type, please input a pandas Series')
        return

    daylength = 1440 // binsize

    # Extract and resample
    data = data.resample(f'{binsize}T').sum()

    start = data.index[0].timetuple()
    end = data.index[-1].timetuple()

    frontpad = np.zeros((start[3]*60 + start[4]) // binsize)
    backpad = np.zeros(daylength - (end[3]*60 + end[4]) // binsize - 1)

    counts = np.concatenate([frontpad, data.values, backpad])

    ndays = counts.shape[0] // daylength

    # Digitize
    edges = np.histogram_bin_edges(counts, bins=scale)
    perc = (np.digitize(counts, edges) - 1) * (counts > 0)

    # Prepare arrays for vertices ##
    '''
    Create an array of starting and ending points, and repeat it the number
    of days required.
    We want to return to zero after each line, so we add a 1440 to 0 bar.
    '''
    x = np.tile(np.concatenate((np.arange(0, 1440, binsize),np.array([1440]))),ndays) # base
    xx = np.empty(x.size * 2, dtype=x.dtype) # final array to populate
    xx[0::2] = x # startpoints
    xx[1::2] = x + binsize # endpoints
    xx[daylength * 2 + 1::daylength * 2 + 2] = 0 # to go back to 0 instead of going further after each line

    '''
    Create an array for values. We start by calculating where the data must go
    by creating an array of values corresponding to days to start, and add the
    values normalized to 0.85 for better visual. We have to not forget to insert
    an additional day and a zero-value for "traveling back".
    Finally, we need to repeat this array for going form start- to endpoints.
    '''
    yidx = ndays - np.arange(0,ndays).repeat(daylength + 1) - 1
    y = np.concatenate(
        (np.insert(perc,np.arange(daylength,daylength*ndays, daylength),0),
         np.array([0]))) / perc.max() * .85
    yy = np.repeat(y + yidx,2)

    # create figure
    fig, ax = plt.subplots()
    verts = [*zip(xx, yy)]
    poly1 = Polygon(verts, facecolor='k', edgecolor=None, closed=False)
    ax.add_patch(poly1)
    if doubleplot:
        verts = [*zip(xx+1440, yy+1)] # for doubleplotting
        poly2 = Polygon(verts, facecolor='k', edgecolor=None, closed=False)
        ax.add_patch(poly2)
        ax.set_xlim(0,2880)
        ax.set_ylim(0,ndays+1)
    else:
        ax.set_xlim(0,1440)
        ax.set_ylim(0,ndays)
    plt.show()

    if title:
        plt.title(data.name)


if __name__ == "__main__":
    # Create a filedialog to ask for filepath
    from tkinter import filedialog
    from tkinter import Tk
    root = Tk()
    filepath =  filedialog.askopenfilename(initialdir = "./",
                                           title = "Select file",
                                           filetypes = (("binary file","*.bin"),
                                                        ("CSV file","*.csv")))
    root.withdraw()

    if filepath.endswith('bin'):
        picklepath = filepath.replace('.bin', '.pkl')

        if os.path.isfile(picklepath):
            with open(picklepath, 'rb') as p:
                data = pickle.load(p)
        else:
            data = extract_bin(filepath)
            with open(picklepath, 'wb') as p:
                pickle.dump(data, p)
                
        data, angle_change, roll = calc_angle(data, method='max', win_size=5)
        sleep = calc_sleep(roll, thresh_method='fixed', thresh_value=5)

        plot_actogram(sleep, scale=1)

    elif filepath.endswith('csv'):
        data = extract_csv(filepath)
        plot_actogram(data['svmg'], scale=10)  # Sleep cannot be calculated, display activity
    else:
        print('Wrong file type selected')

