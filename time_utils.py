'''
Takes in times in UTC string format, outputting the graph of a certain RR Lyrae from Sesar 2010 along with lines corresponding to those times' phases 
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.time import Time


rrLyraeID = 100
#times: 15:32-07-02-2021, 05:32-07-02-2021, 14:32-07-02-2021, 12:32-07-02-2021, 10:32-07-02-2021
'''
Gets times from user via command line 
'''
def getTimes(): 
    times = []
    print("Please enter times in the format 00:00-MM-DD-YYYY. Enter 'x' to quit\n")
    current = input("Input a time: ")
    times.append(current)

    while current != 'x': 
        current = input("Input a time: ")
        times.append(current)
    
    print("\n\n")

    times.pop()
    return times

'''
Converts a list of UTC times to MJD 
Returns a list of MJD times with the same dimensions as the input list 
'''
def utcToMJD(inputTime): 
    strComponents = [i.split('-') for i in inputTime]
    print(strComponents)
    newTimeStr = [i[3] + '-' + i[1] + '-' + i[2] + 'T' + i[0] for i in strComponents]

    print(newTimeStr)


    inputTime = Time(newTimeStr, format = 'isot', scale = 'utc')


    return inputTime.mjd

'''
Converts a singular input MJD time to a phase value given a star's period of pulsation and an optional phi0 phase shift value (assumed to be default zero)
'''
def MJDToPhase(inputTime, period, phi0 = 0): 
    
    return (inputTime % period) + phi0 

if __name__ == '__main__': 


    timesInUTC = getTimes()
    timesInMJD = utcToMJD(timesInUTC)

    mpl.style.use('ggplot')
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]) 




    # Get the first lightcurve id
    from gatspy import datasets
    templates = datasets.fetch_rrlyrae_templates() 
    #print(len(templates))
    period = 0.8

    rrlyrae = datasets.fetch_rrlyrae()


    fig, ax = plt.subplots()
    # Graphing the light curve
    for band in 'ugriz':
        phase, normed_mag = templates.get_template(str(rrLyraeID) + band)
        ax.plot(phase, normed_mag, label=band)


    leg1 = ax.legend(loc='lower center', ncol=5, numpoints=1)
    lines = []

    # Graphing the time lines 
    for i in range(len(timesInUTC)): 
        currentUTC = timesInUTC[i] 
        phase = MJDToPhase(timesInMJD[i], period)
        lines.append(ax.axvline(x=phase))

    leg2 = ax.legend(lines, timesInUTC, loc='upper right')
    ax.add_artist(leg1)


    ax.set(xlabel='phase', ylabel='mag')
    ax.invert_yaxis()

    plt.show()  



'''

from gatspy import datasets
rrlyrae = datasets.fetch_rrlyrae()

# Select data from the first lightcurve
lcid = rrlyrae.ids[0]
t, mag, dmag, bands = rrlyrae.get_lightcurve(lcid)
period = rrlyrae.get_metadata(lcid)['P']
phase = (t / period) % 1

# Plot the result
fig, ax = plt.subplots()
for band in 'ugriz':
    mask = (bands == band)
    ax.errorbar(phase[mask], mag[mask], dmag[mask], label=band,
                fmt='.', capsize=0)
ax.set(xlabel='time (MJD)', ylabel='mag',
       title='lcid={0}'.format(lcid))
ax.invert_yaxis()
ax.legend(loc='upper left', ncol=5, numpoints=1)

'''