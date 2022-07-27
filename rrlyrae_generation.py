# AST-03A Light curve generation code - SIP 2020

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd 
import os
import pickle 
from scipy.interpolate import InterpolatedUnivariateSpline


#Matplotlib formatting stuff 
mpl.style.use('ggplot')
mpl.rc('axes', prop_cycle=mpl.cycler(color=["#55A868", "#C44E52",
                            "#8172B2", "#CCB974"]))
color_dict = {'u': "#55A868", 'g':"#C44E52", 'i':"#8172B2", 'z':"#CCB974"}

# Fetch the RRLyrae data
from gatspy import datasets
rrlyrae = datasets.fetch_rrlyrae()

# This is the line that selects which light curve template to use for the 
# current pass
idindex = 20


# __________________ FUNCTION DEFINITIONS START NOW ______________


# Gets the lightcurve data for the given ID, calculates the RDM and 
# the new apparent mags for a fainter star, and returns the phase, new mags, 
# mag errors (not really used), band values for the mags array, the the period of the
# current lightcurve 
def retrieve_lightcurve_data(id, g_AV = 24):
   t, mag, dmag, bands = rrlyrae.get_lightcurve(id)
   g_mags = mag[bands=='g']
   g_time = t[bands=='g']
   sumG = 0
   for i in range(len(g_mags)-1):
         sumG += g_mags[i]*(g_time[i+1]-g_time[i])
   averageG = float(sumG)/(max(g_time) - min(g_time))
   
   RDM = g_AV - averageG       
   new_ugiz = mag[bands!='r'] + RDM
   period = rrlyrae.get_metadata(id)['P']
   phase = (t[bands!='r']% period) / period
   return phase, new_ugiz, dmag[bands!='r'], bands[bands!='r'], period


# Returns the interpolated mags for the given phases in time_list.csv for the current band
# based on the band's data taken from the RR Lyrae lightcurve template data. Uses linear interpolation
# on the phase's nearest neighbors and gets a mag value for each input phase 
# 
# Note - this method only requires the band you want from the time_list.csv - it gets the phases
# by itself. 
def return_interpolated_data(phase, mag, bands, band, data, graph = 0, phi = 0):

   measurement_phases = (data["time_mjd"][data["band"]==band].values+phi)%1
   
   curr_phase = phase[bands==band]
   curr_mags = mag[bands==band]

   #Fun sorting algorithm - sorts phases and bands by increasing phase while keeping the bands on the same index
   tuples = [(curr_phase[i], curr_mags[i]) for i in range(len(curr_phase))]
   tuples.sort(key = lambda x: x[0])  
   curr_phase= [i[0] for i in tuples]
   curr_mags = [i[1] for i in tuples]
   interpolated_mags = []

   '''Interpolation loop, for each phase, find the two template 
      phases from the real light curve that "sandwich" it and interpolate
      based on those two phases' mag values to find a magnitude for the phase we want. 
      Repeat this for each input phase and return an array of the measured phases for the 
      current band and their respective interpolated mag values. 
   '''
   for measured in measurement_phases:
      index = np.argmax(curr_phase > measured)
      #print(f"Index is {index}")
      
      sandwiched_phases = [curr_phase[index-1], curr_phase[index]]
      if index == 0:
         sandwiched_phases[0]-=1
         if measured > curr_phase[index]:
            measured = measured - 1
      
      #print(sandwiched_phases[0], measured, sandwiched_phases[1])
      rangeLin = abs(sandwiched_phases[1] - sandwiched_phases[0])
      
      sandwiched_mags = (curr_mags[index-1], curr_mags[index])
      
      
      interpolated_mag = sandwiched_mags[0]*abs(sandwiched_phases[0]-measured)/rangeLin  + sandwiched_mags[1]*abs(sandwiched_phases[1]-measured)/rangeLin
      interpolated_mags.append(interpolated_mag)
   #for i in range(len(curr_phase)):
      #print(f"Phase {tuples[i][0]} for mag {tuples[i][1]}")
   #print(len(measurement_phases), len(interpolated_mags))
   
   
   # Graphs the interpolated mags if you want 
   if graph == 1:
      plt.scatter(*zip(*tuples))
      plt.scatter(measurement_phases, interpolated_mags)
      plt.show()

   return measurement_phases, interpolated_mags


# Returns the interpolated/extrapolated photometric error for the given apparent magnitudes
# using UnivariateSpline fitted to the given band's photometric error vs mags data
def get_photometric_error(band, mags):
   
   filename = f"{band}_err.csv"
   error_table = pd.read_csv(filename)
   u_mag = np.array(error_table["Mag"].values)
   u_err = np.array(error_table["Err"].values)
   #print(u_mag, u_err)
   #print(mags)
   
   total_mags = np.concatenate([u_mag, mags])
   
   tlc = InterpolatedUnivariateSpline(u_mag, u_err, k=1, check_finite=True)(total_mags)
   #Graphing code commented out - can be used for debugging purposes
   #plt.plot(u_mag, u_err, 'bo')
   #plt.plot(total_mags, tlc, 'go')
   #plt.show()
   tlc = InterpolatedUnivariateSpline(u_mag, u_err, k=1)
   
   # Returns sus values? Not sure tho 
   return [tlc(mags[i])for i in range(len(mags))]


# ___________ FUNCTION DEFINITIONS END HERE  _______________

# __________ STARTING DRIVER CODE _____________

if __name__ == '__main__': 

   '''Eventually, this should be for range(0, 483), but I haven't
      ran the entire simulation yet. For each lightcurve/phi combination, 
      it outputs the light curve values as a dictionary, with each key in the dictionary
      being one band of u, g, i, and z and each band having two arrays corresponding to the phase
      and mag values that can be accessed at "PHASES" and "MAGS" (it's like a 2d dictionary)
      
      The dictionaries are written to binary files using pickle. 
   '''
   columns = ['ID', 'BAND', 'TIME', 'MAG', 'ERROR', 'PHI_0']
   export_data = []
   data = pd.read_csv("SIP_time_list.csv")

   for idindex in range(0, 483):
      lcid = rrlyrae.ids[idindex]
      phase, mags, dmag, bands, period = retrieve_lightcurve_data(lcid)
      filepath = "SIP_time_list.csv"
      thisData = data.copy(deep=True)
      
      # For each light curve id, iterate over the possible phi values 
      # (with a step of 0.1 rn)
      for phi in np.linspace(0, 0.9, 4):

         actual_times = data["time_mjd"].values+phi*period
         #print(actual_times)
         thisData["time_mjd"] = (data["time_mjd"] % period) / period
         #print(actual_times)
         fig, ax = plt.subplots()


         # For each band (u, g, i, and z), get the interpolated mags, photometric error for 
         # the interpolated mags, and graph them on top of the real lightcurve data. 
         # Note - colors might be a bit screwed up on the graphs lol 
         for band in 'ugiz':
            real_phases, real_mags = return_interpolated_data(phase, mags, bands, band, thisData, graph=0, phi=phi)
            error = get_photometric_error(band, real_mags)
            
            #Uncomment this line if you want to display the template light curve values as well 
            ax.scatter(phase[bands==band], mags[bands==band], marker='.', color=color_dict[band], label=band)

            # Graph the measured phase and mag values along with the error range for each value. 
            #ax.scatter(real_phases, real_mags, marker='s', label=band, color=color_dict[band])
            ax.errorbar(real_phases, real_mags, error, fmt='s', label=band, color=color_dict[band])
            
            real_mags = np.array(real_mags) + np.array(error)*np.random.normal(0, 1, size=len(error))
            ax.scatter(real_phases, real_mags,marker='o', edgecolors=color_dict[band])
            export= [[idindex for _ in range(len(real_mags))], 
                     [band for _ in range(len(real_mags))],
                     actual_times[data["band"]==band],
                     real_mags,
                     error,
                     [phi for _ in range(len(real_mags))]]
            for i in range(len(error)):
               export_data.append([export[count][i] for count in range(len(export))])
            
            
            
            
            

         #Writing out stuff to file 
         print(f"Saving light curve {idindex} with phi value {phi}")

         #Fun matplotlib stuff 
         ax.set(xlabel='time (phase)', ylabel='mag',
            title='lcid={0} generated and template lightcurves for phi value {1}'.format(lcid, phi))
         ax.legend(loc='upper left', ncol=5, numpoints=1)
         ax.invert_yaxis()
         plt.show()
            
   export_data = pd.DataFrame(export_data, columns=columns)
   export_data.to_csv('data.csv')












# Plot the result
'''
fig, ax = plt.subplots()
for band in 'ugiz':
    mask = (bands == band)
    

    ax.errorbar(phase[mask], mag[mask], dmag[mask], label=band,
                fmt='.', capsize=0)
    
    
ax.set(xlabel='time (MJD)', ylabel='mag',
       title='lcid={0}'.format(lcid))
ax.invert_yaxis()
ax.legend(loc='upper left', ncol=5, numpoints=1)
plt.show()'''