'''
AST-11 Lightcurve generation code -- SIP 2021

This code contains crossmatching functionality as well as generation functionality using filter transformations from PS1 to CFHT MegaCam band measurements. If you just want to use this code for
generation, I suggest importing into a separate file and either invoking generateLightCurve (which performs individual mock LC generation) or generate_dataset (which performs dataset-level mock LC generation). 

There are various commented out sections of the code. Some of them are print statements for debugging, others are crossmatching code that was commented to test generation functionality, and still others are
graphing functionality. Overall, they have been commented out for cleaner output and faster runtime. 

Refer to method documentation for more details. 

@author Kailash Ranganathan
@date created 7.22.2021
@date (last updated) 7.26.2022

'''
from time_utils import MJDToPhase
from astropy import coordinates
from astropy import units as u
from astropy.coordinates import match_coordinates_sky, SkyCoord, ICRS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from rrlyrae_generation import * 
from astropy.table import Table
from astropy.io import fits
from scipy import stats
from gatspy import datasets 
import random
import csv 

import os
import gc

'''
Expected input: Pandas dataframe of RA & DEC values 

output: list of SkyCoord objects 
'''
def HHMMSStoCoords(table): 
    coords = []
    for _, row in table.iterrows(): 
        startRA, startDEC = row['ra'], row['dec']

        splittedRA = startRA.split(':')
        splittedDEC = startDEC.split(':')

        newFormatRA = splittedRA[0] + 'h' + splittedRA[1] + 'm' + splittedRA[2] + 's'
        newFormatDEC = splittedDEC[0] + 'd' + splittedDEC[1] + 'm' + splittedDEC[2] + 's' 

        coords.append(SkyCoord(newFormatRA, newFormatDEC, frame='icrs'))
    return coords 



# Catalog data 


#Print the matches and whether they fall within a certain distance of the input coordinates 
#print(totalMatched) # Boolean array 
#print(totalMatches) # Matched data coordinates 
#print(indices) # IDs of matches 

# Write the lightcurve plots to a PDF using PdfPages

#from matplotlib.backends.backend_pdf import PdfPages
#pp = PdfPages('PS1LightcurvePlotsWithNGVSTransformations.pdf')

'''
for index in indices: 
    plt.figure()
    plt.clf()
    rrLyraeIDs = {'g': a['Tg'][index], 'r': a['Tr'][index], 'i': a['Ti'][index], 'z': a['Tz'][index]}
    amps = {'g': gAmp[index], 'r': a['rAmp'][index], 'i': iAmp[index], 'z': a['zAmp'][index]}
    mags = {'g': gmag_sesar[index], 'r': a['rmag'][index], 'i': imag_sesar[index], 'z': a['zmag'][index]}

    for band in 'griz': 
        phase, normed_mag = templates.get_template(str(rrLyraeIDs[band]) + band)
        plt.plot(phase, normed_mag*amps[band] + mags[band], label=band)
    plt.legend(loc='lower center', ncol=5, numpoints=1)
    plt.title(f"PS1 ID {index} light curve")
    plt.gca().invert_yaxis()

    #pp.savefig()
'''

import utilipy.astro.instruments.filtertransforms.MegaCam_PanSTARRS as util
from astropy.table import Table, Column

# Construct astropy table input for filter transformations 

#print(util.U_MP9301(table, g='g', i='i', gmi='g-i'))
#print(util.G_MP9401(table))

#pp.close()


# Filter transforms 
def U_CFHT(gBand, gmi): 
    return gBand + .523 - .343 * gmi + 2.44 * np.square(gmi) - .998 * np.power(gmi, 3)

def G_CFHT(gBand, gmi): 
    return gBand -.001 - .004* gmi - .0056 * np.square(gmi) + .00292 * np.power(gmi, 3)
    

def I_CFHT(iBand, gmi): 
    return iBand -.005 + .004 * gmi + .0124 * np.square(gmi) - .0048 * np.power(gmi, 3)


def Z_CFHT(zBand, gmi): 
    return  zBand -.009 - .029 *gmi + .012 * np.square(gmi) - .00367 * np.power(gmi, 3)

# Given gizData (in dictionary form), output the ugiz filter transformations 
def totalFilterTransform(gizData): 
    out = {}
    gmi = gizData['g'] - gizData['i'] 
    out['u'] = U_CFHT(gizData['g'], gmi)
    out['g'] = G_CFHT(gizData['g'], gmi)
    out['i'] = I_CFHT(gizData['i'], gmi)
    out['z'] = Z_CFHT(gizData['z'], gmi)
    return out 



plotting_colors = {'u': '#800080', 'g': '#00873E', 'i': "#FFAE42", 'z': '#B80F0A'}

'''
Relative distance modulus shifting of a light curve (given by parameters mags and phases) to a given g_AV (average g_band magnitude)

@return the shifted mags and corresponding phases of each magnitude *sorted chronologically by phase*
'''
def RDM_shifted(mags, phases, g_AV = 24): 
    sorted_mags = {}
    #print(f"Phases are {phases}")
    
    sorted_phases = None
    #print(f"Sorted phases are {sorted_phases}")
    for band in 'ugiz': 
        if band == 'u': 
            sorted_phases = [x for x, _ in sorted(zip(phases, mags[band]), key = lambda pair: pair[0])]
        sorted_mags[band] = [x for _, x in sorted(zip(phases, mags[band]), key = lambda pair: pair[0])]
        
    
    g_mags = sorted_mags['g']
    sumG = 0 
    for i in range(len(g_mags)-1):
        sumG += g_mags[i]*(sorted_phases[i+1] - sorted_phases[i])
    averageG = float(sumG)/(max(sorted_phases) - min(sorted_phases))
    RDM = g_AV - averageG     
    
    for band in 'ugiz': 
        sorted_mags[band] += RDM 
        #plt.plot(sorted_phases, sorted_mags[band], color = plotting_colors[band], label = band)
    
    return sorted_mags, sorted_phases



    #sorted_mags = [x for _, x in sorted(zip(phases, mags), key = lambda pair: pair[0])]
    
'''
Given a cadence list DataFrame (parameter "data"), interpolate magnitudes from the lightcurve defined by phase & mag and return 
these interpolated magnitudes along with the list of measured phases. 

@param phase, mag the list of phases and corresponding magnitudes of the current light curve
@param band the band to sample from (this method only supports interpolation of one band at a time)

@param data the cadence list of measurement times given in dataFrame format
@param graph not used, but meant to be a toggle for whether the interpolated LC should be graphed or not
@param phi phase shift option 
'''
def return_interpolated_data(phase, mag, band, data, graph = 0, phi = 0):
   measurement_phases = (data["time_mjd"][data["band"]==band].values+phi)%1

   #PRECONDITION - CURRENT PHASE AND MAGS ARE ALREADY SORTED 
   curr_phase = phase
   curr_mags = mag[band]

   #Fun sorting algorithm - sorts phases and bands by increasing phase while keeping the bands on the same index
   
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
   

   return measurement_phases, interpolated_mags


# Read in cadence data from csv file 
#data = pd.read_csv("cadenceListTwo.csv")

# For every crossmatched LC, do the filter transformations, read in the cadence list data, perform RDM shifting, interpolate the LC based on the cadence list, apply photometric error, and 
# plot/save the results
# This code is the exact same as what happens in the generateLightCurve method. Uncomment if you want a demo 
'''
for index in indices: 
    
    if a['S3ab'][index] != 1: 
        print("Not type ab!")
        continue

    G_AV = 24
    print(f"Graphing {index}")
    plt.figure()
    plt.clf()
    rrLyraeIDs = {'g': a['Tg'][index], 'r': a['Tr'][index], 'i': a['Ti'][index], 'z': a['Tz'][index]}
    amps = {'g': gAmp[index], 'r': a['rAmp'][index], 'i': iAmp[index], 'z': a['zAmp'][index]}
    mags = {'g': gmag_sesar[index], 'r': a['rmag'][index], 'i': imag_sesar[index], 'z': a['zmag'][index]}
    
    for band in 'griz': 
        phasePerBand[band], magsPerBand[band] = templates.get_template(str(rrLyraeIDs[band]) + band)
    phases = (phasePerBand['g'] + a['phi0'][index])%1
    CFHT_mags = totalFilterTransform(mags) 
    for band in 'griz': 
        magsPerBand[band]  = magsPerBand[band]*amps[band]
        
    CFHT_magsPerBand = totalFilterTransform(magsPerBand)
    
    #print(CFHT_mags)
    #print(CFHT_magsPerBand)
    
    data["time_mjd"] = MJDToPhase(data["time_mjd"], a['Per'][index], phi0 = a['phi0'][index])
    

    for band in 'ugiz': 
        CFHT_magsPerBand[band] += CFHT_mags[band]
        if band != 'r': 
            #plt.plot(phases, CFHT_magsPerBand[band], f"{plotting_colors[f'{band}CFHT']}",label = f"{band}CFHT")
            continue 

    CFHT_magsPerBand, phases = RDM_shifted(CFHT_magsPerBand, phases, g_AV = G_AV)
    interp_mags, interp_phases, real_mags = {}, {}, {}
    
    for band in 'ugiz': 
        interp_phases[band], interp_mags[band] = return_interpolated_data(phases, CFHT_magsPerBand, band, data, phi = a['phi0'][index])
        #plt.plot(interp_phases[band], interp_mags[band], 'o')
        error = get_photometric_error(band, interp_mags[band])
        #plt.errorbar(interp_phases[band], interp_mags[band], error, marker = 's', ls = 'none', color = plotting_colors[band])
        real_mags[band] = np.array(interp_mags[band]) + np.array(error)*np.random.normal(0, 1, size=len(error))
        plt.scatter(interp_phases[band], real_mags[band], color = plotting_colors[band])
        
    plt.legend(loc='lower center', ncol=5, numpoints=1)
    plt.title(f"PS1 ID {index} light curve (RRab) shifted to g_avg = {G_AV}")
    plt.gca().invert_yaxis()
    plt.show()
    break
    
    # Uncomment the following lines if you want the light curve data to be saved to the PDF 
    # At this point, the time series data could also be read out to a file, csv, etc. 
    #pp.savefig()

    #pp.savefig()
'''


'''
Generates a dataset of mock LC curves given a cadence list file, PS indices, and a configuration of a number of bins. Each bin corresponds to a range of 
g_shift values for RDM shifting, and LC curves within each bin are generated according to the given cadence list file, with the templates being chosen at random from the given
PS indices. The number of light curves to generate per bin is given by nPerBin. Ultimately, these mocks are then written out to a CSV file. 

Each mock light curve in the CSV file has a BIN number (corresponding to what bin it's in) and a BINLCINDEX number (corresponding to what # LC it is in that given bin). 

For some reason, the CSV writer I'm using puts spaces in between individual lines. Haven't resolved that issue

@return 1 if the generation proceeded smoothly, 0 if there was an I/O Error during writing 
'''
def generate_dataset(cadenceListFile, PSIndices, outputFilename = 'dataset.csv', nBins = 6, gMin = 19, gMax = 24, nPerBin=300): 
    templates = datasets.fetch_rrlyrae_templates() # Gatspy templates 
    g_shifts = []
    bins = np.linspace(gMin, gMax, nBins, endpoint = False) # Left endpoint of each bin
    
    deltaBin = abs(gMax - gMin)/nBins # Range each bin covers

    for bin in bins: 
        g_shifts_in_bin =  np.random.rand(nPerBin)*deltaBin + bin
        #addition = np.full(nPerBin, i)
        g_shifts.append(g_shifts_in_bin)

    cadence_data = pd.read_csv(cadenceListFile)

    generated_dataset = []
    columns = ['BIN', 'BINLCINDEX', 'GSHIFT', 'PSINDEX', 'PHASE', 'BAND', 'MAG']

    binCount = 1
    
    LCIndices = []
    g_shift_values = []
    for g_shift_bin in g_shifts: 
        
        binLCCount = 1
        print(f"Bin {binCount} generation")
        for g_shift in g_shift_bin: 
            current_index = random.choice(PSIndices)
            phases, mags = generateLightCurve(cadence_data, current_index, g_shift, templates)
            for band in mags: 
                #print(band)
                #print(mags[band])
                for phase, mag in zip(phases[band], mags[band]): 
                    generated_dataset.append({'BIN': binCount, 'BINLCINDEX': binLCCount, 'GSHIFT': g_shift, 'PSINDEX': current_index, 'PHASE': phase, 'BAND': band, 'MAG': mag})
            


            binLCCount += 1
        binCount +=1
    
    try:

        with open(outputFilename, 'w') as csv_file: 
            writer = csv.DictWriter(csv_file, fieldnames = columns)
            writer.writeheader()

            for entry in generated_dataset: 
                writer.writerow(entry)
    except IOError: 
        print("I/O Error when writing CSV file for dataset")
        return 0
    return 1




'''
Method to perform an individual mock light curve generation. Given a cadence list, PS index, and desired g_shift, this method performs the various steps of LC generation: 
1. Filter transformations from PS1 griz to CFHT ugiz 
2. RDM shifting to the desired average g-band magnitude 
3. Interpolation of magnitudes based on input cadence list measurement times
4. Application of photometric error given source error data for each band

Returns the phases and magnitudes per band of the mock light curve in dictionary form. So, phases['u'] corresponds to all the light curve phase points for the u-band, and similarly for mags['u']

'''
def generateLightCurve(cadenceList, index, g_shift, templates): 
    
    #print(f"Graphing {index}")
    #plt.figure()
    #plt.clf()

    # Get the RR Lyrae data corresponding to this index 
    rrLyraeIDs = {'g': a['Tg'][index], 'r': a['Tr'][index], 'i': a['Ti'][index], 'z': a['Tz'][index]}
    amps = {'g': gAmp[index], 'r': a['rAmp'][index], 'i': iAmp[index], 'z': a['zAmp'][index]}
    mags = {'g': gmag_sesar[index], 'r': a['rmag'][index], 'i': imag_sesar[index], 'z': a['zmag'][index]}
    phasePerBand = {}
    magsPerBand = {}
    CFHT_magsPerBand = {}
    
    # Read data from templates and perform filter transformations 
    for band in 'griz': 
        phasePerBand[band], magsPerBand[band] = templates.get_template(str(rrLyraeIDs[band]) + band)
    phases = (phasePerBand['g'] + a['phi0'][index])%1
    CFHT_mags = totalFilterTransform(mags) 
    for band in 'griz': 
        magsPerBand[band]  = magsPerBand[band]*amps[band]
    #print(phasePerBand, magsPerBand)
        
    CFHT_magsPerBand = totalFilterTransform(magsPerBand)
    
    #print(CFHT_mags)
    #print(CFHT_magsPerBand)
    
    cadenceList["time_mjd"] = MJDToPhase(cadenceList["time_mjd"], a['Per'][index], phi0 = a['phi0'][index])
    

    for band in 'ugiz': 
        CFHT_magsPerBand[band] += CFHT_mags[band]
        
        if band != 'r': 
            #plt.plot(phases, CFHT_magsPerBand[band], f"{plotting_colors[f'{band}CFHT']}",label = f"{band}CFHT")
            continue 
    CFHT_magsPerBand, phases = RDM_shifted(CFHT_magsPerBand, phases, g_AV = g_shift)
    interp_mags, interp_phases, real_mags = {}, {}, {}
    
    for band in 'ugiz': 
        interp_phases[band], interp_mags[band] = return_interpolated_data(phases, CFHT_magsPerBand, band, cadenceList, phi = a['phi0'][index])
        #plt.plot(interp_phases[band], interp_mags[band], 'o')
        error = get_photometric_error(band, interp_mags[band])
        #plt.errorbar(interp_phases[band], interp_mags[band], error, marker = 's', ls = 'none', color = plotting_colors[band])
        real_mags[band] = np.array(interp_mags[band]) + np.array(error)*np.random.normal(0, 1, size=len(error))
        
        #plt.scatter(interp_phases[band], real_mags[band], color = plotting_colors[band])
    

    # Uncomment if you want to print each generated LC 
    '''
    plt.legend(loc='lower center', ncol=5, numpoints=1)
    plt.title(f"PS1 ID {index} light curve (RRab) shifted to g_avg = {g_shift}")
    plt.gca().invert_yaxis()
    plt.show()
    '''
    return interp_phases, real_mags
#Use these bands - U_MP9301, G_MP9401, I_MP9702, Z_MP9801


# Upon being called, perform crossmatching and generate mock LC dataset with resulting PS indices
if __name__ == '__main__': 
    # Read in list of coordinates
    RAandDEC = pd.read_csv('RRL_mon RA & Dec - Sheet1.csv')

    # Convert coordinates to SkyCoord objects
    coords = HHMMSStoCoords(RAandDEC)
    print(RAandDEC)

    f1 = fits.open('asu_full.fit')
    a = f1[1].data
    ra_RRsesar, dec_RRsesar, Sab, Sc, Per, gAmp, iAmp, gmag_sesar, imag_sesar, DM= a['RAJ2000'], a['DEJ2000'], a['S3ab'], a['S3c'], a['Per'],\
        a['gAmp'], a['iAmp'], a['gmag'], a['imag'], a['DM']

    catalog = SkyCoord(ra_RRsesar,dec_RRsesar,unit='deg')

    totalMatched = []
    totalMatches = []
    indices = []

    # Cross matching 
    for c in coords: 
        idx, d2d, d3d = c.match_to_catalog_sky(catalog) 
        distances = np.asarray(d2d)
        totalMatched.append(distances < 0.000138889)
        totalMatches.append(catalog[idx])
        indices.append(int(idx))
    
    generate_dataset("cadenceListTwoRevised.csv", indices, outputFilename='dataset2.csv', nPerBin = 20)





